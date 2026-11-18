"""Exercise the installed ROS driver, DDS, rosbag2 MCAP, and replay executable.

GRIPTILE_WIRE may select an archived, aligned row44 byte stream instead of the
deterministic CI fixture. This opens only a newly allocated Linux pseudo-terminal.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import time

import rclpy
import rosbag2_py
from griptile_msgs.msg import ContactReference, SensorInfo, TactileGridState, TactileRow44Array
from griptile_msgs.srv import SetContactReference
from sensor_msgs.msg import Image
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message


NS = "/tactile/sensor_0"
QOS_PATH = Path(__file__).resolve().parents[1] / "src/griptile_driver/config/record_qos.yaml"
PACKET = struct.Struct(">2sH20H")


def input_rows():
    archive = os.environ.get("GRIPTILE_WIRE")
    if archive:
        wire = Path(archive).read_bytes()
    else:
        wire = b"".join(PACKET.pack(b"\xa5\x5a", i % 20, *(
            (i * 20 + c) % 65536 for c in range(20))) for i in range(160))
    rows = []
    for header, row, *values in PACKET.iter_unpack(wire):
        assert header == b"\xa5\x5a" and row < 20
        rows.append((row, tuple(values)))
    assert len(set(row for row, _ in rows)) == 20
    return wire, rows


@contextlib.contextmanager
def running(args, log):
    with log.open("wb") as stream:
        child = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            yield child
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                    raise RuntimeError(f"Timed out stopping {args[0:3]}")


def until(node, predicate, seconds=14):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.03)
        if predicate():
            return
    raise AssertionError("Timed out waiting for " + (predicate.__doc__ or repr(predicate)))


def read_bag(bag):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    rows, info = [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == NS + "/samples":
            batch = deserialize_message(data, TactileRow44Array)
            assert batch.producer_dropped_total == 0
            rows.extend((b.row, tuple(b.adc)) for b in batch.blocks)
        elif topic == NS + "/info":
            info.append(deserialize_message(data, SensorInfo))
    return rows, info


def test_sample_bag_roundtrip(tmp_path, monkeypatch):
    # A local ROS domain keeps test commands separate from active devices.
    monkeypatch.setenv("ROS_DOMAIN_ID", str(80 + os.getpid() % 18))
    monkeypatch.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "logs"))
    wire, expected = input_rows()
    grid = dict(expected)
    expected_grid = [value for row in range(20) for value in grid[row]]
    master, slave = os.openpty()
    port = os.ttyname(slave)
    os.close(slave)
    bag = tmp_path / "recording"
    rclpy.init(args=[])
    observer = rclpy.create_node("griptile_roundtrip")
    live, replay, info = [], [], []
    live_images, replay_images = [], []
    observer.create_subscription(Image, NS + "/contact_image", live_images.append, 5)
    observer.create_subscription(Image, "/tactile/replay/contact_image", replay_images.append, 5)
    preview_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=8)
    info_qos = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
    observer.create_subscription(TactileGridState, NS + "/state", live.append, preview_qos)
    observer.create_subscription(SensorInfo, NS + "/info", info.append, info_qos)
    observer.create_subscription(TactileGridState, "/tactile/replay/state", replay.append, preview_qos)
    recorder = ["ros2", "bag", "record", "-s", "mcap", "-o", str(bag),
                "--qos-profile-overrides-path", str(QOS_PATH), "--topics",
                NS + "/samples", NS + "/info"]
    try:
        with running(["ros2", "launch", "griptile_driver", "sensor.launch.py", "port:=" + port,
                      "max_cell_age_ms:=2000"], tmp_path / "driver.log"):
            until(observer, lambda: any(m.connected for m in info))
            with running(recorder, tmp_path / "record.log"):
                until(observer, lambda: observer.count_subscribers(NS + "/samples") >= 1
                      and observer.count_subscribers(NS + "/info") >= 2)
                # Append only the NEXT header, separately counted. The live
                # parser needs lookahead to admit the last real packet.
                stream = wire + b"\xa5\x5a\x00\x00"
                for i in range(0, len(stream), 880):
                    chunk = stream[i:i+880]
                    while chunk:
                        chunk = chunk[os.write(master, chunk):]
                    time.sleep(.002)
                until(observer, lambda: any(all(m.valid_mask) and
                      list(m.adc) == expected_grid for m in live))
                last = next(m for m in reversed(live) if all(m.valid_mask)
                            and list(m.adc) == expected_grid)
                # This input contains no identified unloaded reference.
                # Both paths should render every measured pixel as unknown.
                def unknown_image(frames):
                    return any(frame.header == last.header and frame.encoding == "rgb8"
                               and frame.width == 20 and frame.height == 20 and frame.step == 60
                               and bytes(frame.data) == bytes([128]) * 1200 for frame in frames)

                until(observer, lambda: unknown_image(live_images))
        stored_rows, metadata = read_bag(bag)
        assert stored_rows == expected
        assert any(m.sensor_id == "sensor_0" and m.connected and m.unit == "adc_counts"
                   and m.rows == 20 and m.cols == 20 and not m.has_hardware_timestamp
                   for m in metadata)
        with running(["ros2", "launch", "griptile_driver", "replay.launch.py",
                      "max_cell_age_ms:=2000"], tmp_path / "replay.log"):
            until(observer, lambda: observer.count_subscribers(NS + "/samples") >= 1
                  and observer.count_publishers("/tactile/replay/state") == 1)
            with running(["ros2", "bag", "play", str(bag), "--clock", "100",
                          "--delay", "0.5", "--disable-keyboard-controls"],
                         tmp_path / "play.log") as player:
                until(observer, lambda: any(m.source_sequence == last.source_sequence
                      and all(m.valid_mask) and list(m.adc) == expected_grid
                      and list(m.row_monotonic_ns) == list(last.row_monotonic_ns)
                      and list(m.row_realtime_ns) == list(last.row_realtime_ns)
                      for m in replay))
                until(observer, lambda: unknown_image(replay_images))
                assert player.wait(timeout=8) == 0
            # Only the replay node is still alive. A newly opened viewer must
            # still be able to identify the source of the displayed matrix.
            late_info = []
            observer.create_subscription(SensorInfo, "/tactile/replay/info", late_info.append, info_qos)
            until(observer, lambda: any(m == metadata[-1] for m in late_info))
        report = dict(wire_bytes=len(wire), wire_sha256=hashlib.sha256(wire).hexdigest(),
                      sample_rows=len(expected), adc_values=len(expected)*20,
                      replay_valid_cells=400, all_recorded_rows_equal=True,
                      replay_adc_masks_and_row_clocks_match=True, source_info_recorded=True,
                      replay_info_available_after_player_exit=True,
                      unreferenced_live_and_replay_images_all_unknown=True,
                      synthetic_lookahead_bytes=4, physical_hardware_opened=False,
                      input_kind="archived_serial_bytes" if os.environ.get("GRIPTILE_WIRE") else "synthetic")
        report_file = os.environ.get("GRIPTILE_REPORT")
        if report_file:
            with Path(report_file).open("x") as output:
                json.dump(report, output, indent=2)
                output.write("\n")
    finally:
        os.close(master)
        observer.destroy_node()
        rclpy.shutdown()


def test_recorded_reference_rebuilds_native_contact_image(tmp_path, monkeypatch):
    """A recorder joining after calibration must retain that reference in the bag."""
    import numpy as np
    from griptile_core.contact_image import INVALID_RGB, UNLOADED_RGB, amplitude_rgb

    monkeypatch.setenv("ROS_DOMAIN_ID", str(101 + os.getpid() % 17))
    monkeypatch.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "logs"))
    idle = np.arange(400, dtype=int).reshape(20, 20) + 18000
    pressed = idle.copy()
    pressed[2, 5] += 4400
    pressed[15, 6] += 400

    def wire(matrix):
        return b"".join(PACKET.pack(b"\xa5\x5a", row, *values) for row, values in enumerate(matrix))

    master, slave = os.openpty()
    port = os.ttyname(slave)
    os.close(slave)
    os.set_blocking(master, False)
    rclpy.init(args=[])
    observer = rclpy.create_node("griptile_contact_roundtrip")
    reference_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    refs, states, images, reconstructed = [], [], [], []
    fields, replay_fields = [], []
    observer.create_subscription(Image, NS + "/contact_response", fields.append, 5)
    observer.create_subscription(Image, "/tactile/replay/contact_response", replay_fields.append, 5)
    observer.create_subscription(TactileGridState, NS + "/state", states.append,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
    observer.create_subscription(ContactReference, NS + "/contact_reference", refs.append, reference_qos)
    observer.create_subscription(Image, NS + "/contact_image", images.append, 5)
    observer.create_subscription(Image, "/tactile/replay/contact_image", reconstructed.append, 5)
    capture = observer.create_client(SetContactReference, NS + "/set_contact_reference")
    pending = b""

    def feed(matrix, predicate, description):
        nonlocal pending
        frame = wire(matrix)
        deadline = time.monotonic() + 9
        while time.monotonic() < deadline:
            if not pending:
                pending = frame
            try:
                pending = pending[os.write(master, pending):]
            except BlockingIOError:
                pass
            rclpy.spin_once(observer, timeout_sec=.005)
            time.sleep(.015)
            result = predicate()
            if result:
                return result
        raise AssertionError(description)

    def pixels(message):
        assert (message.encoding, message.height, message.width, message.step) == ("rgb8", 20, 20, 60)
        assert len(message.data) == 1200
        return np.frombuffer(message.data, dtype=np.uint8).reshape(20, 20, 3)

    expected_response = np.zeros((20, 20))
    expected_response[2, 5], expected_response[15, 6] = 4398.5, 398.5
    expected_colors = amplitude_rgb(expected_response, 3000)

    def paired_touches():
        # The RGB and response topics arrive independently. Continue feeding the
        # held load until both halves of one acquisition have been received.
        for image in reversed(images[-8:]):
            if np.array_equal(pixels(image), expected_colors):
                for field in reversed(fields[-8:]):
                    if field.header == image.header:
                        return image, field
        return None

    try:
        bag = tmp_path / "contacts"
        with running(["ros2", "launch", "griptile_driver", "sensor.launch.py", "port:=" + port,
                      "max_cell_age_ms:=2000"], tmp_path / "contact-driver.log"):
            feed(idle, lambda: bool(refs and refs[-1].session_id and images
                                    and np.all(pixels(images[-1]) == INVALID_RGB)
                                    and capture.service_is_ready()), "no unreferenced gray image")
            # Connection is announced before the first valid row is decoded.
            feed(idle, lambda: bool(states and all(states[-1].valid_mask)), "no live rows")
            result = capture.call_async(SetContactReference.Request())
            feed(idle, result.done, "capture request timed out")
            assert result.result().success, result.result().message
            feed(idle, lambda: bool(refs[-1].phase == "ready" and all(refs[-1].valid_mask)
                                    and np.all(pixels(images[-1]) == UNLOADED_RGB)), "no unloaded contact image")
            with running(["ros2", "bag", "record", "-s", "mcap", "-o", str(bag),
                          "--qos-profile-overrides-path", str(QOS_PATH), "--topics",
                          NS + "/samples", NS + "/info", NS + "/contact_reference"], tmp_path / "contact-record.log"):
                feed(idle, lambda: (observer.count_subscribers(NS + "/samples") >= 1 and
                                    observer.count_subscribers(NS + "/contact_reference") >= 2),
                     "recorder not subscribed")
                pressed_image, pressed_field = feed(pressed, paired_touches, "no paired asymmetric contact frame")
                expected = bytes(pressed_image.data)
                assert pressed_field.encoding == "32FC1" and pressed_field.step == 80
                expected_field = bytes(pressed_field.data)
                amplitudes = np.frombuffer(expected_field, dtype="<f4").reshape(20, 20)
                assert (amplitudes[2, 5], amplitudes[15, 6]) == (4398.5, 398.5)
                # Each pose spans several state refreshes on both timelines.
                hold_until = time.monotonic() + .25
                feed(pressed, lambda: time.monotonic() >= hold_until, "could not hold contact")
                release_after = time.monotonic() + .25
                feed(idle, lambda: time.monotonic() >= release_after
                     and np.all(pixels(images[-1]) == UNLOADED_RGB), "release did not clear image")
        with running(["ros2", "launch", "griptile_driver", "replay.launch.py",
                      "max_cell_age_ms:=2000"], tmp_path / "contact-replay.log"):
            until(observer, lambda: observer.count_subscribers(NS + "/contact_reference") >= 2
                  and observer.count_subscribers(NS + "/samples") >= 1)
            with running(["ros2", "bag", "play", str(bag), "--clock", "100", "--rate", "0.5",
                          "--delay", "0.5", "--disable-keyboard-controls"], tmp_path / "contact-play.log"):
                until(observer, lambda: any(bytes(msg.data) == expected for msg in reconstructed))
                until(observer, lambda: any(bytes(msg.data) == expected_field for msg in replay_fields))
                reconstructed.clear()
                until(observer, lambda: any(np.all(pixels(msg) == UNLOADED_RGB) for msg in reconstructed))
    finally:
        os.close(master)
        observer.destroy_node()
        rclpy.shutdown()


def test_known_load_window_uses_the_drivers_unclipped_samples(qt_app, open_window, tmp_path, monkeypatch):
    """Send wire data through the installed launch, zero service and actual Qt hold buttons."""
    import numpy as np
    from griptile_core.contact_image import SURFACE_SCALE, amplitude_rgb, contact_intensity_rgb
    from griptile_viewer.app import ViewerNode
    from griptile_viewer.window import QtWidgets, SURFACE_MODE

    monkeypatch.setenv("ROS_DOMAIN_ID", str(120 + os.getpid() % 17))
    monkeypatch.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "logs"))
    master, slave = os.openpty()
    port = os.ttyname(slave)
    os.close(slave)
    os.set_blocking(master, False)
    rclpy.init(args=["--ros-args", "-r", "__ns:=" + NS])
    node = ViewerNode()
    window = open_window(node.model, node.view_clock, node.request_reference)
    idle = np.full((20, 20), 18000)
    pending = b""
    next_scan = next_draw = 0.0

    def feed(delta, predicate):
        nonlocal pending, next_scan, next_draw
        matrix = idle.copy()
        # With a stable zero, each active taxel subtracts its 1.5-count noise floor.
        if delta:
            sign = 1 if delta > 0 else -1
            matrix[4, 8] += delta + sign * 2
            matrix[12, 3] += delta + sign
        scan = b"".join(PACKET.pack(b"\xa5\x5a", row, *values) for row, values in enumerate(matrix))
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if not pending and now >= next_scan:
                pending = scan
                next_scan = now + 0.02
            if pending:
                try:
                    pending = pending[os.write(master, pending):]
                except BlockingIOError:
                    pass
            rclpy.spin_once(node, timeout_sec=0.002)
            if now >= next_draw:
                window.refresh()
                qt_app.processEvents()
                next_draw = now + 0.033
                if predicate():
                    return
        raise AssertionError(f"Timed out at synthetic ADC {delta}, {window.status.text()}, {window.ref_status.text()}")

    def total_is(value):
        sample = node.model.load_observation(node.view_clock())
        return sample is not None and sample.total_adc == value

    try:
        with running(["ros2", "launch", "griptile_driver", "sensor.launch.py", "port:=" + port],
                     tmp_path / "known-load-driver.log"):
            feed(0, lambda: node.reference_client.service_is_ready() and node.model.mask(node.view_clock()).all())
            window.reference_button.click()
            feed(0, lambda: total_is(0))
            # Exercise the main display with the same installed driver and data a CV
            # subscriber sees, including a value above the display's red endpoint.
            assert window.mode.currentIndex() == SURFACE_MODE and not window.details.isVisible()
            for delta, peak in [(6, "<1%"), (30, "1%"), (3300, "110%"), (0, "0%")]:
                feed(delta, lambda: total_is(2 * delta))
                adc = idle.copy()
                field = np.zeros((20, 20), np.float32)
                if delta:
                    adc[4, 8], adc[12, 3] = 18000 + delta + 2, 18000 + delta + 1
                    field[4, 8], field[12, 3] = delta + .5, delta - .5
                np.testing.assert_array_equal(np.asarray(node.model.message.adc).reshape(20, 20), adc)
                measured = node.model.contact_response(node.view_clock())
                assert measured.dtype == np.float32
                np.testing.assert_array_equal(measured, field)
                center = SURFACE_SCALE // 2
                np.testing.assert_array_equal(window.image.image[center::SURFACE_SCALE, center::SURFACE_SCALE],
                                              amplitude_rgb(field, 3000))
                assert window.peak.text() == f"峰值 {peak}"
                # The driver-calibrated threshold for these two taxels is 3 ADC.
                # A 6-count contact remains colored but has no strong outline seed.
                outlined = window.contour_image.isVisible() and bool(np.any(window.contour_image.image))
                assert outlined == (delta >= 30)
            window.load_button.click()
            dialog = window.load_dialog
            dialog.check_force.setValue(2)
            dialog.calibration_force.setValue(4)
            for delta, next_stage in [(2000, "calibration"), (4000, "check_down"),
                                      (2400, "release"), (-100, "done")]:
                feed(delta, lambda: total_is(2 * abs(delta)) and dialog.capture_button.isEnabled())
                center = SURFACE_SCALE // 2
                pixel = window.image.image[4 * SURFACE_SCALE + center, 8 * SURFACE_SCALE + center]
                assert tuple(pixel) == contact_intensity_rgb(abs(delta) + 0.5, 3000)
                dialog.capture_button.click()
                feed(delta, lambda: dialog.trial is not None and dialog.trial.stage == next_stage)

            assert window.mode.currentIndex() == SURFACE_MODE
            assert dialog.summary.isVisible() and not dialog.results.isVisible()
            assert dialog.readings["check_up"].isVisible() and dialog.readings["check_down"].isVisible()
            dialog.details_button.click()
            feed(2200, lambda: total_is(4400) and dialog.repeat_button.isEnabled())
            assert dialog.repeat_button.isVisible()
            dialog.repeat_button.click()
            feed(2200, lambda: len(dialog.last_report["extra_checks"]) == 1)
            destination = tmp_path / "known-load.json"
            monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a: (str(destination), ""))
            dialog.export_button.click()
            report = json.loads(destination.read_text())
            assert report["holds"]["check_up"]["estimated_n"] == 2
            assert report["holds"]["calibration"]["peak_adc"] > 3000
            assert np.isclose(report["unloading_minus_loading_n"], 0.4)
            assert np.isclose(report["release_residual_n"], 0.1)
            assert report["holds"]["release"]["signed_total_adc"] == -200
            assert np.isclose(report["extra_checks"][0]["error_n"], 0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.close(master)


def test_serial_controls_switch_live_ports_and_clear_the_reference(qt_app, open_window, tmp_path, monkeypatch):
    """Configure an initially idle installed driver from Qt, using two distinct PTYs."""
    import termios
    from griptile_viewer.app import ViewerNode

    monkeypatch.setenv("ROS_DOMAIN_ID", str(138 + os.getpid() % 15))
    monkeypatch.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "logs"))
    with contextlib.ExitStack() as cleanup:
        def make_array(value):
            master, slave = os.openpty()
            cleanup.callback(os.close, master)
            port = os.ttyname(slave)
            os.close(slave)
            os.set_blocking(master, False)
            scan = b"".join(PACKET.pack(b"\xa5\x5a", row, *([value] * 20)) for row in range(20))
            return port, master, scan

        first, master_a, scan_a = make_array(12345)
        second, master_b, scan_b = make_array(23456)
        cleanup.enter_context(running(["ros2", "launch", "griptile_driver", "sensor.launch.py"],
                                      tmp_path / "serial-settings.log"))
        rclpy.init(args=["--ros-args", "-r", "__ns:=" + NS])
        cleanup.callback(rclpy.shutdown)
        node = ViewerNode()
        cleanup.callback(node.destroy_node)
        window = open_window(node.model, node.view_clock, node.request_reference,
                             serial_control=node.serial_control)
        window.serial_button.click()
        dialog = window.serial_dialog
        assert dialog.isVisible()

        def wait_for(description, predicate, master=None, scan=b""):
            deadline = time.monotonic() + 12
            next_update = 0.
            pending = b""
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.002)
                if time.monotonic() < next_update:
                    continue
                next_update = time.monotonic() + .02
                if master is not None:
                    pending = pending or scan
                    try:
                        pending = pending[os.write(master, pending):]
                    except BlockingIOError:
                        pass
                qt_app.processEvents()
                window.refresh()
                if predicate():
                    return
            raise AssertionError(f"{description}: {dialog.status_label.text()}, {window.status.text()}")

        def active_port(path):
            info = node.model.info
            return info is not None and info.connected and info.port == path

        def all_cells(value):
            state = node.model.message
            return (state is not None and state.connected and all(state.valid_mask)
                    and list(state.adc) == [value] * 400)

        def select_port(path, baudrate):
            dialog.port.setEditText(path)
            dialog.baudrate.setCurrentText(str(baudrate))
            wait_for("connect action available", dialog.connect_button.isEnabled)
            dialog.connect_button.click()
            wait_for("serial port opened", lambda: active_port(path))

        wait_for("empty live backend", lambda: node.model.message is not None and not node.model.message.connected)
        assert dialog.baudrate.currentText() == "115200"
        select_port(first, 115200)
        assert termios.tcgetattr(master_a)[4:6] == [termios.B115200] * 2
        wait_for("open port waiting for its first row", lambda: node.serial_control.status.state == "waiting")
        assert not node.model.mask(node.view_clock()).any()
        wait_for("first array", lambda: all_cells(12345) and window.reference_button.isEnabled(),
                 master_a, scan_a)
        assert node.serial_control.status.state == "streaming"
        first_id = node.model.identity
        window.reference_button.click()
        wait_for("first array zero", lambda: node.model.load_observation(node.view_clock()) is not None,
                 master_a, scan_a)

        dialog.port.setEditText(str(tmp_path / "missing-port"))
        assert dialog.connect_button.isEnabled()
        dialog.connect_button.click()
        wait_for("invalid port reported", lambda: bool(node.serial_control.error), master_a, scan_a)
        assert active_port(first) and node.model.identity == first_id and node.model.reference is not None

        select_port(second, 57600)
        wait_for("second array", lambda: all_cells(23456), master_b, scan_b)
        assert termios.tcgetattr(master_b)[4:6] == [termios.B57600] * 2
        assert node.model.identity != first_id and node.model.reference is None
        assert node.model.load_observation(node.view_clock()) is None
        os.close(os.open(first, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK))
        second_id = node.model.identity

        wait_for("disconnect action available", dialog.disconnect_button.isEnabled)
        dialog.disconnect_button.click()
        wait_for("array disconnected", lambda: not node.model.message.connected
                 and not node.model.mask(node.view_clock()).any())
        os.close(os.open(second, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK))
        select_port(first, 115200)
        wait_for("first array reconnected", lambda: all_cells(12345), master_a, scan_a)
        assert node.model.identity not in (first_id, second_id)
        assert node.model.reference is None
