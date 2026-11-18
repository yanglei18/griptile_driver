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
            if predicate():
                return
        raise AssertionError(description)

    def pixels(message):
        assert (message.encoding, message.height, message.width, message.step) == ("rgb8", 20, 20, 60)
        assert len(message.data) == 1200
        return np.frombuffer(message.data, dtype=np.uint8).reshape(20, 20, 3)

    expected_response = np.zeros((20, 20))
    expected_response[2, 5], expected_response[15, 6] = 4398.5, 398.5
    expected_colors = amplitude_rgb(expected_response, 3000)

    def two_touches(message):
        return np.array_equal(pixels(message), expected_colors)

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
                                    and np.all(pixels(images[-1]) == UNLOADED_RGB)), "no unloaded white image")
            with running(["ros2", "bag", "record", "-s", "mcap", "-o", str(bag),
                          "--qos-profile-overrides-path", str(QOS_PATH), "--topics",
                          NS + "/samples", NS + "/info", NS + "/contact_reference"], tmp_path / "contact-record.log"):
                feed(idle, lambda: (observer.count_subscribers(NS + "/samples") >= 1 and
                                    observer.count_subscribers(NS + "/contact_reference") >= 2),
                     "recorder not subscribed")
                feed(pressed, lambda: two_touches(images[-1]), "asymmetric footprint is incorrect")
                pressed_image = images[-1]
                expected = bytes(pressed_image.data)
                until(observer, lambda: any(msg.header == pressed_image.header for msg in fields))
                pressed_field = next(msg for msg in reversed(fields) if msg.header == pressed_image.header)
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
