import csv
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from griptile_core.assembler import Assembler
from griptile_core.contact_image import INVALID_RGB, STRONG_RGB, UNLOADED_RGB, WEAK_RGB, contact_surface_rgb
from griptile_core.grid import LAYOUT_ID
from griptile_core.protocol import Block, PROTOCOL_ID
from griptile_core.serial_source import Batch
from griptile_viewer.model import ArrayView
from griptile_viewer.storage import load_reference, save_reference, save_snapshot


SENSOR = "grip_pad"
START = 9_000_000_000


def snapshot(assembler, n, t, rows=range(20), *, epoch=1, offset=0):
    packet = Batch(sensor_id=SENSOR, session_id="fixture", stream_epoch=epoch,
                   sequence=n, first_packet_index=(n - 1) * len(rows), host_realtime_ns=t + 10**12,
                   host_monotonic_ns=t, connected=True, time_valid=True,
                   blocks=tuple(Block(row, tuple(row * 20 + c + offset for c in range(20))) for row in rows))
    assembler.apply(packet)
    state = assembler.snapshot(t, 100_000_000)
    seconds, nanos = divmod(state.host_realtime_ns, 10**9)
    header = SimpleNamespace(frame_id=SENSOR, stamp=SimpleNamespace(sec=seconds, nanosec=nanos))
    return SimpleNamespace(header=header, **asdict(state))


def info(hardware_id="test-board", epoch=1):
    return SimpleNamespace(sensor_id=SENSOR, session_id="fixture", stream_epoch=epoch, hardware_id=hardware_id,
                           layout_id=LAYOUT_ID, protocol_id=PROTOCOL_ID, unit="adc_counts")


def profile_status(*, epoch=1, rows=range(20), phase="ready"):
    return SimpleNamespace(sensor_id=SENSOR, session_id="fixture", stream_epoch=epoch,
                           hardware_id="test-board", phase=phase, span_adc=3000., progress=1.,
                           baseline_adc=list(range(400)), activation_threshold_adc=[3.] * 400,
                           valid_mask=[i // 20 in rows for i in range(400)])


def image(state, pixels):
    return SimpleNamespace(header=state.header, encoding="rgb8", height=20, width=20, step=60,
                           data=pixels.tobytes())


def test_remote_clock_mask_orientation_and_export(tmp_path):
    model = ArrayView(SENSOR)
    assembler = Assembler(sensor_id=SENSOR)
    message = snapshot(assembler, 1, START, (0, 3))
    model.accept(message, 0)
    values = model.values(0)
    assert values[0, 0] == 0
    assert values[3, 7] == 67
    assert np.isnan(values[1, 0])
    assert not model.mask(100_000_001).any()
    assert not model.accept(message, 100_000_002)
    path = tmp_path / "frame.csv"
    save_snapshot(model, path, 100_000_002)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 400
    assert rows[67]["adc_counts"] == "67"
    assert (rows[67]["source_valid"], rows[67]["display_valid"]) == ("1", "0")
    assert rows[67]["row_monotonic_ns"] == str(START)


def test_driver_reference_used_for_signed_delta_and_identity_bound_file(tmp_path):
    model = ArrayView(SENSOR)
    model.accept_reference(profile_status(rows=(2,)))
    model.accept_info(info())
    model.accept(snapshot(Assembler(sensor_id=SENSOR), 1, START, rows=(2,), offset=-39), 0)
    assert model.values(0, True)[2, 4] == -39
    assert np.isnan(model.values(0, True)[0, 0])
    path = tmp_path / "reference.json"
    save_reference(model, path)
    profile = model.reference
    model.accept_info(info("another-board"))
    with pytest.raises(ValueError):
        load_reference(model, path)
    assert model.reference is None
    model.accept_info(info())
    assert load_reference(model, path) == profile
    model.accept_info(info(""))
    with pytest.raises(ValueError):
        save_reference(model, path)


def test_replay_seek_clears_reference_and_history():
    model = ArrayView(SENSOR, replay=True)
    model.accept(snapshot(Assembler(sensor_id=SENSOR), 1, START), 100)
    model.accept_reference(profile_status())
    assert model.reference is not None
    model.track_clock(0)
    assert model.reference is None and not model.trend
    assert not model.mask(0).any()


def test_rgb_pairs_either_arrival_order_and_expires_without_receipt_clock_extension():
    model = ArrayView(SENSOR)
    assembler = Assembler(sensor_id=SENSOR)
    first = snapshot(assembler, 1, START)
    pixels = np.full((20, 20, 3), UNLOADED_RGB, dtype=np.uint8)
    pixels[3, 14] = STRONG_RGB
    model.accept_image(image(first, pixels))
    model.accept(first, 7)
    assert np.all(model.contact_pixels(7) == INVALID_RGB)
    model.accept_reference(profile_status())
    assert np.array_equal(model.contact_pixels(7), pixels)

    second = snapshot(assembler, 2, START + 10_000_000)
    model.accept(second, 10_000_007)
    assert np.array_equal(model.contact_pixels(10_000_008), pixels)
    released = np.full_like(pixels, UNLOADED_RGB)
    model.accept_image(image(second, released))
    assert np.array_equal(model.contact_pixels(10_000_009), released)
    model.accept_image(image(first, pixels))
    assert np.array_equal(model.contact_pixels(10_000_010), released)
    assert np.all(model.contact_pixels(110_000_008) == INVALID_RGB)
    model.accept(snapshot(assembler, 3, START + 20_000_000, epoch=2), 120_000_000)
    assert model.reference is None
    assert np.all(model.contact_pixels(120_000_000) == INVALID_RGB)


def test_qt_native_rgb_png_and_diagnostic_modes(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from griptile_viewer.window import CONTACT_MODE, DELTA_MODE, RAW_MODE, QtGui, QtWidgets, ViewerWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    model = ArrayView(SENSOR)
    state = snapshot(Assembler(sensor_id=SENSOR), 1, START, (0, 2))
    model.accept(state, 0)
    model.accept_reference(profile_status())
    pixels = np.full((20, 20, 3), INVALID_RGB, dtype=np.uint8)
    pixels[[0, 2], :] = UNLOADED_RGB
    pixels[2, 3] = STRONG_RGB
    pixels[2, 7] = WEAK_RGB
    model.accept_image(image(state, pixels))
    requests = []
    window = ViewerWindow(model, clock_ns=lambda: 0, request_reference=requests.append)
    window.mode.setCurrentIndex(CONTACT_MODE)
    window.show()
    app.processEvents()
    model.select(2, 3)
    window.refresh()
    assert window.mode.currentIndex() == CONTACT_MODE
    assert window.image.axisOrder == "row-major"
    assert np.array_equal(window.image.image, pixels)
    assert "接触 2 点" in window.legend.text()
    assert "ADC 43" in window.cell_label.text()
    assert not (window.map.renderHints() & QtGui.QPainter.SmoothPixmapTransform)
    path = tmp_path / "native.png"
    window.save_png(path)
    decoded = QtGui.QImage(str(path))
    assert (decoded.height(), decoded.width()) == (20, 20)
    restored = [[decoded.pixelColor(c, r).getRgb()[:3] for c in range(20)] for r in range(20)]
    np.testing.assert_array_equal(restored, pixels)
    window.mode.setCurrentIndex(RAW_MODE)
    window.refresh()
    assert window.image.image[2, 3] == 43
    assert np.isnan(window.image.image[1, 0])
    window.mode.setCurrentIndex(DELTA_MODE)
    window.refresh()
    assert window.image.image[2, 3] == 0
    window.reference_button.click()
    assert requests == [None]
    assert window.mode.currentIndex() == CONTACT_MODE
    window.close()
    app.processEvents()


def response(state, amplitude, *, bigendian=False):
    data = np.asarray(amplitude, dtype=">f4" if bigendian else "<f4").tobytes()
    return SimpleNamespace(header=state.header, height=20, width=20, encoding="32FC1", step=80,
                           is_bigendian=bigendian, data=data)


@pytest.mark.parametrize("bigendian", [False, True])
def test_response_matches_source_stamp_reference_and_freshness(bigendian):
    model = ArrayView(SENSOR)
    assembler = Assembler(sensor_id=SENSOR)
    state = snapshot(assembler, 1, START, (2, 13))
    field = np.arange(400, dtype=np.float32).reshape(20, 20)
    model.accept_response(response(state, field, bigendian=bigendian))
    model.accept(state, 0)
    assert np.isnan(model.contact_response(0)).all()
    model.accept_reference(profile_status(rows=(2,)))
    native = model.contact_response(0)
    np.testing.assert_array_equal(native[2], field[2])
    assert np.isnan(native[13]).all()
    following = snapshot(assembler, 2, START + 50_000_000, (2, 13))
    model.accept(following, 50_000_000)
    assert np.isnan(model.contact_response(100_000_001)).all()
    model.accept_response(response(following, field + 100))
    assert model.contact_response(100_000_001)[2, 3] == 143
    # Late arrival of the previous image cannot reinstate the old intensity.
    model.accept_response(response(state, field))
    assert model.contact_response(100_000_001)[2, 3] == 143
    assert np.isnan(model.contact_response(150_000_001)).all()


def test_continuous_mode_png_selection_coordinates_and_native_export(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from griptile_viewer.window import CONTACT_MODE, SURFACE_MODE, QtCore, QtGui, QtWidgets, ViewerWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    model = ArrayView(SENSOR)
    assembler = Assembler(sensor_id=SENSOR)
    state = snapshot(assembler, 1, START)
    model.accept(state, 0)
    model.accept_reference(profile_status())
    field = np.zeros((20, 20), dtype=np.float32)
    field[3, 12] = 3000
    field[10, 1] = np.nan
    model.accept_response(response(state, field))
    clock_ns = 0
    requests = []
    window = ViewerWindow(model, clock_ns=lambda: clock_ns, request_reference=requests.append)
    assert window.mode.currentIndex() == SURFACE_MODE
    assert window.mode.currentText() == "柔和接触图（估算）"
    window.show()
    app.processEvents()
    window.refresh()
    assert window.map.renderHints() & QtGui.QPainter.SmoothPixmapTransform
    assert "接触 1 点" in window.legend.text() and "399/400" in window.legend.text()
    original = window.image.image
    assert original.shape == (300, 300, 3)
    np.testing.assert_array_equal(original, contact_surface_rgb(field, model.current_reference.span_adc))
    peak = original[3 * 15 + 7, 12 * 15 + 7]
    assert peak[0] > peak[1] and peak[0] > peak[2]
    # The 300-pixel image still occupies [0,20) on both axes, including after mode switches.
    for row, col in ((0, 0), (19, 19), (3, 12), (17, 2)):
        scene = window.image.mapToScene(QtCore.QPointF((col + .5) * 15, (row + .5) * 15))
        assert window.cell_at(scene) == (row, col)
    path = tmp_path / "surface.png"
    window.save_png(path)
    assert window.image.image is original or np.shares_memory(window.image.image, original)
    decoded = QtGui.QImage(str(path)).convertToFormat(QtGui.QImage.Format_RGB888)
    assert (decoded.width(), decoded.height()) == (300, 300)
    pointer = decoded.constBits()
    pointer.setsize(decoded.byteCount())
    data = np.frombuffer(pointer, np.uint8).reshape(300, decoded.bytesPerLine())[:, :300 * 3]
    np.testing.assert_array_equal(data.reshape(300, 300, 3), original)
    window.reference_button.click()
    assert requests == [None] and window.mode.currentIndex() == SURFACE_MODE
    clock_ns = 100_000_001
    window.refresh()
    assert np.all(window.image.image == INVALID_RGB)
    window.mode.setCurrentIndex(CONTACT_MODE)
    window.refresh()
    assert window.image.image.shape == (20, 20, 3)
    path = tmp_path / "native.png"
    window.save_png(path)
    assert QtGui.QImage(str(path)).size() == QtCore.QSize(20, 20)
    window.close()
    app.processEvents()
