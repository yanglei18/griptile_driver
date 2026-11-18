import csv
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from griptile_core.assembler import Assembler
from griptile_core.contact_image import INVALID_RGB, STRONG_RGB, UNLOADED_RGB, WEAK_RGB
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
