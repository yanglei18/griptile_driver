import csv
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from griptile_core.assembler import Assembler
from griptile_core.contact_image import (
    CONTACT_COLOR_STOPS, INVALID_RGB, STRONG_RGB, UNLOADED_RGB, WEAK_RGB, amplitude_rgb, contact_surface_rgb,
)
from griptile_core.grid import LAYOUT_ID
from griptile_core.protocol import Block, PROTOCOL_ID
from griptile_core.serial_source import Batch
from griptile_viewer.model import ArrayView
from griptile_viewer.storage import load_reference, save_reference, save_snapshot


SENSOR = "grip_pad"
START = 9_000_000_000


def snapshot(assembler, n, t, rows=range(20), *, epoch=1, offset=0, first_packet_index=None):
    if first_packet_index is None:
        first_packet_index = (n - 1) * len(rows)
    packet = Batch(sensor_id=SENSOR, session_id="fixture", stream_epoch=epoch,
                   sequence=n, first_packet_index=first_packet_index, host_realtime_ns=t + 10**12,
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


def test_qt_native_rgb_png_and_diagnostic_modes(open_window, tmp_path):
    from griptile_viewer.window import CONTACT_MODE, DELTA_MODE, RAW_MODE, QtGui

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
    window = open_window(model, clock_ns=lambda: 0, request_reference=requests.append)
    window.details_button.click()
    model.select(2, 3)
    window.mode.setCurrentIndex(CONTACT_MODE)
    window.refresh()
    assert window.mode.currentIndex() == CONTACT_MODE
    assert window.image.axisOrder == "row-major"
    assert np.array_equal(window.image.image, pixels)
    assert "有响应 2 点" in window.legend.text()
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
    assert not window.scale.isVisible()
    assert window.image.image[2, 3] == 43
    assert np.isnan(window.image.image[1, 0])
    window.mode.setCurrentIndex(DELTA_MODE)
    window.refresh()
    assert window.image.image[2, 3] == 0
    window.reference_button.click()
    assert requests == [None]
    assert window.mode.currentIndex() == CONTACT_MODE


def test_compact_contact_screen_and_reference_action_follow_source_state(qt_app, open_window):
    from griptile_viewer.window import RAW_MODE, SURFACE_MODE

    model = ArrayView(SENSOR)
    window = open_window(model, clock_ns=lambda: 0, request_reference=lambda *_: None)
    size = window.size()
    assert window.load_button.isVisible() and window.scale.isVisible()
    assert window.serial_button.isVisible()
    assert window.brand.isVisible() and window.brand.text() == "GripTile"
    assert window.monitor.isAncestorOf(window.scale) and window.monitor.isAncestorOf(window.map)
    assert window.peak.isVisible() and window.peak.text() == "峰值 —"
    assert not window.reference_button.isVisible() and not window.ref_status.text()
    assert not window.details.isVisible() and not window.map.getAxis("left").isVisible()
    assert not window.marker.isVisible()
    compact_height = window.map.height()
    assert compact_height > window.height() * .75

    model.accept(snapshot(Assembler(sensor_id=SENSOR), 1, START), 0)
    window.refresh()
    assert window.reference_button.isVisible() and window.reference_button.isEnabled()
    model.accept_reference(profile_status())
    window.refresh()
    assert not window.reference_button.isVisible() and "就绪" in window.ref_status.text()

    window.details_button.click()
    qt_app.processEvents()
    assert all(widget.isVisible() for widget in (window.mode, window.curve, window.png_button,
                                                  window.reference_button, window.cell_label))
    assert window.marker.isVisible() and window.map.getAxis("left").isVisible()
    window.mode.setCurrentIndex(RAW_MODE)
    window.refresh()
    assert not window.scale.isVisible()
    window.details_button.click()
    qt_app.processEvents()
    assert window.mode.currentIndex() == SURFACE_MODE and window.scale.isVisible()
    assert not window.curve.isVisible() and not window.marker.isVisible()
    assert window.size() == size and window.map.height() == compact_height


def test_compact_display_fits_the_entire_square_array_when_resized(qt_app, open_window):
    model = ArrayView(SENSOR)
    window = open_window(model)
    for width, height in ((800, 480), (400, 700), (600, 600)):
        window.resize(width, height)
        qt_app.processEvents()
        screen = window.map.viewport()
        field = window.image.mapRectToScene(window.image.boundingRect())
        assert abs(screen.width() - screen.height()) <= 2
        assert abs(field.width() - field.height()) <= 1
        assert .92 * screen.width() <= field.width() <= screen.width() + 1
        assert field.left() >= -1 and field.right() <= screen.width() + 1
        assert field.top() >= -1 and field.bottom() <= screen.height() + 1
        # The acquisition and inspection controls remain reachable in either orientation.
        for button in (window.serial_button, window.load_button, window.details_button):
            assert button.isVisibleTo(window)
            rect = button.rect().translated(button.mapTo(window, button.rect().topLeft()))
            assert window.rect().contains(rect)


def response(state, amplitude, *, bigendian=False):
    data = np.asarray(amplitude, dtype=">f4" if bigendian else "<f4").tobytes()
    return SimpleNamespace(header=state.header, height=20, width=20, encoding="32FC1", step=80,
                           is_bigendian=bigendian, data=data)


@pytest.mark.parametrize("replay", [False, True])
def test_window_collects_contact_print_before_open_and_separates_references(open_window, replay):
    model = ArrayView(SENSOR, replay=replay)
    assembler = Assembler(sensor_id=SENSOR)
    first = snapshot(assembler, 1, START)
    reference = profile_status()
    reference.header, reference.direction = first.header, "both"
    reference.span_adc = 1600.0
    model.accept_reference(reference)
    model.accept(first, 0)
    now = 0
    window = open_window(model, clock_ns=lambda: now)
    for i in range(1, 16):
        now = i * 50_000_000
        state = snapshot(assembler, i + 1, START + now)
        scalar = response(state, np.full((20, 20), 30 * i))
        if i % 2:
            model.accept_response(scalar)
        model.accept(state, now)
        if not i % 2:
            window.refresh()
            model.accept_response(scalar)
        window.refresh()
    window.contact_print_button.click()
    page = window.contact_print_page
    assert page.window() is window and window.pages.currentWidget() is page
    assert page.isVisible() and page.shutter_button.isEnabled()
    page.shutter_button.click()
    assert window.pages.currentWidget() is page and not page.history_button.isChecked()
    saved = model.contact_prints.latest
    assert not saved.automatic
    assert saved.span_adc == 1600.0
    assert saved.stamps_ns == tuple(START + 10**12 + i * 50_000_000 for i in (3, 9, 15))
    assert [frame[0, 0] for frame in saved.frames] == [90, 270, 450]
    # An old scalar cannot continue producing exposures when only the state topic is alive.
    now += 110_000_000
    state = snapshot(assembler, 17, START + now)
    model.accept(state, now)
    window.refresh()
    assert model.contact_prints.latest is saved and not page.shutter_button.isEnabled()
    assert model.contact_prints.contact_present is None and page.history_button.isEnabled()
    if replay:
        now = 0
    else:
        now += 10_000_000
        ended = snapshot(assembler, 18, START + now)
        ended.connected = False
        model.accept(ended, now)
        window.refresh()
        assert model.contact_prints.latest is saved and not model.contact_prints.can_snapshot
        assert model.contact_prints.contact_present is None
        now += 10_000_000
        state = snapshot(assembler, 19, START + now)
        model.accept(state, now)
        reference.header = state.header
        model.accept_reference(reference)
    window.refresh()
    assert model.contact_prints.latest is None
    assert not page.export_button.isEnabled()


def test_print_navigation_and_background_capture_share_one_window(open_window, qt_app):
    model = ArrayView(SENSOR)
    assembler = Assembler(sensor_id=SENSOR)
    first = snapshot(assembler, 1, START)
    ref = profile_status()
    ref.header, ref.direction = first.header, "both"
    model.accept_reference(ref)
    model.accept(first, 0)
    now = 0
    existing_windows = {w for w in qt_app.topLevelWidgets() if w.isVisible()}
    window = open_window(model, clock_ns=lambda: now)
    monitor = window.pages.currentWidget()
    assert window.contact_print_button.isVisible() and not window.contact_print_button.toolTip()
    assert {w for w in qt_app.topLevelWidgets() if w.isVisible()} == existing_windows | {window}
    window.contact_print_button.click()
    page = window.contact_print_page
    assert page.window() is window and not page.isWindow()
    assert page.isVisible() and window.pages.currentWidget() is page
    assert all(not widget.toolTip() for widget in
               (page, page.plate, page.back_button, page.zoom, page.style_combo,
                page.shutter_button, page.history_button, page.export_button))
    page.back_button.click()
    assert window.pages.currentWidget() is monitor and not page.isVisible()

    # A full press and release while watching the monitor must save a print without navigating.
    for i in range(1, 30):
        now = i * 50_000_000
        state = snapshot(assembler, i + 1, START + now)
        model.accept(state, now)
        model.accept_response(response(state, np.full((20, 20), 900 if i <= 20 else 0)))
        window.refresh()
        assert window.pages.currentWidget() is monitor
    saved = model.contact_prints.latest
    assert saved is not None and saved.automatic
    window.open_contact_print()
    assert window.pages.currentWidget() is page and page.history_button.isEnabled()
    assert page.plate.empty_text == "无接触" and not page.history_button.isChecked()
    page.history_button.click()
    assert page.history_button.isChecked() and page.history_button.text() == "返回实时"
    page.back_button.click()
    window.contact_print_button.click()
    assert not page.history_button.isChecked() and page.plate.empty_text == "无接触"
    assert model.contact_prints.latest is saved
    assert {w for w in qt_app.topLevelWidgets() if w.isVisible()} == existing_windows | {window}


def test_hidden_array_waits_for_navigation_or_export_without_stalling_load_tool(open_window, tmp_path):
    from griptile_viewer.window import QtGui

    model = ArrayView(SENSOR)
    assembler = Assembler(sensor_id=SENSOR)
    first = snapshot(assembler, 1, START)
    reference = profile_status()
    reference.header, reference.direction = first.header, "both"
    model.accept_reference(reference)
    model.accept(first, 0)
    now = 0
    window = open_window(model, clock_ns=lambda: now)
    window.open_load_dialog()
    window.open_contact_print()
    assert not window.map.isVisible() and window.load_dialog.isVisible()
    original = window.image.image
    for i in range(1, 16):
        now = i * 50_000_000
        state = snapshot(assembler, i + 1, START + now)
        field = np.full((20, 20), 100. * i)
        model.accept(state, now)
        model.accept_response(response(state, field))
        window.refresh()
    assert window.image.image is original
    assert model.contact_prints.can_snapshot
    assert "峰值 1500 ADC" in window.load_dialog.live.text()

    # Explicit export while the array is hidden still uses the latest sample.
    path = tmp_path / "hidden-array.png"
    window.save_png(path)
    image = QtGui.QImage(str(path))
    assert image.pixelColor(150, 150).getRgb()[:3] == tuple(amplitude_rgb(field, 3000.)[0, 0])
    assert not window.map.isVisible()

    now += 50_000_000
    state = snapshot(assembler, 17, START + now)
    field[0, 0] = 2300.
    model.accept(state, now)
    model.accept_response(response(state, field))
    window.contact_print_page.back_button.click()
    assert window.map.isVisible()
    np.testing.assert_array_equal(window.image.image, contact_surface_rgb(field, 3000.))
    assert "峰值 2300 ADC" in window.load_dialog.live.text()


def test_contact_print_requires_new_rows_and_native_scalar():
    model = ArrayView(SENSOR)
    assembler = Assembler(sensor_id=SENSOR)
    first = snapshot(assembler, 1, START)
    ref = profile_status()
    ref.header, ref.direction = first.header, "both"
    model.accept_reference(ref)
    model.accept(first, 0)
    field = np.full((20, 20), 40)
    rgb = np.full((20, 20, 3), 220, np.uint8)
    for i in range(1, 24):
        now = i * 40_000_000
        state = snapshot(assembler, i + 1, START + now)
        model.accept(state, now)
        model.accept_image(image(state, rgb))
        model.update_contact_print(now)
    assert model.contact_prints.snapshot() is None
    # State timestamps advance, but only the first row is physically scanned again.
    for j in range(24):
        now += 40_000_000
        state = snapshot(assembler, j + 25, START + now, rows=(0,) if j else range(20),
                         first_packet_index=480 if not j else 499 + j)
        model.accept(state, now)
        model.accept_response(response(state, field))
        model.update_contact_print(now)
    partial = model.contact_prints.snapshot()
    assert partial and np.isfinite(partial.frames[-1][0]).all()
    assert not np.isfinite(partial.frames[-1][1:]).any()
    assert not partial.automatic


def test_delayed_viewer_draw_consumes_fresh_queued_print_frames_before_expiring():
    model = ArrayView(SENSOR)
    assembler = Assembler(sensor_id=SENSOR)
    first = snapshot(assembler, 1, START)
    ref = profile_status()
    ref.header, ref.direction = first.header, "both"
    model.accept_reference(ref)
    model.accept(first, 0)
    for i in range(1, 20):
        now = i * 40_000_000
        state = snapshot(assembler, i + 1, START + now)
        model.accept(state, now)
        model.accept_response(response(state, np.full((20, 20), 30 * i)))
        if i not in (11, 12):
            model.update_contact_print(now)
    # The draw after 400 ms happened at 520 ms; all three source samples were still fresh.
    printout = model.contact_prints.snapshot()
    assert printout is not None
    assert printout.frames[0][0, 0] == 120 and printout.frames[-1][0, 0] == 570


def test_empty_source_scan_between_press_and_release_breaks_capture():
    model = ArrayView(SENSOR)
    assembler = Assembler(sensor_id=SENSOR)
    first = snapshot(assembler, 1, START)
    reference = profile_status()
    reference.header, reference.direction = first.header, "both"
    model.accept_reference(reference)
    model.accept(first, 0)
    for i in range(1, 40):
        now = i * 40_000_000
        state = snapshot(assembler, i + 1, START + now)
        if i == 22:
            state.valid_mask = [False] * 400
        model.accept(state, now)
        model.accept_response(response(state, np.full((20, 20), 900 if i < 22 else 0)))
        model.update_contact_print(now)
    assert model.contact_prints.contact_present is False and not model.contact_prints.can_snapshot
    assert model.contact_prints.latest is None


def test_contour_is_a_display_layer_bound_to_the_measured_reference(open_window, tmp_path):
    from griptile_viewer.window import CONTACT_MODE, RAW_MODE, SURFACE_MODE, QtGui

    model = ArrayView(SENSOR)
    assembler = Assembler(sensor_id=SENSOR)
    first = snapshot(assembler, 1, START)
    status = profile_status()
    status.header, status.direction = first.header, "both"
    model.accept(first, 0)
    model.accept_reference(status)
    now = 20_000_000
    pressed = snapshot(assembler, 2, START + now)
    field = np.zeros((20, 20), dtype=np.float32)
    field[4:6, 6:9] = 16
    field[4, 6] = 180
    model.accept(pressed, now)
    model.accept_response(response(pressed, field))
    window = open_window(model, clock_ns=lambda: now)

    def pixels(img):
        rgb = img.convertToFormat(QtGui.QImage.Format_RGB888)
        pointer = rgb.constBits()
        pointer.setsize(rgb.byteCount())
        rows = np.frombuffer(pointer, np.uint8).reshape(rgb.height(), rgb.bytesPerLine())
        return rows[:, :3 * rgb.width()].reshape(rgb.height(), rgb.width(), 3).copy()

    def drawn():
        return np.all(pixels(window.map.viewport().grab().toImage()) == (58, 83, 97), axis=2).any()

    assert window.contour_checkbox.text() == "接触轮廓" and window.contour_checkbox.isChecked()
    assert not window.contour_checkbox.isVisible() and drawn()
    window.details_button.click()
    assert window.contour_checkbox.isVisible()
    window.contour_checkbox.click()
    assert not drawn()
    window.contour_checkbox.click()
    assert drawn()
    path = tmp_path / "surface.png"
    window.save_png(path)
    np.testing.assert_array_equal(pixels(QtGui.QImage(str(path))), contact_surface_rgb(field, status.span_adc))
    np.testing.assert_array_equal(model.contact_response(now), field)
    for mode in (CONTACT_MODE, RAW_MODE):
        window.mode.setCurrentIndex(mode)
        window.refresh()
        assert not drawn()
    window.mode.setCurrentIndex(SURFACE_MODE)
    window.refresh()
    assert drawn()

    # A direct ready-to-ready reference replacement may reuse an image header.
    # Its old measurements cannot supply a boundary using the new noise profile.
    replacement = profile_status()
    replacement.header, replacement.direction = pressed.header, "both"
    model.accept_reference(replacement)
    window.refresh()
    assert not drawn()
    now += 20_000_000
    fresh = snapshot(assembler, 3, START + now)
    model.accept(fresh, now)
    model.accept_response(response(fresh, field))
    window.refresh()
    assert drawn()
    now += 100_000_001
    window.refresh()
    assert not drawn()


@pytest.mark.parametrize("replay", [False, True])
def test_live_native_uses_matched_response_and_recorded_rgb_retains_its_own_colors(open_window, replay):
    from griptile_viewer.window import CONTACT_MODE, SURFACE_MODE

    model = ArrayView(SENSOR, replay=replay)
    assembler = Assembler(sensor_id=SENSOR)
    first = snapshot(assembler, 1, START)
    model.accept(first, 0)
    status = profile_status()
    status.span_adc = 1800.
    model.accept_reference(status)
    original = np.full((20, 20, 3), 255, np.uint8)
    original[0, 0] = (36, 64, 164)
    original[1, 0] = INVALID_RGB
    model.accept_image(image(first, original))
    field = np.zeros((20, 20), dtype=np.float32)
    field[0, 0] = 0.05 * status.span_adc
    field[9, 9] = .01  # Visible strength is near zero; the measured contact is still counted.
    field[1, 0] = np.nan
    # This unpaired field must never color pixels at the earlier acquisition stamp.
    following = snapshot(assembler, 2, START + 40_000_000)
    model.accept_response(response(following, field))
    model.accept_response(response(first, np.full_like(field, np.nan)))
    now = 0
    window = open_window(model, clock_ns=lambda: now)
    window.details_button.click()
    window.mode.setCurrentIndex(CONTACT_MODE)
    window.refresh()
    np.testing.assert_array_equal(window.image.image, original)
    assert "有响应 1 点" in window.legend.text()
    assert window.source_scale.isVisible() and "白色无载" in window.source_scale.text()
    model.accept_response(response(first, field))
    window.refresh()
    native = amplitude_rgb(field, status.span_adc)
    np.testing.assert_array_equal(window.image.image, original if replay else native)
    assert f"有响应 {1 if replay else 2} 点" in window.legend.text() and "399/400" in window.legend.text()
    assert window.scale.isVisible() == (not replay)
    window.mode.setCurrentIndex(SURFACE_MODE)
    window.refresh()
    np.testing.assert_array_equal(window.image.image[7::15, 7::15], native)
    assert window.scale.isVisible() and not window.source_scale.isVisible()
    now = 100_000_001
    window.mode.setCurrentIndex(CONTACT_MODE)
    window.refresh()
    assert np.all(window.image.image == INVALID_RGB)
    # A fresh old-palette image outlives the measured field. Live native must
    # indicate missing data until a new measurement arrives in this acquisition.
    fresh = snapshot(assembler, 3, START + now)
    model.accept(fresh, now)
    model.accept_image(image(fresh, original))
    window.refresh()
    if replay:
        np.testing.assert_array_equal(window.image.image, original)
    else:
        assert np.all(window.image.image == INVALID_RGB)
        assert "0/400" in window.legend.text()
    model.accept_response(response(fresh, field))
    window.refresh()
    np.testing.assert_array_equal(window.image.image, original if replay else native)

    # A different acquisition can legitimately publish only historical RGB.
    now += 1
    replacement = snapshot(assembler, 4, START + now, epoch=2)
    model.accept(replacement, now)
    model.accept_reference(profile_status(epoch=2))
    model.accept_image(image(replacement, original))
    window.refresh()
    np.testing.assert_array_equal(window.image.image, original)


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


def test_continuous_mode_png_selection_coordinates_and_native_export(open_window, tmp_path):
    from griptile_viewer.window import CONTACT_MODE, SURFACE_MODE, QtCore, QtGui, QtWidgets

    model = ArrayView(SENSOR)
    assembler = Assembler(sensor_id=SENSOR)
    state = snapshot(assembler, 1, START)
    model.accept(state, 0)
    status = profile_status()
    status.span_adc = 2000.
    model.accept_reference(status)
    field = np.zeros((20, 20), dtype=np.float32)
    # The decoder and the viewer must use this reference's scale in both contact modes.
    levels = [max(position, 1.e-6) * status.span_adc for position, _ in CONTACT_COLOR_STOPS]
    levels[-1] *= 1.2  # The same color also represents a response above the display range.
    swatches = slice(7, 7 + len(levels))
    field[3, swatches] = levels
    field[10, 1] = np.nan
    native = amplitude_rgb(field, status.span_adc)
    model.accept_image(image(state, native))
    model.accept_response(response(state, field))
    clock_ns = 0
    requests = []
    window = open_window(model, clock_ns=lambda: clock_ns, request_reference=requests.append)
    assert window.mode.currentIndex() == SURFACE_MODE
    assert not (window.map.renderHints() & QtGui.QPainter.SmoothPixmapTransform)
    window.refresh()
    assert window.scale.isVisible()
    assert window.peak.text() == "局部峰值 120%"
    assert f"有响应 {len(levels)} 点" in window.legend.text() and "399/400" in window.legend.text()
    original = window.image.image
    assert original.shape == (300, 300, 3)
    np.testing.assert_array_equal(original[7::15, 7::15], native)
    np.testing.assert_array_equal(native[3, swatches], [rgb for _, rgb in CONTACT_COLOR_STOPS])
    gradient = window.scale.findChild(QtWidgets.QFrame, "contactGradient").grab().toImage()
    for position, rgb in CONTACT_COLOR_STOPS:
        x = round(position * (gradient.width() - 1))
        swatch = gradient.pixelColor(x, gradient.height() // 2).getRgb()[:3]
        np.testing.assert_allclose(swatch, rgb, atol=4)
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
    window.mode.setCurrentIndex(CONTACT_MODE)
    window.refresh()
    np.testing.assert_array_equal(window.image.image, native)
    assert f"有响应 {len(levels)} 点" in window.legend.text() and "399/400" in window.legend.text()
    path = tmp_path / "native.png"
    window.save_png(path)
    assert QtGui.QImage(str(path)).size() == QtCore.QSize(20, 20)
    clock_ns = 100_000_001
    window.mode.setCurrentIndex(SURFACE_MODE)
    window.refresh()
    assert np.all(window.image.image == INVALID_RGB)
    assert window.peak.text() == "峰值 —"


def test_load_observation_uses_same_state_and_unclipped_response_at_all_taxels():
    model = ArrayView(SENSOR)
    assembler = Assembler(sensor_id=SENSOR)
    first = snapshot(assembler, 1, START)
    status = profile_status()
    status.header = first.header
    status.direction = "both"
    status.baseline_adc = [2000.] * 400
    model.accept_reference(status)
    model.accept(first, 0)
    field = np.zeros((20, 20), np.float32)
    field[2, 5], field[15, 6] = 5100, 200
    model.accept_response(response(first, field))
    assert model.load_context is not None
    assert model.load_observation(0) is None  # Exact ref boundary has ambiguous image provenance.

    t2 = 20_000_000
    second = snapshot(assembler, 2, START + t2, offset=2500)
    second.adc = list(second.adc)
    second.adc[15 * 20 + 6] = 1200
    model.accept_response(response(second, field))
    model.accept(second, t2)
    context = model.load_context
    rec = model.load_observation(t2)
    assert rec.context == context and rec.row_stamps == tuple(second.row_monotonic_ns)
    assert (rec.total_adc, rec.peak_adc, rec.signed_total_adc, rec.active_cells) == (5300, 5100, 4900, 2)
    model.accept_reference(status)
    assert model.load_context == context
    # State may arrive before its field. Pairing an old field against new raw ADC
    # would incorrectly flip the first taxel's sign here.
    third = snapshot(assembler, 3, START + 2 * t2)
    model.accept(third, 2 * t2)
    older = model.load_observation(2 * t2)
    assert older.signed_total_adc == 4900 and older.source_now_ns == third.snapshot_monotonic_ns
    model.accept_response(response(third, field))
    assert model.load_observation(2 * t2).signed_total_adc == -5300
    assert model.load_observation(2 * t2 + third.max_age_ns + 1) is None


def test_direct_reference_replacement_waits_until_every_row_crosses_the_new_reference():
    model = ArrayView(SENSOR)
    assembler = Assembler(sensor_id=SENSOR)
    state = snapshot(assembler, 1, START)
    status = profile_status()
    status.header, status.direction = state.header, "increase"
    model.accept(state, 0)
    model.accept_reference(status)
    context = model.load_context

    elapsed = 10_000_000
    second = snapshot(assembler, 2, START + elapsed)
    model.accept(second, elapsed)
    field = np.zeros((20, 20))
    model.accept_response(response(second, field))
    assert model.load_observation(elapsed) is not None
    replacement = profile_status()
    replacement.header, replacement.direction = second.header, "increase"
    model.accept_reference(replacement)
    assert model.load_context != context
    for n, rows in [(3, (0, 3)), (4, range(20))]:
        next_state = snapshot(assembler, n, START + n * elapsed, rows)
        model.accept(next_state, n * elapsed)
        model.accept_response(response(next_state, field))
        if n == 3:
            assert model.load_observation(n * elapsed) is None
        else:
            assert model.load_observation(n * elapsed) is not None
    replaced = model.load_context
    replacement.baseline_adc[7] += 1
    model.accept_reference(replacement)
    assert model.load_context.reference_digest != replaced.reference_digest
    # A late earlier ready event cannot revive a profile for the force estimate.
    model.accept_reference(status)
    assert model.load_context is None
    model.accept_reference(replacement)
    assert model.load_context is not None
    model.accept_info(info("other-board"))
    assert model.load_context is None


def test_incomplete_force_data_never_becomes_zero_and_disconnect_retires_context():
    model = ArrayView(SENSOR)
    assembler = Assembler(sensor_id=SENSOR)
    first = snapshot(assembler, 1, START)
    status = profile_status()
    status.header, status.direction = first.header, "both"
    model.accept(first, 0)
    model.accept_reference(status)
    context = model.load_context
    second = snapshot(assembler, 2, START + 1)
    model.accept(second, 1)
    field = np.zeros((20, 20))
    field[0, 0] = np.nan
    model.accept_response(response(second, field))
    assert model.load_context == context and model.load_observation(1) is None
    third = snapshot(assembler, 3, START + 2)
    third.connected = False
    model.accept(third, 2)
    assert model.load_context is None
    fourth = snapshot(assembler, 4, START + 3)
    model.accept(fourth, 3)
    assert model.load_context is None


def test_live_equalization_changes_only_displayed_contact_strength(open_window, tmp_path):
    from griptile_core.equalization import EqualizationProfile, apply_equalization
    from griptile_viewer.equalization_store import EqualizationSettings
    from griptile_viewer.window import CONTACT_MODE, RAW_MODE, SURFACE_MODE, QtGui

    model = ArrayView(SENSOR)
    assembler = Assembler(sensor_id=SENSOR)
    first = snapshot(assembler, 1, START)
    status = profile_status()
    status.header, status.direction = first.header, "both"
    model.accept_info(info())
    model.accept(first, 0)
    model.accept_reference(status)
    assert model.uniform_observation(0) is None  # Pre-reference rows.
    now = 10_000_000
    pressed = snapshot(assembler, 2, START + now, offset=123)
    model.accept(pressed, now)
    sample = model.uniform_observation(now)
    np.testing.assert_array_equal(sample.raw_adc.ravel(), pressed.adc)
    assert model.uniform_observation(now + pressed.max_age_ns + 1) is None

    amplitude = np.zeros((20, 20), dtype=np.float32)
    amplitude[5:10, 5:10:2] = 300
    amplitude[5:10, 6:10:2] = 150
    model.accept_response(response(pressed, amplitude))
    load = model.load_observation(now)
    window = open_window(model, clock_ns=lambda: now)
    window.equalization = EqualizationSettings(model, tmp_path)
    window.equalization_button.click()
    assert window.equalization_dialog.isVisible()
    window.refresh()
    contour = window.contour_image.image.copy()
    profile = EqualizationProfile("both", (100., 200.) * 200, (150., 300.) * 200)
    window.equalization.adopt(profile, sample.context)
    corrected = apply_equalization(amplitude, profile)
    for mode, render in ((SURFACE_MODE, contact_surface_rgb), (CONTACT_MODE, amplitude_rgb)):
        window.mode.setCurrentIndex(mode)
        window.refresh()
        np.testing.assert_array_equal(window.image.image, render(corrected, status.span_adc))
        assert "8%" in window.peak.text() and "均衡" in window.ref_status.text()
        if mode == SURFACE_MODE:
            np.testing.assert_array_equal(window.contour_image.image, contour)
        path = tmp_path / f"original-{mode}.png"
        window.save_png(path)
        saved = QtGui.QImage(str(path)).convertToFormat(QtGui.QImage.Format_RGB888)
        ptr = saved.constBits()
        ptr.setsize(saved.byteCount())
        rows = np.frombuffer(ptr, dtype=np.uint8).reshape(saved.height(), saved.bytesPerLine())
        np.testing.assert_array_equal(rows[:, :3 * saved.width()].reshape(saved.height(), saved.width(), 3),
                                      render(amplitude, status.span_adc))
    assert model.load_observation(now) == load
    np.testing.assert_array_equal(model.contact_response(now), amplitude)
    window.mode.setCurrentIndex(RAW_MODE)
    window.refresh()
    np.testing.assert_array_equal(window.image.image.ravel(), pressed.adc)
    window.mode.setCurrentIndex(CONTACT_MODE)
    # A ready-to-ready event may replay the same reference while the latest
    # rolling scan still includes older rows. Only post-event rows may use gains.
    restored = profile_status()
    restored.header, restored.direction = pressed.header, "both"
    model.accept_reference(restored)
    now += 10_000_000
    mixed = snapshot(assembler, 3, START + now, rows=(2,), first_packet_index=40)
    model.accept(mixed, now)
    model.accept_response(response(mixed, amplitude))
    window.refresh()
    assert window.equalization.active == profile
    expected = np.full((20, 20), np.nan, dtype=np.float32)
    expected[2] = corrected[2]
    np.testing.assert_array_equal(window.image.image, amplitude_rgb(expected, status.span_adc))
    assert model.uniform_observation(now) is None
    model.accept_info(info(hardware_id="replacement"))
    window.refresh()
    assert window.equalization.active is None
    assert window.image.image.shape == (20, 20, 3)
    assert np.all(window.image.image == INVALID_RGB)


def test_only_live_matching_metadata_allows_spatial_calibration():
    for replay in (True, False):
        model = ArrayView(SENSOR, replay=replay)
        state = snapshot(Assembler(sensor_id=SENSOR), 1, START)
        model.accept(state, 0)
        status = profile_status()
        status.header, status.direction = state.header, "both"
        model.accept_reference(status)
        assert model.equalization_context is None  # Missing trusted source metadata.
        sensor = info()
        model.accept_info(sensor)
        assert (model.equalization_context is None) == replay
        wrong = info()
        wrong.protocol_id = "previous protocol"
        model.accept_info(wrong)
        assert model.equalization_context is None
