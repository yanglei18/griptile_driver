from dataclasses import replace
from threading import Event
from time import monotonic, sleep

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from griptile_core.contact_image import CONTACT_COLOR_STOPS, contact_intensity_rgb, SURFACE_SCALE, UNLOADED_RGB
from griptile_viewer import contact_print_dialog as ui
from griptile_viewer.print_capture import ContactPrint


class _Capture:
    def __init__(self):
        self.can_snapshot = False
        self.latest = None
        self.live = None
        self.contact_present = None
        self.contact_id = None

    def press(self, record):
        self.live = record
        self.contact_present = True
        self.can_snapshot = len(record.frames) == 3
        self.contact_id = object()

    def release(self):
        self.live = self.contact_id = None
        self.contact_present = self.can_snapshot = False

    def live_preview(self):
        return self.live

    def snapshot(self):
        if self.can_snapshot:
            self.latest = self.live
            return self.latest


@pytest.fixture
def display(qt_app):
    capture = _Capture()
    window = ui.ContactPrintDialog(None, capture)
    window.show()
    qt_app.processEvents()
    yield window, capture
    window.close()
    window.deleteLater()
    qt_app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    qt_app.processEvents()


def _sample(*, automatic=False, span_adc=3000):
    frames = np.zeros((3, 20, 20), dtype=np.float32)
    frames[0, 10, 1] = 4000
    frames[1, 7, 5] = 1500
    frames[2, 7, 4] = 2600
    return ContactPrint((1_000_000_000, 1_301_000_000, 1_600_000_000),
                        tuple(frames), automatic, False, span_adc=span_adc, footprints=tuple(frames > 0))


def _settle(window, record, qt_app):
    end = monotonic() + 5
    while monotonic() < end:
        window.refresh()
        qt_app.processEvents()
        if window.plate.record is record and window._work is None:
            return
        sleep(.005)
    pytest.fail(f"touch print did not settle: {window.status.text()}")


def test_manual_capture_actual_timing_full_array_and_export(display, qt_app, tmp_path):
    window, capture = display
    assert not window.isModal()
    assert not window.shutter_button.isEnabled()
    assert not window.history_button.isEnabled() and window.plate.empty_text == "等待有效读数"
    assert not window.export_button.isEnabled()
    with pytest.raises(ValueError):
        window.save_png(tmp_path / "empty.png")

    record = _sample(span_adc=2000)
    capture.press(record)
    window.refresh()
    window.shutter_button.click()
    _settle(window, record, qt_app)
    assert window.plate.proof_labels == ("600 毫秒前", "299 毫秒前", "此刻")
    assert window.plate.caption() == "实时接触" and not window.export_button.isEnabled()
    assert not window.history_button.isChecked()
    window.history_button.click()
    _settle(window, record, qt_app)
    assert window.plate.proof_labels[-1] == "取印时" and window.plate.historical
    assert "299 毫秒前" in window.plate.accessibleDescription()
    assert "1 个接触测点" in window.plate.accessibleDescription()
    assert "三帧响应" in window.plate.accessibleDescription()
    assert "2,600 ADC" in window.plate.peak_text and "2000 ADC" in window.plate.accessibleDescription()
    assert window.plate.bounds == ui.FULL_ARRAY and not window.zoom.isChecked()
    assert window.export_button.isEnabled() and window.zoom.isEnabled()
    assert len(window.plate.images) == 4
    center = SURFACE_SCALE // 2
    master, first, middle, last = window.plate.images
    first_cell = QtCore.QPoint(SURFACE_SCALE + center, 10 * SURFACE_SCALE + center)
    assert window.style_combo.currentData() == "imprint" and window.plate.face_style == "imprint"
    assert master.pixelColor(first_cell) == ui.GROUND
    assert "暖铜增强" in window.plate.accessibleDescription()
    for pic, row, col, adc in ((first, 10, 1, 4000), (middle, 7, 5, 1500), (last, 7, 4, 2600)):
        rgb = pic.pixelColor(col * SURFACE_SCALE + center, row * SURFACE_SCALE + center).getRgb()[:3]
        assert rgb == contact_intensity_rgb(adc, record.span_adc)

    full_array = window.plate.export_image()
    # The exported color key uses the same fixed palette as the measured pixels.
    margin = 640 // 24
    for level, rgb in CONTACT_COLOR_STOPS:
        x = int(2 * (margin + level * (640 - 2 * margin)))
        x = min(x, 2 * (640 - margin) - 1)
        np.testing.assert_allclose(full_array.pixelColor(x, 1140).getRgb()[:3], rgb, atol=3)
    window.resize(335, 475)
    qt_app.processEvents()
    path = tmp_path / "actual.png"
    window.save_png(path)
    loaded = QtGui.QImage(str(path))
    assert loaded.size() == QtCore.QSize(1280, 1200)
    assert loaded.pixelColor(2, 2) == ui.GROUND
    assert loaded.pixelColor(2, 1198) == ui.PAPER
    assert loaded.convertToFormat(full_array.format()) == full_array
    assert window.plate.record is record
    capture.release()
    window.history_button.click()
    assert window.plate.record is None and not window.plate.images and window.plate.empty_text == "无接触"


def test_manual_zoom_uses_measured_support_while_proofs_keep_weak_responses(display, qt_app):
    window, capture = display
    base = _sample()
    frames = np.array(base.frames)
    # The support on a dim cell can be reliable while stronger residuals elsewhere are not.
    frames[0, 15, 4] = 90.
    frames[2, 0, 19] = 700.
    masks = np.array(base.footprints)
    masks[0, 15, 4] = True
    shaped = replace(base, frames=tuple(frames), footprints=tuple(masks))
    capture.press(shaped)
    _settle(window, shaped, qt_app)
    assert window.plate.bounds == ui.FULL_ARRAY and not window.zoom.isChecked()
    window.zoom.click()
    locked = QtCore.QRect(0, 6, 7, 11)  # Includes the earlier weak contact.
    assert window.plate.bounds == locked
    assert window.plate.footprint_label() == "1 个接触测点"
    x, y = 19 * SURFACE_SCALE + SURFACE_SCALE // 2, SURFACE_SCALE // 2
    master, _, _, latest = window.plate.images
    assert latest.pixelColor(x, y).getRgb()[:3] == contact_intensity_rgb(700, shaped.span_adc)
    assert master.pixelColor(x, y) == ui.GROUND

    # A later footprint and a saved print each have a different close-up.
    # Neither should move the region selected by the user.
    shifted = replace(shaped, frames=tuple(np.roll(frames, 8, axis=2)),
                      footprints=tuple(np.roll(masks, 8, axis=2)))
    capture.live = shifted
    _settle(window, shifted, qt_app)
    assert window.plate.bounds == locked
    capture.latest = base
    capture.release()
    window.refresh()
    assert window.zoom.isEnabled() and window.plate.bounds == locked
    window.history_button.click()
    _settle(window, base, qt_app)
    assert window.plate.bounds == locked
    enlarged = window.plate.export_image()
    window.zoom.click()
    assert window.plate.bounds == ui.FULL_ARRAY
    assert window.plate.export_image() != enlarged
    window.history_button.click()

    unknown = replace(shaped, footprints=None)
    capture.press(unknown)
    _settle(window, unknown, qt_app)
    assert not window.zoom.isEnabled() and window.plate.bounds == ui.FULL_ARRAY
    assert "接触范围待测" in window.plate.accessibleDescription()
    assert window.plate.face_style == "response"
    assert window.plate.images[0].pixelColor(x, y) == latest.pixelColor(x, y)


def test_edge_taxel_stays_registered_when_proofs_arrive_and_style_changes(display, qt_app, monkeypatch):
    window, capture = display
    # Mark exactly one native taxel in the face so its painted position and size
    # can be measured independently of the palette, adjacent ink and Qt text.
    marker = QtGui.QColor("#ff00ff")

    def render(record, style):
        blank = np.zeros((20 * SURFACE_SCALE, 20 * SURFACE_SCALE, 3), np.uint8)
        face = blank.copy()
        face[-SURFACE_SCALE:, :SURFACE_SCALE] = marker.getRgb()[:3]
        return (face, blank, blank, blank) if len(record.frames) == 3 else (face,)

    def painted_taxel():
        boxes = []
        for image in (window.plate.grab().toImage(), window.plate.export_image()):
            mask = image.createMaskFromColor(marker.rgb(), QtCore.Qt.MaskOutColor)
            box = QtGui.QRegion(QtGui.QBitmap.fromImage(mask)).boundingRect()
            assert 0 < box.width() < image.width() / 20
            assert box.left() < image.width() / 2
            boxes.append(box)
        return boxes

    monkeypatch.setattr(ui, "_render_exposures", render)
    base = _sample()
    fields = np.array(base.frames)
    fields[:, 19, 0] = 2000
    triplet = replace(base, frames=tuple(fields), footprints=tuple(fields > 0))
    first = replace(triplet, frames=triplet.frames[:1], stamps_ns=triplet.stamps_ns[:1],
                    footprints=triplet.footprints[:1])
    capture.press(first)
    _settle(window, first, qt_app)
    expected = painted_taxel()
    capture.live = triplet
    _settle(window, triplet, qt_app)
    assert painted_taxel() == expected
    window.style_combo.setCurrentIndex(window.style_combo.findData("response"))
    _settle(window, triplet, qt_app)
    assert painted_taxel() == expected


def test_first_frame_shows_its_face_before_three_exposures_exist(display, qt_app):
    window, capture = display
    triplet = _sample()
    first = replace(triplet, stamps_ns=triplet.stamps_ns[:1], frames=triplet.frames[:1],
                    footprints=triplet.footprints[:1])
    capture.press(first)
    _settle(window, first, qt_app)
    assert len(window.plate.images) == 1 and window.plate.proof_labels == ("此刻",)
    assert "等待三帧对照" in window.plate.accessibleDescription()
    assert "灰白为单帧印纹" in window.plate.accessibleDescription() and "暖铜增强" not in window.plate.accessibleDescription()
    assert window.plate.footprint_label() == "1 个接触测点"
    assert not window.shutter_button.isEnabled() and not window.export_button.isEnabled()
    capture.live = triplet
    capture.can_snapshot = True
    _settle(window, triplet, qt_app)
    assert len(window.plate.images) == 4 and window.shutter_button.isEnabled()
    capture.release()
    window.refresh()
    assert not window.plate.images and window.plate.empty_text == "无接触"


def test_stale_render_cannot_replace_new_stream_and_unchanged_print_is_cached(
        display, qt_app, monkeypatch):
    window, capture = display
    first = _sample()
    second = _sample(automatic=True)
    jobs = []
    release = Event()
    render = ui.render_contact_print

    def slow(frames, span, footprint, *, style):
        jobs.append(id(frames))
        assert release.wait(4)
        return render(frames, span, footprint, style=style)

    monkeypatch.setattr(ui, "render_contact_print", slow)
    try:
        capture.press(first)
        window.refresh()
        capture.release()
        window.refresh()
        capture.latest = second
        window.refresh()
        assert window.plate.record is None and not window.plate.images and window.plate.empty_text == "无接触"
        assert not window.export_button.isEnabled()
        assert not window.shutter_button.isEnabled() and not window.history_button.isChecked()
    finally:
        release.set()
    _settle(window, None, qt_app)
    assert window.plate.empty_text == "无接触"
    window.history_button.click()
    _settle(window, second, qt_app)
    for _ in range(5):
        window.refresh()
    assert jobs == [id(first.frames), id(second.frames)]
    assert window.plate.caption() == "历史留印 · 自动留印"
    capture.latest = None
    window.refresh()
    assert not window.export_button.isEnabled()
    assert not window.plate.images


def test_export_preserves_print_displayed_when_file_picker_opened(display, qt_app, tmp_path, monkeypatch):
    window, capture = display
    capture.latest = _sample()
    window.refresh()
    window.history_button.click()
    _settle(window, capture.latest, qt_app)
    before = window.plate.export_image()
    path = tmp_path / "pinned.png"

    def choose(*_args):
        capture.latest = _sample(automatic=True)
        _settle(window, capture.latest, qt_app)
        return str(path), "PNG"

    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", choose)
    window.export_button.click()
    after = QtGui.QImage(str(path)).convertToFormat(before.format())
    assert after == before
    assert after != window.plate.export_image()


def test_live_preview_advances_and_reopen_defaults_to_current_contact(display, qt_app):
    window, capture = display
    first = _sample()
    capture.press(first)
    _settle(window, first, qt_app)
    second = _sample()
    capture.live = second
    _settle(window, second, qt_app)
    assert window.plate.caption() == "实时接触"
    capture.latest = first
    capture.release()
    window.refresh()
    assert not window.plate.images and not window.plate.proof_labels
    window.history_button.click()
    _settle(window, first, qt_app)
    window.close()
    window.show()
    assert not window.history_button.isChecked()
    assert window.plate.record is None and window.plate.empty_text == "无接触"
    assert window.history_button.isEnabled() and not window.export_button.isEnabled()


def test_replacing_live_segment_discards_in_flight_preview_without_an_intervening_draw(
        display, qt_app, monkeypatch):
    window, capture = display
    first, second = _sample(), _sample()
    blocker = Event()
    render = ui.render_contact_print

    def render_after_release(frames, span, footprint, *, style):
        if frames is first.frames:
            assert blocker.wait(4)
        return render(frames, span, footprint, style=style)

    monkeypatch.setattr(ui, "render_contact_print", render_after_release)
    try:
        capture.press(first)
        window.refresh()
        capture.release()
        capture.press(second)
        window.refresh()
        assert window.plate.record is None
    finally:
        blocker.set()
    _settle(window, second, qt_app)
    assert window.plate.record is second and not window.plate.historical


def test_style_switch_rejects_old_render_and_exports_selected_style(display, qt_app, monkeypatch, tmp_path):
    window, capture = display
    record = _sample(span_adc=1500)
    capture.latest = record
    window.refresh()
    blocker = Event()
    render = ui.render_contact_print
    styles = []

    def slow_ink(frames, span, footprint, *, style):
        styles.append(style)
        if style == "imprint":
            assert blocker.wait(4)
        return render(frames, span, footprint, style=style)

    monkeypatch.setattr(ui, "render_contact_print", slow_ink)
    try:
        window.history_button.click()
        window.style_combo.setCurrentIndex(window.style_combo.findData("response"))
        assert not window.export_button.isEnabled()
    finally:
        blocker.set()
    _settle(window, record, qt_app)
    assert styles == ["imprint", "response"] and window.plate.face_style == "response"
    assert "主图按响应强度着色" in window.plate.accessibleDescription()
    response_face, *proofs = window.plate.images
    assert response_face.pixelColor(0, 0).getRgb()[:3] == UNLOADED_RGB
    path = tmp_path / "response.png"
    window.save_png(path)
    response_export = QtGui.QImage(str(path))
    window.style_combo.setCurrentIndex(window.style_combo.findData("imprint"))
    _settle(window, record, qt_app)
    assert window.plate.face_style == "imprint" and window.plate.images[0].pixelColor(0, 0) == ui.GROUND
    assert list(window.plate.images[1:]) == proofs
    window.save_png(path)
    assert QtGui.QImage(str(path)) != response_export
