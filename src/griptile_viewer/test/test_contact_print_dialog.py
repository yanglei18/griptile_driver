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
        self.contact_present = self.can_snapshot = True
        self.contact_id = object()

    def release(self):
        self.live = self.contact_id = None
        self.contact_present = self.can_snapshot = False

    def preview(self):
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
                        tuple(frames), automatic, False, span_adc=span_adc)


def _settle(window, record, qt_app):
    end = monotonic() + 5
    while monotonic() < end:
        window.refresh()
        qt_app.processEvents()
        if window.plate.record is record and window._work is None:
            return
        sleep(.005)
    pytest.fail(f"touch print did not settle: {window.status.text()}")


def test_manual_capture_actual_timing_shared_crop_resize_and_export(display, qt_app, tmp_path):
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
    assert "2,600 ADC" in window.plate.peak_text and "2000 ADC" in window.plate.accessibleDescription()
    assert window.plate.bounds == QtCore.QRect(0, 5, 8, 8)
    assert window.export_button.isEnabled() and window.zoom.isEnabled()
    assert len(window.plate.images) == 4
    center = SURFACE_SCALE // 2
    master, first, middle, last = window.plate.images
    first_cell = QtCore.QPoint(SURFACE_SCALE + center, 10 * SURFACE_SCALE + center)
    assert master.pixelColor(first_cell).getRgb()[:3] == UNLOADED_RGB
    for pic, row, col, adc in ((first, 10, 1, 4000), (middle, 7, 5, 1500), (last, 7, 4, 2600)):
        rgb = pic.pixelColor(col * SURFACE_SCALE + center, row * SURFACE_SCALE + center).getRgb()[:3]
        assert rgb == contact_intensity_rgb(adc, record.span_adc)

    cropped = window.plate.export_image()
    # The exported color key uses the same fixed palette as the measured pixels.
    margin = 640 // 24
    for level, rgb in CONTACT_COLOR_STOPS:
        x = int(2 * (margin + level * (640 - 2 * margin)))
        x = min(x, 2 * (640 - margin) - 1)
        np.testing.assert_allclose(cropped.pixelColor(x, 1140).getRgb()[:3], rgb, atol=3)
    window.zoom.setChecked(False)
    window.resize(335, 475)
    qt_app.processEvents()
    path = tmp_path / "actual.png"
    window.save_png(path)
    loaded = QtGui.QImage(str(path))
    assert loaded.size() == QtCore.QSize(1280, 1200)
    assert loaded.pixelColor(2, 2) == ui.GROUND
    assert loaded.pixelColor(2, 1198) == ui.PAPER
    assert loaded.convertToFormat(cropped.format()) != cropped
    assert window.plate.record is record
    capture.release()
    window.history_button.click()
    assert window.plate.record is None and not window.plate.images and window.plate.empty_text == "无接触"


def test_stale_render_cannot_replace_new_stream_and_unchanged_print_is_cached(
        display, qt_app, monkeypatch):
    window, capture = display
    first = _sample()
    second = _sample(automatic=True)
    jobs = []
    release = Event()
    render = ui.render_contact_print

    def slow(frames, span):
        jobs.append(id(frames))
        assert release.wait(4)
        return render(frames, span)

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

    def render_after_release(frames, span):
        if frames is first.frames:
            assert blocker.wait(4)
        return render(frames, span)

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
