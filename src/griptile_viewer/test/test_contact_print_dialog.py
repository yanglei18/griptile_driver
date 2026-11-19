from threading import Event
from time import monotonic, sleep

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from griptile_viewer import contact_print_dialog as ui
from griptile_viewer.print_capture import ContactPrint


class _Capture:
    def __init__(self):
        self.can_snapshot = False
        self.latest = None
        self.next_manual = None

    def snapshot(self):
        if self.can_snapshot:
            self.latest = self.next_manual
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


def _sample(*, automatic=False):
    frames = np.zeros((3, 20, 20), dtype=np.float32)
    frames[0, 10, 1] = 4000
    frames[1, 7, 5] = 1500
    frames[2, 7, 4] = 2600
    return ContactPrint((1_000_000_000, 1_301_000_000, 1_600_000_000),
                        tuple(frames), automatic, False)


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
    assert not window.export_button.isEnabled()
    with pytest.raises(ValueError):
        window.save_png(tmp_path / "empty.png")

    capture.next_manual = record = _sample()
    capture.can_snapshot = True
    window.refresh()
    window.shutter_button.click()
    _settle(window, record, qt_app)
    assert window.plate.proof_labels == ("600 毫秒前", "299 毫秒前", "此刻")
    assert "299 毫秒前" in window.plate.accessibleDescription()
    assert window.plate.bounds == QtCore.QRect(0, 5, 8, 8)
    assert window.export_button.isEnabled() and window.zoom.isEnabled()
    assert len(window.plate.images) == 4

    cropped = window.plate.export_image()
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


def test_stale_render_cannot_replace_new_stream_and_unchanged_print_is_cached(
        display, qt_app, monkeypatch):
    window, capture = display
    first = _sample()
    second = _sample(automatic=True)
    jobs = []
    release = Event()
    render = ui.render_contact_print

    def slow(frames):
        jobs.append(id(frames))
        assert release.wait(4)
        return render(frames)

    monkeypatch.setattr(ui, "render_contact_print", slow)
    try:
        capture.latest = first
        window.refresh()
        capture.latest = None
        window.refresh()
        capture.latest = second
        window.refresh()
        assert window.plate.record is None
        assert not window.export_button.isEnabled()
    finally:
        release.set()
    _settle(window, second, qt_app)
    for _ in range(5):
        window.refresh()
    assert jobs == [id(first.frames), id(second.frames)]
    assert "自动留印" in window.plate.caption()
    capture.latest = None
    window.refresh()
    assert not window.export_button.isEnabled()
    assert not window.plate.images


def test_export_preserves_print_displayed_when_file_picker_opened(display, qt_app, tmp_path, monkeypatch):
    window, capture = display
    capture.latest = _sample()
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
