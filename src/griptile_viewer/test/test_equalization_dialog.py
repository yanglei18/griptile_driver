from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from griptile_core.display_response import ReferenceProfile
from griptile_core.grid import LAYOUT_ID, SHAPE
from griptile_viewer.load_capture import LoadContext


class Controller:
    def __init__(self, persistent):
        self.persistent = persistent
        self.profile = None
        self.enabled = False
        self.error = ""
        self.fail_save = False

    def sync(self):
        pass

    def adopt(self, profile, context):
        if self.fail_save:
            raise OSError("写入失败")
        self.profile = profile
        self.enabled = True

    def set_enabled(self, enabled):
        self.enabled = enabled

    def clear(self):
        self.profile = None
        self.enabled = False


@pytest.fixture
def setup_dialog(qt_app):
    from pyqtgraph.Qt import QtCore, QtWidgets
    from griptile_viewer.equalization_dialog import EqualizationDialog

    owner = QtWidgets.QWidget()
    source = SimpleNamespace(now=1_000_000_000, fresh=True, delta=np.zeros(SHAPE), stamp=1_000_000_000)
    baseline = np.full(SHAPE, 16000, dtype=np.int32)
    profile = ReferenceProfile(baseline.ravel(), np.full(400, 4.), np.ones(400, dtype=bool))
    context = LoadContext("jaw", "session", 1, "usb:test", LAYOUT_ID, "both", 3000., "jaw", 1000, "digest", 1)
    model = SimpleNamespace(reference=profile, replay=False, equalization_context=context, binding=lambda: {"id": "test"})

    def observe(now_ns):
        if not source.fresh or model.equalization_context is None:
            return None
        return SimpleNamespace(context=model.equalization_context, source_now_ns=now_ns, row_stamps=(source.stamp,) * 20,
                               raw_adc=np.asarray(baseline + source.delta, dtype=np.uint16))

    owner.clock_ns = lambda: source.now
    model.uniform_observation = observe
    owner.model = model
    owner.equalization = Controller(True)
    dialog = EqualizationDialog(owner, wall_clock=owner.clock_ns)
    dialog.show()
    qt_app.processEvents()
    dialog.refresh()

    def advance(delta, frames=14, *, update_stamp=True, interval_ns=100_000_000):
        source.delta = np.rint(delta).astype(np.int32)
        for _ in range(frames):
            source.now += interval_ns
            if update_stamp:
                source.stamp = source.now
            dialog.refresh()

    yield dialog, owner, model, source, advance
    dialog.close()
    owner.close()
    owner.deleteLater()
    qt_app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    qt_app.processEvents()


SENSITIVITY = np.linspace(.72, 1.3, 400).reshape(SHAPE)


def test_two_levels_independent_check_and_persist_only_on_adoption(setup_dialog, monkeypatch):
    d, owner, _, _, step = setup_dialog
    control = owner.equalization
    assert "1/2" in d.guide.text() and d.capture_button.isEnabled()
    d.capture_button.click()
    step(600 * SENSITIVITY)
    assert d.fit is not None and d.candidate is None and "2/2" in d.guide.text()
    assert not d.adopt_button.isVisible() and control.profile is None

    d.capture_button.click()
    step(620 * SENSITIVITY)  # Too close to the fitted load; retry the validation step only.
    assert d.fit is not None and d.candidate is None and d.feedback.text()
    assert control.profile is None
    d.capture_button.click()
    step(1000 * SENSITIVITY)
    assert d.candidate is not None and d.adopt_button.isEnabled()
    assert "400/400" in d.observation.text()
    assert d.candidate.after_cv < d.candidate.before_cv and control.profile is None
    validated = d.candidate
    control.fail_save = True
    d.adopt_button.click()
    assert d.candidate is validated and control.profile is None and "写入失败" in d.feedback.text()
    control.fail_save = False
    d.adopt_button.click()
    assert control.profile is validated and d.fit is None and d.candidate is None
    assert control.enabled and d.enable_button.isVisible()
    d.enable_button.click()
    assert not control.enabled and control.profile is validated
    d.enable_button.click()
    assert control.enabled

    from pyqtgraph.Qt import QtWidgets
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args: QtWidgets.QMessageBox.Cancel)
    d.clear_button.click()
    assert control.profile is validated
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args: QtWidgets.QMessageBox.Yes)
    d.clear_button.click()
    assert control.profile is None and not d.clear_button.isVisible()


def test_reference_switch_and_partial_hold_cannot_overwrite_previous(setup_dialog):
    d, owner, model, _, step = setup_dialog
    old = SimpleNamespace(after_cv=.03)
    owner.equalization.profile = old
    owner.equalization.enabled = True
    d.capture_button.click()
    step(np.pad(np.ones((2, 2)) * 750, ((5, 13), (5, 13))))
    assert d.fit is None and d.feedback.text() and owner.equalization.profile is old
    d.capture_button.click()
    step(600 * SENSITIVITY)
    assert d.fit is not None
    d.capture_button.click()
    step(1000 * SENSITIVITY, 4)
    model.equalization_context = replace(model.equalization_context, generation=2)
    d.refresh()
    assert d.fit is None and d.capture is None and d.candidate is None
    assert "变更" in d.feedback.text() and owner.equalization.profile is old


def test_confirmed_context_session_only_and_capture_cancel_on_close(setup_dialog):
    d, owner, model, _, step = setup_dialog
    model.binding = lambda: None
    owner.equalization.persistent = False
    d.capture_button.click()
    step(600 * SENSITIVITY, 4)
    d.close()
    d.show()
    d.refresh()
    assert d.capture is None and d.fit is None and d.capture_button.isEnabled()
    d.capture_button.click()
    step(600 * SENSITIVITY)
    d.capture_button.click()
    step(1000 * SENSITIVITY)
    assert "本次" in d.adopt_button.text()
    d.adopt_button.click()
    assert "仅本次连接" in d.saved_status.text()


def test_duplicate_rows_missing_data_and_time_gaps_restart_capture(setup_dialog):
    d, _, _, src, step = setup_dialog
    d.capture_button.click()
    step(600 * SENSITIVITY, 8)
    assert len(d.capture.samples) == 8
    step(600 * SENSITIVITY, 5, update_stamp=False)
    step(600 * SENSITIVITY, 1)
    assert len(d.capture.samples) == 1 and d.fit is None
    step(600 * SENSITIVITY, 4)
    src.fresh = False
    d.refresh()
    assert not d.capture.samples
    src.fresh = True
    step(600 * SENSITIVITY, 1)
    assert not d.capture.samples  # Fence the first mixed rolling scan after a gap.
    step(600 * SENSITIVITY, 14)
    assert d.fit is not None
    d.capture_button.click()
    step(1000 * SENSITIVITY, 1)
    step(1000 * SENSITIVITY, 1, interval_ns=10_000_000_000)
    assert d.capture is None and d.candidate is None and "超时" in d.feedback.text()
    assert d.fit is not None
