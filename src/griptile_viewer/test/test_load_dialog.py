"""Exercise the operator's real Qt controls and hold state machine with a controlled clock."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from griptile_core.grid import LAYOUT_ID
from griptile_viewer.load_capture import LoadContext, LoadObservation
from griptile_viewer.model import ArrayView


@pytest.fixture
def bench(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from griptile_viewer.window import QtWidgets, ViewerWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    clock = SimpleNamespace(now=1_000_000_000)
    context = LoadContext("jaw", "session", 1, "board", LAYOUT_ID, "both", 3000., "jaw", 42, "abc", 1)
    model = ArrayView("jaw")
    source = SimpleNamespace(context=context, total=0., signed=0., peak=0., active=0, fresh=True)
    monkeypatch.setattr(ArrayView, "load_context", property(lambda self: source.context))

    def observe(now):
        if source.context is None or not source.fresh:
            return None
        return LoadObservation(source.context, (now,) * 20, now, source.total, source.signed,
                               source.peak, source.active)

    monkeypatch.setattr(model, "load_observation", observe)
    requests = []
    window = ViewerWindow(model, clock_ns=lambda: clock.now, request_reference=requests.append)
    window.show()
    app.processEvents()
    window.load_button.click()
    dialog = window.load_dialog
    dialog.wall_clock = lambda: clock.now
    dialog.check_force.setValue(2.)
    dialog.calibration_force.setValue(5.)

    def hold(total, signed, peak, active):
        source.total, source.signed, source.peak, source.active = total, signed, peak, active
        dialog.capture_button.click()
        assert dialog.capture is not None
        for _ in range(3):
            clock.now += 200_000_000
            window.refresh()
        assert dialog.capture is None

    yield window, dialog, source, clock, hold
    window.close()
    app.processEvents()


def test_verify_loading_release_recheck_and_export(bench, monkeypatch, tmp_path):
    from griptile_viewer.window import QtWidgets, SURFACE_MODE

    window, dialog, source, clock, hold = bench
    assert not dialog.isModal() and window.mode.currentIndex() == SURFACE_MODE
    assert "1/4" in dialog.guide.text()
    hold(3400., 3400., 1700., 2)
    assert dialog.trial.stage == "calibration"
    assert not dialog.check_force.isEnabled()
    # The zero-response anchor can be retried on its own. Previous hold survives.
    hold(0., 0., 0., 0)
    assert dialog.trial.stage == "calibration"
    assert "没有触觉响应" in dialog.feedback.text()
    hold(10000., 10000., 5000., 2)
    source.total = 11000.
    window.refresh()
    assert "5.500 N" in dialog.live.text() and "超出标定范围" in dialog.live.text()
    assert "色标已满" in dialog.live.text()
    hold(4600., -4600., 2300., 2)
    dialog.capture_button.click()
    for total in (300., 200., 50.):
        clock.now += 200_000_000
        source.total, source.signed, source.peak = total, -total, total / 2
        window.refresh()
    assert dialog.trial.stage == "done" and dialog.results.rowCount() == 3
    report = dialog.last_report
    from griptile_core.contact_image import contact_intensity_rgb
    from griptile_viewer.window import QtCore
    expected = contact_intensity_rgb(1700, 3000)
    assert tuple(report["holds"]["check_up"]["peak_rgb"]) == expected
    assert dialog.results.item(0, 4).data(QtCore.Qt.DecorationRole).getRgb()[:3] == expected
    assert "与夹紧时反向" in dialog.summary.text()
    assert "波动" in dialog.results.item(2, 0).text()
    assert report["holds"]["release"]["stable"] is False
    assert report["holds"]["release"]["signed_spread_adc"] == 250.
    assert report["holds"]["check_up"]["estimated_n"] == pytest.approx(1.7)
    assert report["holds"]["check_up"]["polarity"] == "same"
    assert report["holds"]["check_down"]["polarity"] == "opposite"
    assert "反向" in dialog.results.item(1, 0).text()
    assert report["holds"]["check_down"]["estimated_n"] == pytest.approx(2.3)
    assert report["holds"]["release"]["signed_total_adc"] == -200.
    assert report["release_residual_n"] == pytest.approx(.1)
    assert report["unloading_minus_loading_n"] == pytest.approx(.6)
    assert report["source"]["hardware_id"] == "board" and report["source"]["direction"] == "both"
    assert "同载荷复测" in dialog.capture_button.text()
    dialog.location.setText("边缘压头")
    hold(6000., -5., 1000., 6)
    assert dialog.results.rowCount() == 4
    QtWidgets.QApplication.processEvents()
    assert dialog.results.viewport().height() >= sum(dialog.results.rowHeight(i) for i in range(4))
    assert dialog.results.rowAt(0) == 0
    assert report["extra_checks"][0]["location"] == "边缘压头"
    assert report["extra_checks"][0]["polarity"] == "mixed"
    assert "方向待定" in dialog.results.item(3, 0).text()
    assert report["extra_checks"][0]["estimated_n"] == pytest.approx(3.)
    path = tmp_path / "report.json"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(path), "JSON"))
    dialog.export_button.click()
    assert json.loads(path.read_text()) == report
    # A different reference stops live estimates; the previous report remains exportable.
    source.context = replace(source.context, generation=2)
    window.refresh()
    assert dialog.trial is None and "估算" not in dialog.live.text()
    assert dialog.last_report is report and not dialog.results.isVisible()
    assert dialog.export_button.isEnabled() and "上次" in dialog.export_button.text()


def test_wait_for_reference_partial_frames_cancel_and_resume(bench):
    window, dialog, source, clock, hold = bench
    context = source.context
    source.context = None
    window.refresh()
    assert not dialog.capture_button.isEnabled() and dialog.zero_button.isVisible()
    source.context = context
    source.fresh = False
    window.refresh()
    assert not dialog.capture_button.isEnabled() and "400" in dialog.live.text()
    source.fresh = True
    window.refresh()
    dialog.capture_button.click()
    clock.now += 5_000_000
    window.refresh()
    assert dialog.capture is not None
    dialog.close()
    window.load_button.click()
    assert window.load_dialog is dialog and dialog.capture is None
    assert dialog.trial.stage == "check_up"
    hold(80., 80., 20., 4)
    dialog.capture_button.click()
    source.context = None
    window.refresh()
    assert dialog.capture is None and dialog.trial is None
    assert "变更" in dialog.feedback.text()
    source.context = context
    window.model.replay = True
    window.refresh()
    assert not dialog.capture_button.isEnabled() and not dialog.zero_button.isVisible()


def test_incomplete_reference_can_be_recaptured_from_the_dialog(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from griptile_core.assembler import Assembler
    from griptile_core.protocol import Block
    from griptile_core.serial_source import Batch
    from griptile_viewer.window import QtWidgets, ViewerWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    model = ArrayView("jaw")
    assembler = Assembler(sensor_id="jaw")
    t = 2_000_000_000
    assembler.apply(Batch(sensor_id="jaw", session_id="bench", stream_epoch=1, sequence=1,
                          first_packet_index=0, host_realtime_ns=t, host_monotonic_ns=t,
                          connected=True, time_valid=True,
                          blocks=tuple(Block(row, (100,) * 20) for row in range(20))))
    header = SimpleNamespace(frame_id="jaw", stamp=SimpleNamespace(sec=2, nanosec=0))
    model.accept(SimpleNamespace(header=header, **asdict(assembler.snapshot(t, 100_000_000))), 0)
    reference = SimpleNamespace(header=header, sensor_id="jaw", session_id="bench", stream_epoch=1,
                                hardware_id="", phase="ready", progress=1., direction="both", span_adc=3000.,
                                baseline_adc=[100.] * 400, activation_threshold_adc=[3.] * 400,
                                valid_mask=[True] * 399 + [False])
    model.accept_reference(reference)
    requests = []
    window = ViewerWindow(model, clock_ns=lambda: 0, request_reference=requests.append)
    window.show()
    app.processEvents()
    window.load_button.click()
    dialog = window.load_dialog
    assert model.load_context is not None
    assert "399/400" in dialog.guide.text() and "400/400" in dialog.live.text()
    assert dialog.zero_button.isVisible() and dialog.zero_button.isEnabled()
    dialog.zero_button.click()
    assert requests == [None]
    window.close()
    app.processEvents()
