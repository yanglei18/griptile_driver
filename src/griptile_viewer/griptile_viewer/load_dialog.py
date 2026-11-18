"""Guided, session-local load verification alongside the live contact map."""

from dataclasses import asdict
from datetime import datetime, timezone
import time

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from griptile_core.contact_image import contact_intensity_rgb
from griptile_core.grid import CELL_COUNT
from griptile_core.load_calibration import LoadTrial, relative_load_polarity
from .load_capture import HoldCapture
from .storage import save_json_atomic


_STEP_TEXT = {
    "check_up": "1/4　从无载缓慢夹到 {check:g} N，保持测力计读数。",
    "calibration": "2/4　继续加力至 {high:g} N，保持读数，记录标定点。",
    "check_down": "3/4　缓慢松到 {check:g} N，保持读数，记录卸载。",
    "release": "4/4　完全松开，确认测力计回到 0 N，记录残余响应。",
    "done": "验证已完成。可在其他位置或更换压头，用 {check:g} N 复测。",
}


def _force_input():
    field = QtWidgets.QDoubleSpinBox()
    field.setDecimals(3)
    field.setRange(0, 1_000_000)
    field.setSpecialValueText("输入读数")
    field.setSuffix(" N")
    field.setKeyboardTracking(False)
    return field


class LoadDialog(QtWidgets.QDialog):
    def __init__(self, viewer, wall_clock=time.monotonic_ns):
        super().__init__(viewer)
        self.viewer = viewer
        self.model = viewer.model
        self.wall_clock = wall_clock
        self.context = self.trial = self.capture = self.last_report = None
        self.evidence = {}
        self.setWindowTitle("GripTile · 已知载荷验证")
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setStyleSheet("""
            QDialog { background: #182129; }
            QDoubleSpinBox, QLineEdit, QTableWidget, QHeaderView::section {
                background: #25343f; color: #e5eef2; }
        """)
        self.resize(455, 350)
        layout = QtWidgets.QVBoxLayout(self)
        self.guide = QtWidgets.QLabel()
        self.guide.setWordWrap(True)
        layout.addWidget(self.guide)
        form = QtWidgets.QFormLayout()
        self.check_force, self.calibration_force = _force_input(), _force_input()
        form.addRow("验证载荷（加载、卸载各一次）", self.check_force)
        form.addRow("标定载荷（较高的一档）", self.calibration_force)
        layout.addLayout(form)
        self.live = QtWidgets.QLabel()
        self.live.setWordWrap(True)
        layout.addWidget(self.live)
        self.feedback = QtWidgets.QLabel()
        self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)
        self.location = QtWidgets.QLineEdit()
        self.location.setAccessibleName("复测位置或压头")
        self.location.setMaxLength(80)
        self.location.setPlaceholderText("复测位置或压头（选填）")
        layout.addWidget(self.location)
        self.results = QtWidgets.QTableWidget(0, 5)
        self.results.setHorizontalHeaderLabels(["测项", "已知 N", "估算 N", "误差 N", "峰值 ADC"])
        self.results.horizontalHeaderItem(3).setToolTip("估算总法向力减去测力计读数")
        self.results.horizontalHeaderItem(4).setToolTip("各次完整扫描的峰值响应的中位数及其代表色；总力由全阵列响应估算。")
        self.results.verticalHeader().hide()
        self.results.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.results.setWordWrap(False)
        self.results.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.results)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        actions = QtWidgets.QHBoxLayout()
        self.capture_button = QtWidgets.QPushButton("记录本步")
        self.zero_button = QtWidgets.QPushButton("采集无载参考")
        self.reset_button = QtWidgets.QPushButton("重新开始")
        self.export_button = QtWidgets.QPushButton("导出报告")
        for button in (self.zero_button, self.capture_button, self.export_button, self.reset_button):
            button.setAutoDefault(False)
            actions.addWidget(button)
        layout.addLayout(actions)
        self.zero_button.clicked.connect(viewer.start_reference)
        self.capture_button.clicked.connect(self.start_hold)
        self.reset_button.clicked.connect(self.reset_trial)
        self.export_button.clicked.connect(self.export_report)
        self.check_force.valueChanged.connect(self.refresh)
        self.calibration_force.valueChanged.connect(self.refresh)

    def reset_trial(self):
        self.trial = self.context = self.capture = None
        self.evidence.clear()
        self.location.clear()
        # Keep the last completed result exportable until its replacement is finished.
        self.feedback.clear()
        self.refresh()

    def start_hold(self):
        self.refresh()
        if not self.capture_button.isEnabled():
            return
        obs = self.model.load_observation(self.viewer.clock_ns())
        if obs is None:
            return
        if self.trial is None:
            self.trial = LoadTrial(self.check_force.value(), self.calibration_force.value())
            self.context = obs.context
            self.evidence.clear()
        self.capture = HoldCapture(obs, self.wall_clock(), require_stable=self.trial.stage == "calibration")
        self.feedback.clear()
        self.refresh()

    def _finish_hold(self, held):
        stage = self.trial.stage
        evidence = asdict(held)
        evidence["peak_rgb"] = list(contact_intensity_rgb(held.peak_adc, self.context.span_adc))
        if stage == "done":
            result = self.trial.verify(self.trial.check_n, held.total_adc)
            result.update(evidence, location=self.location.text().strip())
            self._set_polarity(result)
            self.last_report["extra_checks"].append(result)
            self.location.clear()
            self._show_results()
            self.feedback.setText("复测已追加。" if held.stable else "复测读数有波动，已保留本次测量。")
            return
        self.trial.record(held.total_adc)
        self.evidence[stage] = evidence
        if self.trial.stage == "done":
            report = self.trial.report()
            for name, summary in self.evidence.items():
                report["holds"][name].update(summary)
                self._set_polarity(report["holds"][name])
            report.update(source=asdict(self.context), captured_utc=datetime.now(timezone.utc).isoformat(),
                          extra_checks=[])
            self.last_report = report
            self._show_results()
        status = "读数有波动，已保留本次测量。" if not held.stable else ""
        self.feedback.setText(f"已记录 {held.samples} 组完整扫描。{status}")

    def _set_polarity(self, hold):
        anchor = self.evidence["calibration"]
        hold["polarity"] = relative_load_polarity(anchor["total_adc"], anchor["signed_total_adc"],
                                                  hold["total_adc"], hold["signed_total_adc"])

    def _show_results(self):
        report = self.last_report
        rows = [(label, report["holds"][key]) for label, key in
                (("加载验证", "check_up"), ("卸载验证", "check_down"), ("松开回零", "release"))]
        for index, check in enumerate(report["extra_checks"], start=1):
            rows.append((check["location"] or f"复测 {index}", check))
        self.results.setRowCount(len(rows))
        for index, (label, check) in enumerate(rows):
            flags = []
            if not check["stable"]:
                flags.append("波动")
            if check["polarity"] == "opposite":
                flags.append("反向")
            elif check["polarity"] == "mixed":
                flags.append("方向待定")
            if flags:
                label += "\n" + " · ".join(flags)
            values = (label, f"{check['known_n']:g}", f"{check['estimated_n']:.3f}", f"{check['error_n']:+.3f}")
            if not check["in_calibrated_range"]:
                values = (label, values[1], values[2] + "（外推）", values[3])
            for column, value in enumerate(values):
                self.results.setItem(index, column, QtWidgets.QTableWidgetItem(value))
            peak = QtWidgets.QTableWidgetItem(f"{check['peak_adc']:g}")
            peak.setData(QtCore.Qt.DecorationRole, QtGui.QColor(*check["peak_rgb"]))
            self.results.setItem(index, 4, peak)
            self.results.item(index, 0).setToolTip(f"相对标定档的响应方向；有符号响应 {check['signed_total_adc']:+g} ADC")
        self.results.resizeRowsToContents()
        table_height = self.results.horizontalHeader().height() + 2 * self.results.frameWidth()
        # Keep the complete cycle and first repeat visible together. Additional
        # locations scroll without moving the original verification out of view.
        table_height += sum(self.results.rowHeight(i) for i in range(min(len(rows), 4)))
        self.results.setFixedHeight(table_height + 3)
        release_signed = report["holds"]["release"]["signed_total_adc"]
        reversed_signal = "（与夹紧时反向）" if report["holds"]["release"]["polarity"] == "opposite" else ""
        self.summary.setText(f"色块为峰值触点的代表色。卸载－加载 {report['unloading_minus_loading_n']:+.3f} N。\n"
                             f"松开残余 {report['release_residual_n']:.3f} N，"
                             f"有符号响应 {release_signed:+g} ADC{reversed_signal}。")

    def refresh(self):
        model = self.model
        context = model.load_context
        if self.trial is not None and context != self.context:
            self.trial = self.context = self.capture = None
            self.evidence.clear()
            self.feedback.setText("连接或无载参考已变更，请重新开始载荷验证。")
        if not self.isVisible():
            return
        obs = None if model.replay else model.load_observation(self.viewer.clock_ns())
        if self.capture is not None:
            try:
                held = self.capture.update(obs, self.wall_clock())
                if held is not None:
                    self.capture = None
                    self._finish_hold(held)
                else:
                    self.feedback.setText(self.capture.progress_text)
            except ValueError as error:
                self.capture = None
                self.feedback.setText(str(error))

        check, high = self.check_force.value(), self.calibration_force.value()
        stage = self.trial.stage if self.trial else "check_up"
        valid_forces = 0 < check < high
        reference_count = sum(model.reference.valid_mask) if model.reference is not None else 0
        partial_reference = model.reference is not None and reference_count < CELL_COUNT
        if model.replay:
            guide = "请在连接测力计的实时采集窗口完成标定和验证。"
        elif context is None:
            guide = "打开夹爪并连接阵列，等待或采集无载参考。"
        elif partial_reference:
            guide = f"无载参考仅覆盖 {reference_count}/{CELL_COUNT} 个测点。等阵列完整更新后，请重新采集无载参考。"
        elif not valid_forces:
            guide = "选择测力计上可保持的两档载荷。先填较低的验证载荷，再填较高的标定载荷。"
        else:
            guide = _STEP_TEXT[stage].format(check=check, high=high)
        self.guide.setText(guide)
        if obs is None:
            fresh_count = int(model.mask(self.viewer.clock_ns()).sum())
            live = f"新鲜测点 {fresh_count}/{CELL_COUNT}，等候完整、与参考配对的触觉响应。"
        else:
            estimate = self.trial.estimate(obs.total_adc) if self.trial else None
            live = f"接触 {obs.active_cells} 点　峰值 {obs.peak_adc:g} ADC"
            if obs.peak_adc >= context.span_adc:
                live += "（色标已满）"
            live += f"\n响应总和 {obs.total_adc:g} ADC　有符号总和 {obs.signed_total_adc:+g} ADC"
            if estimate is not None:
                live += f"\n估算总法向力 {estimate:.3f} N"
                if estimate > self.trial.calibration_n:
                    live += "（超出标定范围，外推）"
        self.live.setText(live)
        self.check_force.setEnabled(self.trial is None)
        self.calibration_force.setEnabled(self.trial is None)
        self.capture_button.setText("采集中…" if self.capture else ("同载荷复测" if stage == "done" else "记录本步"))
        self.capture_button.setEnabled(self.capture is None and valid_forces and obs is not None)
        self.zero_button.setVisible(not model.replay and (context is None or partial_reference))
        self.zero_button.setEnabled(self.viewer.reference_button.isEnabled())
        self.reset_button.setVisible(self.trial is not None)
        self.location.setVisible(stage == "done")
        self.location.setEnabled(self.capture is None)
        self.export_button.setVisible(self.last_report is not None)
        self.results.setVisible(stage == "done")
        self.summary.setVisible(stage == "done")
        self.export_button.setText("导出报告" if stage == "done" else "导出上次报告")

    def hideEvent(self, event):
        if self.capture is not None:
            self.capture = None
            self.feedback.setText("采集已暂停，可重新记录本步。")
        super().hideEvent(event)

    def export_report(self):
        report = self.last_report
        if report is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出载荷验证报告", "griptile-load.json", "JSON (*.json)")
        if path:
            try:
                save_json_atomic(path, report)
            except (TypeError, ValueError, OSError) as error:
                self.feedback.setText(f"保存报告失败：{error}")
            else:
                self.feedback.setText("报告已保存。")
