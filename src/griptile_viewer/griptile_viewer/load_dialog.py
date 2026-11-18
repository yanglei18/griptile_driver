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
    "check_up": "1/4　夹紧至 {check:g} N",
    "calibration": "2/4　继续夹紧至 {high:g} N",
    "check_down": "3/4　慢慢松回 {check:g} N",
    "release": "4/4　完全松开至 0 N",
    "done": "同一测力计读数　{check:g} N",
}


def _flags(hold):
    flags = []
    if not hold["stable"]:
        flags.append("读数波动")
    if hold["polarity"] == "opposite":
        flags.append("信号反向")
    elif hold["polarity"] == "mixed":
        flags.append("方向待定")
    if not hold["in_calibrated_range"]:
        flags.append("超范围")
    return " · ".join(flags)


def _force_text(value, signed=False):
    if 0 < abs(value) < 0.001:
        return ("−" if value < 0 else ("+" if signed else "")) + "<0.001"
    return format(value, "+.3f" if signed else ".3f").rstrip("0").rstrip(".")


class _PeakReading(QtWidgets.QWidget):
    def __init__(self, label):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(QtWidgets.QLabel(label))
        self.swatch = QtWidgets.QLabel()
        self.swatch.setAccessibleName(label + "峰值色")
        layout.addWidget(self.swatch)
        self.value = QtWidgets.QLabel()
        self.value.setObjectName("forceValue")
        layout.addWidget(self.value)
        self.flags = QtWidgets.QLabel()
        self.flags.setWordWrap(True)
        layout.addWidget(self.flags)

    def show_hold(self, hold):
        color = QtGui.QColor(*hold["peak_rgb"])
        swatch = QtGui.QPixmap(52, 10)
        swatch.fill(color)
        self.swatch.setPixmap(swatch)
        self.swatch.setToolTip(f"峰值 {hold['peak_adc']:g} ADC · {color.name()}")
        self.value.setText(f"总力 {_force_text(hold['estimated_n'])} N")
        self.flags.setText(_flags(hold))
        self.flags.setVisible(bool(self.flags.text()))


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
            QLabel#loadGuide, QLabel#loadSummary { color: #f1f5f5; font-size: 18px; font-weight: bold; }
            QLabel#forceValue { color: #f1f5f5; font-size: 16px; }
            QPushButton#hold { background: #315d60; color: white; padding: 7px 14px; }
            QToolButton { color: #bfd0d8; }
            QDoubleSpinBox, QLineEdit, QTableWidget, QHeaderView::section {
                background: #25343f; color: #e5eef2; }
        """)
        self.setFixedWidth(448)
        self._layout_state = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(8)
        self.guide = QtWidgets.QLabel()
        self.guide.setObjectName("loadGuide")
        self.guide.setWordWrap(True)
        layout.addWidget(self.guide)
        self.inputs = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.inputs)
        form.setContentsMargins(0, 0, 0, 0)
        self.check_force, self.calibration_force = _force_input(), _force_input()
        form.addRow("较低载荷 · 验证", self.check_force)
        form.addRow("较高载荷 · 标定", self.calibration_force)
        layout.addWidget(self.inputs)

        self.current_force = QtWidgets.QLabel()
        self.current_force.setObjectName("forceValue")
        layout.addWidget(self.current_force)
        self.feedback = QtWidgets.QLabel()
        self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)
        self.capture_button = QtWidgets.QPushButton("记录本步")
        self.capture_button.setObjectName("hold")
        self.zero_button = QtWidgets.QPushButton("采集无载参考")
        for button in (self.capture_button, self.zero_button):
            button.setAutoDefault(False)
            layout.addWidget(button)

        self.summary = QtWidgets.QLabel()
        self.summary.setObjectName("loadSummary")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.comparison = QtWidgets.QWidget()
        samples = QtWidgets.QHBoxLayout(self.comparison)
        samples.setContentsMargins(0, 0, 0, 0)
        self.readings = {}
        for key, label in (("check_up", "夹紧 · 峰值色"), ("check_down", "松回 · 峰值色"),
                           ("release", "完全松开 · 峰值色")):
            self.readings[key] = _PeakReading(label)
            samples.addWidget(self.readings[key], 1, QtCore.Qt.AlignTop)
        layout.addWidget(self.comparison)

        self.details_button = QtWidgets.QToolButton()
        self.details_button.setText("详情与操作")
        self.details_button.setCheckable(True)
        self.details_button.setAutoRaise(True)
        self.details_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.details_button.toggled.connect(self.refresh)
        layout.addWidget(self.details_button, alignment=QtCore.Qt.AlignLeft)
        self.details = QtWidgets.QWidget()
        extra = QtWidgets.QVBoxLayout(self.details)
        extra.setContentsMargins(0, 0, 0, 0)
        self.live = QtWidgets.QLabel()
        self.live.setWordWrap(True)
        extra.addWidget(self.live)
        self.results = QtWidgets.QTableWidget(0, 5)
        self.results.setHorizontalHeaderLabels(["测项", "已知 N", "估算 N", "误差 N", "峰值 ADC"])
        self.results.horizontalHeaderItem(3).setToolTip("估算总法向力减去测力计读数")
        self.results.verticalHeader().hide()
        self.results.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.results.setWordWrap(False)
        self.results.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        extra.addWidget(self.results)
        self.location = QtWidgets.QLineEdit()
        self.location.setAccessibleName("复测位置或压头")
        self.location.setMaxLength(80)
        self.location.setPlaceholderText("复测位置或压头（选填）")
        extra.addWidget(self.location)
        self.repeat_button = QtWidgets.QPushButton("同载荷复测")
        self.reset_button = QtWidgets.QPushButton("重新开始")
        self.export_button = QtWidgets.QPushButton("导出报告")
        actions = QtWidgets.QHBoxLayout()
        for button in (self.repeat_button, self.export_button, self.reset_button):
            button.setAutoDefault(False)
            actions.addWidget(button)
        extra.addLayout(actions)
        layout.addWidget(self.details)
        self.zero_button.clicked.connect(viewer.start_reference)
        self.repeat_button.clicked.connect(self.start_hold)
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
        status = "读数有波动，已保留本次测量。" if not held.stable and self.trial.stage != "done" else ""
        self.feedback.setText(status)

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
            flags = _flags(check)
            if flags:
                label += "\n" + flags
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
        for key, reading in self.readings.items():
            reading.show_hold(report["holds"][key])
        delta = report["unloading_minus_loading_n"]
        relative = "松回较高" if delta > 0 else ("松回较低" if delta < 0 else "两次相同")
        self.summary.setText(f"松回 − 夹紧　{_force_text(delta, signed=True)} N（{relative}）\n"
                             f"松开残余　{_force_text(report['release_residual_n'])} N")

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
        connected = bool(model.message and model.message.connected)
        collecting = bool(model.current_reference and model.current_reference.phase == "collecting")
        ready = context is not None and not partial_reference and not model.replay
        done = stage == "done"
        if model.replay:
            guide = "请在实时采集窗口验证载荷"
        elif not connected and context is None:
            guide = "请接入夹爪触觉阵列"
        elif partial_reference:
            guide = f"无载参考覆盖 {reference_count}/{CELL_COUNT} 点，请松开夹爪重新采集"
        elif collecting:
            guide = "采集无载参考中，请保持夹爪松开"
        elif context is None:
            guide = "松开夹爪，采集无载参考"
        elif check > 0 and 0 < high <= check:
            guide = "标定载荷应大于验证载荷"
        elif not valid_forces:
            guide = "输入测力计的两档载荷"
        else:
            guide = _STEP_TEXT[stage].format(check=check, high=high)
        self.guide.setText(guide)
        estimate = self.trial.estimate(obs.total_adc) if obs is not None and self.trial is not None else None
        if obs is None:
            fresh_count = int(model.mask(self.viewer.clock_ns()).sum())
            live = f"新鲜测点 {fresh_count}/{CELL_COUNT}，等候完整、与参考配对的触觉响应。"
        else:
            live = f"接触 {obs.active_cells} 点　峰值 {obs.peak_adc:g} ADC"
            if obs.peak_adc >= context.span_adc:
                live += "（色标已满）"
            live += f"\n响应总和 {obs.total_adc:g} ADC　有符号总和 {obs.signed_total_adc:+g} ADC"
            if estimate is not None:
                live += f"\n估算总法向力 {estimate:.3f} N"
                if estimate > self.trial.calibration_n:
                    live += "（超出标定范围，外推）"
        self.live.setText(live)
        show_estimate = estimate is not None and stage in ("check_down", "release")
        if show_estimate:
            range_hint = "（超出标定范围）" if estimate > self.trial.calibration_n else ""
            self.current_force.setText(f"当前估算　{_force_text(estimate)} N{range_hint}")
        self.current_force.setVisible(show_estimate)
        self.inputs.setVisible(ready and self.trial is None)
        self.check_force.setEnabled(self.trial is None)
        self.calibration_force.setEnabled(self.trial is None)
        button_text = "同载荷复测" if done else "记录本步"
        if ready and valid_forces and obs is None and not done:
            button_text = "等待完整的触觉读数"
        self.capture_button.setText("正在采集…" if self.capture else button_text)
        self.capture_button.setVisible(ready and not done)
        self.capture_button.setEnabled(self.capture is None and valid_forces and obs is not None)
        self.repeat_button.setEnabled(self.capture_button.isEnabled())
        self.repeat_button.setVisible(done)
        self.zero_button.setVisible(connected and not model.replay and not collecting and
                                    (context is None or partial_reference))
        self.zero_button.setEnabled(self.viewer.reference_button.isEnabled())
        show_details = bool(ready or partial_reference or self.last_report)
        expanded = show_details and self.details_button.isChecked()
        self.details_button.setVisible(show_details)
        self.details_button.setText("详情、复测与报告" if done else "详情与操作")
        self.details_button.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.details.setVisible(expanded)
        self.feedback.setVisible(bool(self.feedback.text()))
        self.reset_button.setVisible(self.trial is not None)
        self.location.setVisible(done)
        self.location.setEnabled(self.capture is None)
        self.export_button.setVisible(self.last_report is not None)
        self.results.setVisible(done)
        self.comparison.setVisible(done)
        self.summary.setVisible(done)
        self.export_button.setText("导出报告" if done else "导出上次报告")
        # Resize only when sections or feedback change, avoiding churn on every ROS frame.
        size_state = (guide, ready, self.trial is None, done, expanded, show_details,
                      bool(self.feedback.text()), show_estimate, self.zero_button.isVisible(), self.results.rowCount())
        if size_state != self._layout_state:
            self._layout_state = size_state
            self.layout().activate()
            self.resize(self.width(), self.sizeHint().height())

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
