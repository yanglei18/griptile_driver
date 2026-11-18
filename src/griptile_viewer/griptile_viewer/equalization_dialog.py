"""Capture two independent uniform-pressure holds for relative taxel equalization."""

import time

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from griptile_core.equalization import fit_equalization, measure_uniform_hold, validate_equalization


class _UniformCapture:
    """Retain a consecutive window of disjoint full-array scans after the click."""

    MIN_SAMPLES = 12
    DURATION_NS = 1_000_000_000
    MAX_GAP_NS = 250_000_000
    TIMEOUT_NS = 10_000_000_000

    def __init__(self, initial, wall_ns):
        self.context = initial.context
        self.started = wall_ns
        self.after_ns = initial.source_now_ns
        self.samples = []
        self.first_end_ns = None
        self.last_wall_ns = wall_ns
        self.interrupted = False

    def update(self, obs, wall_ns):
        if wall_ns - self.started >= self.TIMEOUT_NS:
            raise ValueError("采集超时，请检查阵列连接并重试当前档。")
        if obs is None:
            self.samples.clear()
            self.interrupted = True
            return None
        if obs.context != self.context:
            raise ValueError("连接或无载参考已变更。")
        if self.interrupted:
            # Rows acquired during a dropout cannot take part in a new hold.
            self.after_ns = obs.source_now_ns
            self.interrupted = False
            return None
        if min(obs.row_stamps) <= self.after_ns:
            return None
        end_ns = max(obs.row_stamps)
        if end_ns - self.after_ns > self.MAX_GAP_NS or wall_ns - self.last_wall_ns > self.MAX_GAP_NS:
            self.samples.clear()
        if not self.samples:
            self.first_end_ns = end_ns
        self.samples.append(obs.raw_adc.copy())
        self.after_ns = end_ns
        self.last_wall_ns = wall_ns
        if len(self.samples) >= self.MIN_SAMPLES and end_ns - self.first_end_ns >= self.DURATION_NS:
            return np.stack(self.samples)
        return None


class EqualizationDialog(QtWidgets.QDialog):
    def __init__(self, viewer, wall_clock=time.monotonic_ns):
        super().__init__(viewer)
        self.viewer = viewer
        self.model = viewer.model
        self.wall_clock = wall_clock
        self.context = self.capture = self.fit = self.candidate = None
        self.setWindowTitle("GripTile · 阵元均压标定")
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setStyleSheet("QDialog {background: #182129;} QLabel#step {font-size: 17px; font-weight: bold;}")
        self.resize(448, 270)
        layout = QtWidgets.QVBoxLayout(self)
        self.guide = QtWidgets.QLabel()
        self.guide.setObjectName("step")
        self.guide.setWordWrap(True)
        layout.addWidget(self.guide)
        self.fixture = QtWidgets.QLabel("请用稳压气囊或均压夹具覆盖整个感应面。瓶盖和局部压头只适合测试接触形状。")
        self.fixture.setWordWrap(True)
        layout.addWidget(self.fixture)
        self.observation = QtWidgets.QLabel()
        self.observation.setWordWrap(True)
        layout.addWidget(self.observation)
        self.feedback = QtWidgets.QLabel()
        self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)
        self.capture_button = QtWidgets.QPushButton()
        self.adopt_button = QtWidgets.QPushButton()
        self.restart_button = QtWidgets.QPushButton("重新采集")
        row = QtWidgets.QHBoxLayout()
        for button in (self.capture_button, self.adopt_button, self.restart_button):
            button.setAutoDefault(False)
            row.addWidget(button)
        row.addStretch()
        layout.addLayout(row)
        self.saved_status = QtWidgets.QLabel()
        self.saved_status.setWordWrap(True)
        layout.addWidget(self.saved_status)
        self.enable_button = QtWidgets.QPushButton()
        self.clear_button = QtWidgets.QPushButton("移除系数")
        saved_actions = QtWidgets.QHBoxLayout()
        for button in (self.enable_button, self.clear_button):
            button.setAutoDefault(False)
            saved_actions.addWidget(button)
        saved_actions.addStretch()
        layout.addLayout(saved_actions)
        self.capture_button.clicked.connect(self.start_capture)
        self.adopt_button.clicked.connect(self.adopt)
        self.restart_button.clicked.connect(self.restart)
        self.enable_button.clicked.connect(self.toggle_profile)
        self.clear_button.clicked.connect(self.clear_profile)

    def closeEvent(self, event):
        self.capture = None
        super().closeEvent(event)

    def _discard(self):
        self.context = self.capture = self.fit = self.candidate = None

    def restart(self):
        self._discard()
        self.feedback.clear()
        self.refresh()

    def start_capture(self):
        self.refresh()
        if not self.capture_button.isEnabled() or self.capture is not None or self.candidate is not None:
            return
        obs = self.model.uniform_observation(self.viewer.clock_ns())
        if obs is None:
            return
        self.context = obs.context
        self.capture = _UniformCapture(obs, self.wall_clock())
        self.feedback.clear()
        self.refresh()

    def _accept_hold(self, samples):
        hold = measure_uniform_hold(samples, self.model.reference, self.context.direction)
        if self.fit is None:
            self.fit = fit_equalization(hold)
        else:
            self.candidate = validate_equalization(self.fit, hold)

    def adopt(self):
        control = self.viewer.equalization
        control.sync()
        self.refresh()
        if self.candidate is None or not self.adopt_button.isEnabled():
            return
        try:
            control.adopt(self.candidate, self.context)
        except (OSError, ValueError) as error:
            self.feedback.setText(str(error))
        else:
            self._discard()
            self.feedback.setText("验证通过的系数已启用。")
        self.refresh()

    def toggle_profile(self):
        control = self.viewer.equalization
        try:
            control.sync()
            if control.profile is not None:
                control.set_enabled(not control.enabled)
                self.feedback.clear()
        except (OSError, ValueError) as error:
            self.feedback.setText(str(error))
        self.refresh()

    def clear_profile(self):
        control = self.viewer.equalization
        control.sync()
        previous_profile = control.profile
        context = self.model.equalization_context
        if previous_profile is None:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "移除均压系数", "移除本传感器的均压系数？如仅需对照原图，可以选择停用。",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Cancel)
        if answer != QtWidgets.QMessageBox.Yes:
            return
        try:
            control.sync()
            if self.model.equalization_context != context or control.profile is not previous_profile:
                raise ValueError("传感器或系数已更换，请重新确认。")
            control.clear()
            self.feedback.setText("均压系数已移除。")
        except (OSError, ValueError) as error:
            self.feedback.setText(str(error))
        self.refresh()

    def refresh(self):
        context = self.model.equalization_context
        if self.context is not None and self.context != context:
            self._discard()
            self.feedback.setText("连接或无载参考已变更，请重新采集两档。")
        if not self.isVisible():
            return
        obs = self.model.uniform_observation(self.viewer.clock_ns()) if not self.model.replay else None
        if self.capture is not None:
            try:
                samples = self.capture.update(obs, self.wall_clock())
                if samples is not None:
                    self.capture = None
                    self._accept_hold(samples)
            except ValueError as error:
                self.capture = None
                self.feedback.setText(str(error))

        profile = self.candidate
        validating = self.fit is not None
        if profile is not None:
            self.guide.setText("两档采集完成 · 独立验证通过")
            self.observation.setText(
                f"有效 400/400　验证档空间变异系数 {profile.before_cv:.2%} → {profile.after_cv:.2%}\n"
                f"校正后最大相对偏差 {profile.after_max_error:.2%}")
        elif context is None or self.model.replay:
            self.guide.setText("等待实时阵列和无载参考")
            self.observation.setText("请连接传感器。若尚无无载参考，请松开夹爪后在主窗口采集。")
        else:
            self.guide.setText("2/2　提高均压并验证" if validating else "1/2　施加较低的均匀压力")
            covered = sum(self.model.reference.valid_mask)
            if covered != 400:
                self.observation.setText(f"无载参考覆盖 {covered}/400 点。请松开夹爪，在主窗口重新采集无载参考。")
            elif obs is None:
                self.observation.setText("等待 400 个触点的完整、新鲜读数。")
            elif self.capture is not None:
                self.observation.setText(f"已采集 {len(self.capture.samples)} 个完整扫描，请保持压力。")
            elif validating:
                self.observation.setText(
                    f"第一档中位响应 {self.fit.fit_level:.0f} ADC。提高整面压力，使响应至少增加 25%，"
                    "保持稳定后采集验证档。")
            else:
                self.observation.setText("无载参考已就绪。均匀加压并保持稳定后采集，约需 1 秒。")
        self.capture_button.setText("正在采集…" if self.capture else ("采集验证档" if validating else "采集第一档"))
        self.capture_button.setVisible(profile is None)
        self.capture_button.setEnabled(obs is not None and self.capture is None and profile is None)
        self.restart_button.setVisible(self.capture is not None or validating)
        self.adopt_button.setVisible(profile is not None)
        self.adopt_button.setEnabled(profile is not None and obs is not None)
        self.adopt_button.setText("保存并启用" if self.model.binding() else "本次连接启用")
        self.feedback.setVisible(bool(self.feedback.text()))

        control = self.viewer.equalization
        existing = control.profile
        if existing is not None:
            status = "均压校正已启用" if control.enabled else "均压校正已停用"
            scope = "本机保存" if control.persistent else "仅本次连接"
            self.saved_status.setText(f"{status} · {scope} · 验证后变异系数 {existing.after_cv:.2%}")
        else:
            self.saved_status.setText(control.error)
        self.saved_status.setVisible(bool(self.saved_status.text()))
        self.enable_button.setText("停用校正" if control.enabled else "启用校正")
        for button in (self.enable_button, self.clear_button):
            button.setVisible(existing is not None)
            button.setEnabled(self.capture is None)
