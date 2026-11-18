"""Qt view; axes are native zero-based matrix addresses."""

import os
import time

# Prefer the same Qt binding as the ROS desktop distribution.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from griptile_core.contact_image import (
    CONTACT_COLOR_STOPS, DEFAULT_CONTACT_SPAN, INVALID_RGB, UNLOADED_RGB, contact_surface_rgb,
)

from griptile_core.grid import ROWS, COLS
from .load_dialog import LoadDialog
from .storage import load_reference, save_reference, save_snapshot


def _palette(colors):
    stops = np.linspace(0, 1, len(colors))
    return pg.ColorMap(stops, [pg.mkColor(color) for color in colors]).getLookupTable(nPts=256)


RAW = _palette(("#151629", "#383d87", "#157d98", "#45b88c", "#fee681"))
DELTA = _palette(("#2d5eaf", "#8ab9e6", "#f9f8ee", "#f2a272", "#a0373c"))


CONTACT_MODE, SURFACE_MODE, RAW_MODE, DELTA_MODE = range(4)


def _swatch(rgb, label):
    return '<span style="color:#%02x%02x%02x">■</span> %s' % (*rgb, label)


def _contact_legend():
    legend = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(legend)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addStretch()
    layout.addWidget(QtWidgets.QLabel(_swatch(UNLOADED_RGB, "无接触")))
    layout.addSpacing(12)
    layout.addWidget(QtWidgets.QLabel("弱"))
    bar = QtWidgets.QFrame()
    bar.setObjectName("contactGradient")
    stops = ", ".join(f"stop:{position:g} {QtGui.QColor(*rgb).name()}"
                      for position, rgb in CONTACT_COLOR_STOPS)
    bar.setStyleSheet(f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {stops});")
    bar.setMinimumWidth(100)
    bar.setMaximumWidth(240)
    bar.setFixedHeight(12)
    bar.setToolTip("蓝：弱　青绿：中等　黄橙：较强　红：达到满量程")
    bar.setAccessibleName("接触强度色带")
    bar.setAccessibleDescription(bar.toolTip())
    layout.addWidget(bar, 1)
    layout.addWidget(QtWidgets.QLabel("强"))
    layout.addSpacing(12)
    missing = QtWidgets.QLabel(_swatch(INVALID_RGB, "缺测"))
    missing.setToolTip("没有有效测量或无载参考")
    layout.addWidget(missing)
    layout.addStretch()
    return legend


class ViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, model, clock_ns=time.monotonic_ns, request_reference=None):
        super().__init__()
        self.model = model
        self.clock_ns = clock_ns
        self.request_reference = request_reference
        self.last_receipt_wall_ns = time.monotonic_ns()
        self.last_message = None
        self.setWindowTitle(f"GripTile Viewer · {model.sensor_id}")
        self.setStyleSheet("""
            QMainWindow, QStatusBar { background: #182129; }
            QLabel, QStatusBar { color: #e0e9ed; font-size: 14px; }
            QLabel#status { font-size: 17px; font-weight: 600; padding: 4px 0; }
            QLabel#reference, QLabel#legend { color: #b9cad2; font-size: 13px; }
            QPushButton, QComboBox, QSpinBox, QToolButton {
                background: #25343f; color: #e5eef2; border: 1px solid #4a5b65;
                border-radius: 4px; padding: 5px 9px; min-height: 19px;
            }
            QPushButton:hover { background: #304752; border-color: #7bcbbd; }
            QPushButton#load { background: #28534f; border-color: #49847c; }
            QPushButton#load:hover { background: #33645e; }
            QPushButton:disabled, QSpinBox:disabled { color: #90a1ab; background: #202b34; }
            QComboBox QAbstractItemView { color: #e5eef2; background: #25343f; }
        """)
        self.resize(660, 620)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(18, 14, 18, 14)
        self.status = QtWidgets.QLabel()
        self.status.setObjectName("status")
        header = QtWidgets.QHBoxLayout()
        header.addWidget(self.status)
        self.ref_status = QtWidgets.QLabel()
        self.ref_status.setObjectName("reference")
        header.addWidget(self.ref_status)
        header.addStretch()
        self.reference_button = QtWidgets.QPushButton("采集无载参考")
        self.reference_button.setToolTip("保持夹爪无载，采集 3 秒")
        self.reference_button.clicked.connect(self.start_reference)
        header.addWidget(self.reference_button)
        self.load_dialog = None
        self.load_button = QtWidgets.QPushButton("已知载荷验证")
        self.load_button.setObjectName("load")
        self.load_button.clicked.connect(self.open_load_dialog)
        header.addWidget(self.load_button)
        self.details_button = QtWidgets.QToolButton()
        self.details_button.setText("详情")
        self.details_button.setCheckable(True)
        self.details_button.setArrowType(QtCore.Qt.RightArrow)
        self.details_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.details_button.setToolTip("显示模式、单点读数、参考和导出")
        header.addWidget(self.details_button)
        layout.addLayout(header)
        self.details = QtWidgets.QWidget()
        details_layout = QtWidgets.QVBoxLayout(self.details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        controls = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["接触图 RGB", "连续接触面（估算）", "原始 ADC", "相对无载参考 ΔADC"])
        self.mode.setCurrentIndex(SURFACE_MODE)
        controls.addWidget(self.mode)
        self.span_label = QtWidgets.QLabel("差值色限 ±")
        controls.addWidget(self.span_label)
        self.span = QtWidgets.QSpinBox()
        self.span.setRange(1, 65535)
        self.span.setValue(3000)
        controls.addWidget(self.span)
        controls.addStretch()
        details_layout.addLayout(controls)
        self.map = pg.PlotWidget(background="#182129")
        self.map.setLabel("bottom", "列")
        self.map.setLabel("left", "行")
        self.map.setAspectLocked()
        self.map.hideButtons()
        self.map.getViewBox().setMenuEnabled(False)
        self.map.setMouseEnabled(False, False)
        self.map.invertY()
        self.map.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
        self.map.setXRange(0, COLS, padding=0)
        self.map.setYRange(0, ROWS, padding=0)
        ticks = [(n + .5, str(n)) for n in (0, 4, 9, 14, 19)]
        for axis in ("left", "bottom"):
            self.map.getAxis(axis).setTicks([ticks])
        self.image = pg.ImageItem(axisOrder="row-major", autoDownsample=False)
        self.map.addItem(self.image)
        self.marker = self.map.plot([], [], pen=None, symbol="s", symbolPen=pg.mkPen("w", width=2),
                                    symbolBrush=None, symbolSize=13)
        self.map.scene().sigMouseClicked.connect(self.pick_cell)
        self.map.scene().sigMouseMoved.connect(self.inspect_cell)
        layout.addWidget(self.map, stretch=4)
        self.scale = _contact_legend()
        layout.addWidget(self.scale)
        self.cell_label = QtWidgets.QLabel("点击单元查看曲线")
        details_layout.addWidget(self.cell_label)
        self.curve = pg.PlotWidget(background="#171b21")
        self.curve.setMinimumHeight(80)
        self.curve.setMaximumHeight(145)
        self.curve.hideButtons()
        self.curve.setLabel("bottom", "相对最新样本", units="s")
        self.curve.setLabel("left", "原始 ADC")
        self.curve.showGrid(x=True, y=True, alpha=.2)
        self.line = self.curve.plot(pen=pg.mkPen("#72d8c5", width=2), connect="finite")
        details_layout.addWidget(self.curve)
        self.legend = QtWidgets.QLabel()
        self.legend.setObjectName("legend")
        self.legend.setWordWrap(True)
        details_layout.addWidget(self.legend)
        footer = QtWidgets.QHBoxLayout()
        self._file_button(footer, "保存参考", "JSON (*.json)", "save_ref")
        self._file_button(footer, "加载参考", "JSON (*.json)", "load_ref")
        self._file_button(footer, "导出 ADC 快照", "CSV (*.csv)", "csv")
        self.png_button = self._file_button(footer, "导出图像", "PNG (*.png)", "png")
        footer.addStretch()
        details_layout.addLayout(footer)
        layout.addWidget(self.details, stretch=2)
        self._format = None
        self._surface_input = self._surface_span = self._surface_pixels = None
        self.details_button.toggled.connect(self._show_details)
        self._show_details(False)

    def _show_details(self, expanded):
        self.details.setVisible(expanded)
        self.details_button.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        for axis in ("left", "bottom"):
            self.map.showAxis(axis, expanded)
        self.marker.setVisible(expanded)
        if not expanded:
            self.mode.setCurrentIndex(SURFACE_MODE)
        self.refresh()

    def _file_button(self, layout, caption, pattern, action):
        button = QtWidgets.QPushButton(caption)
        button.clicked.connect(lambda: self.file_action(caption, pattern, action))
        layout.addWidget(button)
        return button

    def file_action(self, caption, pattern, action):
        dialog = QtWidgets.QFileDialog
        chooser = dialog.getOpenFileName if action == "load_ref" else dialog.getSaveFileName
        path, _ = chooser(self, caption, "", pattern)
        if not path:
            return
        try:
            if action == "load_ref":
                self._request(load_reference(self.model, path))
            elif action == "save_ref":
                save_reference(self.model, path)
            elif action == "csv":
                save_snapshot(self.model, path, self.clock_ns())
            else:
                self.save_png(path)
        except (ValueError, OSError, TypeError) as error:
            self.statusBar().showMessage(str(error), 6000)

    def save_png(self, path):
        if self.mode.currentIndex() in (CONTACT_MODE, SURFACE_MODE):
            self.refresh()
            pixels = self.image.image
            height, width = pixels.shape[:2]
            image = QtGui.QImage(pixels.data, width, height, pixels.strides[0], QtGui.QImage.Format_RGB888)
        else:
            image = self.map.grab()
        if not image.save(str(path), "PNG"):
            raise OSError("无法保存图像")

    def _request(self, profile=None):
        if self.request_reference is None:
            raise ValueError("等待驱动的无载参考服务")
        self.request_reference(profile)
        if self.mode.currentIndex() != SURFACE_MODE:
            self.mode.setCurrentIndex(CONTACT_MODE)

    def open_load_dialog(self):
        if self.load_dialog is None:
            self.load_dialog = LoadDialog(self)
            # Place the guide beside the contact map on first open. Reopening
            # respects any position the operator chose during the experiment.
            screen = self.screen().availableGeometry()
            frame = self.frameGeometry()
            width = self.load_dialog.width()
            if screen.right() - frame.right() >= frame.left() - screen.left():
                left = min(frame.right() + 8, screen.right() - width + 1)
            else:
                left = max(frame.left() - width - 8, screen.left())
            top = min(frame.top(), screen.bottom() - self.load_dialog.height() + 1)
            self.load_dialog.move(max(left, screen.left()), max(top, screen.top()))
        self.load_dialog.show()
        self.load_dialog.raise_()
        self.load_dialog.activateWindow()
        self.load_dialog.refresh()

    def start_reference(self):
        try:
            self._request()
        except ValueError as error:
            self.statusBar().showMessage(str(error), 5000)

    def cell_at(self, position):
        data = self.map.getViewBox().mapSceneToView(position)
        if 0 <= data.y() < ROWS and 0 <= data.x() < COLS:
            return int(data.y()), int(data.x())
        return None

    def pick_cell(self, event):
        cell = self.cell_at(event.scenePos())
        if cell is not None:
            self.model.select(*cell)

    def inspect_cell(self, position):
        if not self.details_button.isChecked():
            return
        cell = self.cell_at(position)
        if cell is not None:
            delta = self.mode.currentIndex() == DELTA_MODE
            value = self.model.values(self.clock_ns(), delta)[cell]
            label = f"{value:g} {'ΔADC' if delta else 'ADC'}" if np.isfinite(value) else "无有效值"
            self.statusBar().showMessage(f"行 {cell[0]}  列 {cell[1]}    {label}", 1500)

    def _surface(self, amplitude, span):
        if self._surface_span != span or not np.array_equal(self._surface_input, amplitude, equal_nan=True):
            self._surface_pixels = contact_surface_rgb(amplitude, span)
            self._surface_input, self._surface_span = amplitude, span
        return self._surface_pixels

    def refresh(self):
        now = self.clock_ns()
        model = self.model
        model.track_clock(now)
        message = model.message
        wall_ns = time.monotonic_ns()
        if message is not self.last_message:
            self.last_message = message
            self.last_receipt_wall_ns = wall_ns
        replay_pause = model.replay and wall_ns - self.last_receipt_wall_ns > 1_000_000_000
        valid = model.mask(now)
        mode = self.mode.currentIndex()
        delta = mode == DELTA_MODE
        contact = mode in (CONTACT_MODE, SURFACE_MODE)
        surface = mode == SURFACE_MODE
        self.span.setVisible(delta)
        self.span_label.setVisible(delta)
        self.scale.setVisible(contact)
        mode_changed = self._format != mode
        if mode_changed:
            self.image.setLookupTable({RAW_MODE: RAW, DELTA_MODE: DELTA}.get(mode))
            self.png_button.setText("导出估算图" if surface else "导出图像")
            self._format = mode
        if contact:
            if surface:
                amplitude = model.contact_response(now)
                status = model.current_reference
                span = status.span_adc if model.reference is not None else DEFAULT_CONTACT_SPAN
                frame = self._surface(amplitude, span)
                usable = np.isfinite(amplitude)
                active = amplitude > 0
            else:
                frame = model.contact_pixels(now)
                usable = np.any(frame != INVALID_RGB, axis=2)
                active = usable & np.any(frame != UNLOADED_RGB, axis=2)
            self.image.setImage(frame, autoLevels=False, levels=None)
            summary = (f"接触 {active.sum()} 点　有效像素 {usable.sum()}/400" if usable.any()
                       else "接触点待测　有效像素 0/400")
            if surface:
                summary = "300×300 插值显示 · 20×20 实测点　" + summary
        else:
            levels = (-self.span.value(), self.span.value()) if delta else (0, 65535)
            self.image.setImage(model.values(now, delta), autoLevels=False, levels=levels)
            summary = f"色域 {levels[0]} 至 {levels[1]} {'ΔADC' if delta else 'ADC'}　灰色为无效单元"
        if mode_changed:
            # Both image resolutions occupy the same native sensor coordinates.
            self.image.setRect(QtCore.QRectF(0, 0, COLS, ROWS))
        row, col = model.selected
        self.marker.setData([col + .5], [row + .5])
        raw = model.values(now)[row, col]
        value = model.values(now, True)[row, col]
        raw_text = f"{raw:g}" if np.isfinite(raw) else "—"
        delta_text = f"{value:+g}" if np.isfinite(value) else "—"
        self.cell_label.setText(f"行 {row}  列 {col}    ADC {raw_text}    ΔADC {delta_text}")
        if self.details_button.isChecked():
            if model.trend:
                stamps, values = zip(*model.trend)
                self.line.setData([(stamp - stamps[-1]) / 1e9 for stamp in stamps], values)
            else:
                self.line.setData([], [])

        if message is None:
            state = "等待阵列数据"
        elif not message.connected:
            state = "已断开"
        elif replay_pause:
            state = "回放暂停或等待下一帧"
        elif not valid.any():
            state = "等待新数据"
        else:
            state = "已连接"
        self.status.setText(f"{'回放' if model.replay else '实时'} · {state}")
        status = model.current_reference
        phase = status.phase if status else "unreferenced"
        covered = sum(model.reference.valid_mask) if model.reference is not None else 0
        ref_info = ""
        if phase == "ready" and model.reference:
            text = "参考就绪" if covered == ROWS * COLS else f"参考 {covered}/400"
            ref_info = f"参考 {covered}/400　满量程 {status.span_adc:g} ADC"
        elif phase == "collecting":
            text = f"保持无载 {status.progress:.0%}"
        elif phase == "failed":
            text = "参考未完成"
            ref_info = status.reason
        else:
            text = "待采无载参考" if valid.any() else ""
        self.ref_status.setText(text)
        can_collect = self.request_reference is not None and bool(valid.any()) and phase != "collecting"
        self.reference_button.setEnabled(can_collect)
        self.reference_button.setVisible(can_collect and (covered < ROWS * COLS or self.details_button.isChecked()))
        reason = message.reason if message is not None and not valid.any() else ""
        self.legend.setText("    ".join(filter(None, (model.sensor_id, summary, ref_info, reason, model.diagnostic,
                                                    model.rejection, model.service_error))))
        if self.load_dialog is not None:
            self.load_dialog.refresh()
