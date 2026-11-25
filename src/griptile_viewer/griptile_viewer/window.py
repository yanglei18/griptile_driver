"""Qt view; axes are native zero-based matrix addresses."""

import os
import time

# Prefer the same Qt binding as the ROS desktop distribution.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from griptile_core.contact_image import (
    CONTACT_COLOR_STOPS, CONTACT_MAP_STOPS, CONTACT_OUTLINE_RGB, DEFAULT_CONTACT_SPAN, INVALID_RGB,
    SURFACE_SCALE, UNLOADED_RGB, amplitude_rgb, contact_map_rgb, footprint_surface,
)

from griptile_core.equalization import apply_equalization
from griptile_core.grid import ROWS, COLS
from .equalization_store import EqualizationSettings
from .equalization_dialog import EqualizationDialog
from .contact_print_dialog import ContactPrintPage
from .load_dialog import LoadDialog
from .serial_dialog import SerialDialog
from .storage import load_reference, save_reference, save_snapshot


def _palette(colors):
    stops = np.linspace(0, 1, len(colors))
    return pg.ColorMap(stops, [pg.mkColor(color) for color in colors]).getLookupTable(nPts=256)


RAW = _palette(("#151629", "#383d87", "#157d98", "#45b88c", "#fee681"))
DELTA = _palette(("#2d5eaf", "#8ab9e6", "#f9f8ee", "#f2a272", "#a0373c"))


CONTACT_MODE, SURFACE_MODE, RAW_MODE, DELTA_MODE = range(4)
# Earlier published RGB8 images reserve white for a valid unloaded taxel.
_LEGACY_UNLOADED_RGB = (255, 255, 255)


def _contact_legend(peak):
    legend = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(legend)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(7)
    layout.addWidget(QtWidgets.QLabel("无载"))
    bar = QtWidgets.QFrame()
    bar.setObjectName("contactGradient")
    bar.setMinimumWidth(36)
    bar.setFixedHeight(6)
    bar.setAccessibleName("接触强度色带")
    layout.addWidget(bar, 1)
    layout.addWidget(QtWidgets.QLabel("强接触"))
    layout.addSpacing(6)
    layout.addWidget(peak)
    return legend


def _set_contact_legend(legend, surface):
    bar = legend.findChild(QtWidgets.QFrame, "contactGradient")
    palette = CONTACT_MAP_STOPS if surface else CONTACT_COLOR_STOPS
    stops = ", ".join(f"stop:{position:g} {QtGui.QColor(*rgb).name()}" for position, rgb in palette)
    bar.setStyleSheet(f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {stops});")
    text = "无接触为深色，接触由蓝到紫到红。灰色表示缺测。" if surface else "响应由浅蓝灰、青、绿、黄到红。灰色表示缺测。"
    bar.setAccessibleDescription(text)
    bar.setToolTip(text)


def _peak_label(amplitude, usable, span):
    if not usable.any():
        return "峰值 —"
    ratio = amplitude[usable].max() / span
    value = "<1%" if 0 < ratio < .01 else f"{ratio:.0%}"
    label = "峰值" if usable.all() else "局部峰值"
    return f"{label} {value}"


class _MonitorLayout(QtWidgets.QVBoxLayout):
    """Fit the entire square array and its bezel into either window orientation."""

    def __init__(self, frame, plot):
        super().__init__()
        self.frame = frame
        self.plot = plot
        self.compact = True
        self.setContentsMargins(0, 0, 0, 0)
        self.addWidget(frame)

    def _bezel_size(self):
        body = self.frame.layout()
        margins = body.contentsMargins()
        plot_item = body.itemAt(body.indexOf(self.plot))
        return margins.left() + margins.right(), body.sizeHint().height() - plot_item.sizeHint().height()

    def minimumSize(self):
        size = super().minimumSize()
        if self.compact:
            horizontal, vertical = self._bezel_size()
            size.setHeight(max(size.height(), size.width() - horizontal + vertical))
        return size

    def setGeometry(self, rect):
        if self.compact:
            horizontal, vertical = self._bezel_size()
            side = max(0, min(rect.width() - horizontal, rect.height() - vertical))
            fitted = QtCore.QRect(0, 0, side + horizontal, side + vertical)
            fitted.moveCenter(rect.center())
            rect = fitted
        super().setGeometry(rect)


class ViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, model, clock_ns=time.monotonic_ns, request_reference=None, serial_control=None):
        super().__init__()
        self.model = model
        self.clock_ns = clock_ns
        self.request_reference = request_reference
        self._last_reference_error = ""
        self.serial_control = serial_control
        self.serial_dialog = None
        self.equalization = EqualizationSettings(model)
        self.equalization_dialog = None
        self.contact_print_page = None
        self.last_receipt_wall_ns = time.monotonic_ns()
        self.last_message = None
        self.setWindowTitle(f"GripTile · 触觉显示 · {model.sensor_id}")
        self.setStyleSheet("""
            QMainWindow, QStatusBar { background: #151c22; }
            QLabel, QCheckBox, QStatusBar { color: #e0e9ed; font-size: 14px; }
            QFrame#contactMonitor { background: #273038; border-radius: 13px; }
            QFrame#contactMonitor QLabel { color: #d6dee4; font-size: 12px; }
            QFrame#contactMonitor QLabel#brand { color: #f0f5f7; font-weight: 600; font-size: 18px; }
            QFrame#contactMonitor QLabel#contactGuide { background: #202a30; color: #e0e9ed; font-size: 16px; padding: 10px; }
            QLabel#reference, QLabel#legend { color: #b9cad2; font-size: 13px; }
            QPushButton, QComboBox, QSpinBox, QToolButton {
                background: #25343f; color: #e5eef2; border: 1px solid #4a5b65;
                border-radius: 4px; padding: 5px 9px; min-height: 19px;
            }
            QPushButton:hover { background: #304752; border-color: #7bcbbd; }
            QPushButton#contactPrint { color: #f4d6ad; border-color: #876c51; }
            QPushButton:disabled, QSpinBox:disabled { color: #90a1ab; background: #202b34; }
            QComboBox QAbstractItemView { color: #e5eef2; background: #25343f; }
        """)
        self.resize(560, 620)
        central = QtWidgets.QWidget()
        self.pages = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.pages)
        self.pages.addWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        self.monitor = QtWidgets.QFrame()
        self.monitor.setObjectName("contactMonitor")
        monitor_layout = QtWidgets.QVBoxLayout(self.monitor)
        monitor_layout.setContentsMargins(12, 7, 12, 8)
        monitor_layout.setSpacing(7)
        header = QtWidgets.QHBoxLayout()
        self.brand = QtWidgets.QLabel("GripTile")
        self.brand.setObjectName("brand")
        header.addWidget(self.brand)
        self.contact_print_button = QtWidgets.QPushButton("触印")
        self.contact_print_button.setObjectName("contactPrint")
        self.contact_print_button.setAccessibleDescription("查看当前接触，也可回看上次留印")
        self.contact_print_button.clicked.connect(self.open_contact_print)
        header.addStretch()
        self.status = QtWidgets.QLabel()
        self.status.setObjectName("status")
        self.ref_status = QtWidgets.QLabel()
        self.ref_status.setObjectName("reference")
        header.addWidget(self.ref_status)
        header.addWidget(self.status)
        monitor_layout.addLayout(header)
        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(6)
        self.reference_button = QtWidgets.QPushButton("空载归零")
        self.reference_button.setAccessibleDescription("完全松开夹爪后，采集空载读数")
        self.reference_button.clicked.connect(self.start_reference)
        actions.addWidget(self.reference_button)
        actions.addWidget(self.contact_print_button)
        actions.addStretch()
        self.details_button = QtWidgets.QToolButton()
        self.details_button.setText("详情")
        self.details_button.setCheckable(True)
        self.details_button.setArrowType(QtCore.Qt.RightArrow)
        self.details_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.details_button.setAccessibleDescription("串口设置、标定、显示模式与导出")
        actions.addWidget(self.details_button)
        self.details = QtWidgets.QWidget()
        details_layout = QtWidgets.QVBoxLayout(self.details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        tools = QtWidgets.QHBoxLayout()
        self.serial_button = QtWidgets.QPushButton("串口设置")
        self.serial_button.setVisible(not model.replay)
        self.serial_button.clicked.connect(self.open_serial_dialog)
        tools.addWidget(self.serial_button)
        self.equalization_button = QtWidgets.QPushButton("阵元标定")
        self.equalization_button.setToolTip("使用均匀载荷标定各触点的相对灵敏度")
        self.equalization_button.setVisible(not model.replay)
        self.equalization_button.clicked.connect(self.open_equalization_dialog)
        tools.addWidget(self.equalization_button)
        self.load_dialog = None
        self.load_button = QtWidgets.QPushButton("已知载荷验证")
        self.load_button.clicked.connect(self.open_load_dialog)
        tools.addWidget(self.load_button)
        tools.addStretch()
        details_layout.addLayout(tools)
        controls = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["接触图 RGB", "连续接触面（估算）", "原始 ADC", "相对无载参考 ΔADC"])
        self.mode.setCurrentIndex(SURFACE_MODE)
        controls.addWidget(self.mode)
        self.contour_checkbox = QtWidgets.QCheckBox("接触轮廓")
        self.contour_checkbox.setToolTip("按各触点的无载波动描出连续接触区的边缘。仅用于屏幕显示。")
        self.contour_checkbox.setChecked(True)
        controls.addWidget(self.contour_checkbox)
        self.span_label = QtWidgets.QLabel("差值色限 ±")
        controls.addWidget(self.span_label)
        self.span = QtWidgets.QSpinBox()
        self.span.setRange(1, 65535)
        self.span.setValue(3000)
        controls.addWidget(self.span)
        controls.addStretch()
        details_layout.addLayout(controls)
        self.map = pg.PlotWidget(background="#273038")
        guide_layout = QtWidgets.QVBoxLayout(self.map.viewport())
        guide_layout.setContentsMargins(0, 0, 0, 0)
        self.map_hint = QtWidgets.QLabel()
        self.map_hint.setObjectName("contactGuide")
        self.map_hint.setAlignment(QtCore.Qt.AlignCenter)
        self.map_hint.setWordWrap(True)
        self.map_hint.setTextFormat(QtCore.Qt.PlainText)
        self.map_hint.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        guide_layout.addWidget(self.map_hint)
        self.map.getPlotItem().layout.setContentsMargins(0, 0, 0, 0)
        self.map.setMinimumSize(160, 160)
        self.map.setLabel("bottom", "列")
        self.map.setLabel("left", "行")
        self.map.setAspectLocked()
        self.map.hideButtons()
        self.map.getViewBox().setMenuEnabled(False)
        self.map.setMouseEnabled(False, False)
        self.map.invertY()
        self.map.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
        self.map.setRange(rect=QtCore.QRectF(0, 0, COLS, ROWS), padding=0)
        ticks = [(n + .5, str(n)) for n in (0, 4, 9, 14, 19)]
        for axis in ("left", "bottom"):
            self.map.getAxis(axis).setTicks([ticks])
        self.image = pg.ImageItem(axisOrder="row-major", autoDownsample=False)
        self.map.addItem(self.image)
        # Keep the visual annotation separate from numeric colors, inspection and PNG export.
        self.contour_image = pg.ImageItem(axisOrder="row-major", autoDownsample=False)
        self.contour_image.setLookupTable(np.array([(0, 0, 0, 0), (*CONTACT_OUTLINE_RGB, 255)], dtype=np.uint8))
        self.contour_image.setTransform(QtGui.QTransform.fromScale(1 / SURFACE_SCALE, 1 / SURFACE_SCALE))
        self.contour_image.setAcceptedMouseButtons(QtCore.Qt.NoButton)
        self.contour_image.hide()
        self.map.addItem(self.contour_image, ignoreBounds=True)
        self.marker = self.map.plot([], [], pen=None, symbol="s", symbolPen=pg.mkPen("w", width=2),
                                    symbolBrush=None, symbolSize=13)
        self.map.scene().sigMouseClicked.connect(self.pick_cell)
        self.map.scene().sigMouseMoved.connect(self.inspect_cell)
        monitor_layout.addWidget(self.map, stretch=1)
        self.peak = QtWidgets.QLabel("峰值 —")
        self.peak.setTextFormat(QtCore.Qt.PlainText)
        self.peak.setToolTip("当前有效触点的最大响应，占固定显示量程的百分比")
        self.scale = _contact_legend(self.peak)
        monitor_layout.addWidget(self.scale)
        self.source_scale = QtWidgets.QLabel()
        self.source_scale.setAlignment(QtCore.Qt.AlignCenter)
        monitor_layout.addWidget(self.source_scale)
        self._monitor_layout = _MonitorLayout(self.monitor, self.map)
        layout.addLayout(self._monitor_layout, stretch=4)
        layout.addLayout(actions)
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
        self._outline_enabled = False
        self.contour_checkbox.toggled.connect(self.refresh)
        self.details_button.toggled.connect(self._show_details)
        self._show_details(False)

    def _show_details(self, expanded):
        self.details.setVisible(expanded)
        self._monitor_layout.compact = not expanded
        self._monitor_layout.invalidate()
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
        self.refresh(render_array=True)
        mode = self.mode.currentIndex()
        if mode in (CONTACT_MODE, SURFACE_MODE):
            if self.equalization.active is None:
                pixels = self.image.image
            else:
                # PNG has always carried the original contact palette, including the
                # driver's exact RGB on native-only streams. Keep this export stable.
                now = self.clock_ns()
                span = self.model.current_reference.span_adc
                if mode == SURFACE_MODE:
                    contact, _ = self.model.contact_surface_inputs(now, confirmed_only=True)
                    pixels = contact_map_rgb(contact, span)
                else:
                    amplitude = self.model.contact_response(now, allow_rgb_fallback=True)
                    pixels = self.model.contact_pixels(now) if amplitude is None else amplitude_rgb(amplitude, span)
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
        self._last_reference_error = ""
        if self.mode.currentIndex() != SURFACE_MODE:
            self.mode.setCurrentIndex(CONTACT_MODE)

    def open_serial_dialog(self):
        if self.model.replay:
            return
        if self.serial_dialog is None:
            self.serial_dialog = SerialDialog(self, self.serial_control)
        self.serial_dialog.open_settings()

    @property
    def contact_print_dialog(self):
        """Archived replay scripts access the active print through this name."""
        return self.contact_print_page

    def open_contact_print(self):
        if self.contact_print_page is None:
            self.contact_print_page = ContactPrintPage(self.pages, self.model.contact_prints)
            self.pages.addWidget(self.contact_print_page)
            self.contact_print_page.back_button.clicked.connect(self.show_array)
        self.pages.setCurrentWidget(self.contact_print_page)
        self.setWindowTitle(f"GripTile · 触印 · {self.model.sensor_id}")
        self.contact_print_page.back_button.setFocus()

    def show_array(self):
        self.pages.setCurrentIndex(0)
        self.setWindowTitle(f"GripTile · 触觉显示 · {self.model.sensor_id}")
        self.contact_print_button.setFocus()
        self.refresh()

    def open_equalization_dialog(self):
        if self.model.replay:
            return
        if self.equalization_dialog is None:
            self.equalization_dialog = EqualizationDialog(self)
        self.equalization_dialog.show()
        self.equalization_dialog.raise_()
        self.equalization_dialog.activateWindow()
        self.equalization_dialog.refresh()

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
        changed = not np.array_equal(self._surface_input, amplitude, equal_nan=True)
        if changed or self._surface_span != span:
            self._surface_pixels = contact_map_rgb(amplitude, span)
        outline = self.contour_checkbox.isChecked()
        if outline and (changed or not self._outline_enabled):
            _, edge = footprint_surface(amplitude > 0, np.isfinite(amplitude))
            if edge.any():
                self.contour_image.setImage(edge.astype(np.uint8), autoLevels=False, levels=(0, 1))
            else:
                self.contour_image.clear()
        self.contour_image.setVisible(outline and self.contour_image.image is not None)
        self._outline_enabled = outline
        self._surface_input, self._surface_span = amplitude, span
        return self._surface_pixels

    def refresh(self, *, render_array=False):
        """Advance acquisition and active tools; explicitly requested exports also need a fresh array."""
        now = self.clock_ns()
        model = self.model
        model.track_clock(now)
        self.equalization.sync()
        model.update_contact_print(now, self.equalization.active)
        if self.contact_print_page is not None:
            self.contact_print_page.refresh()
        message = model.message
        wall_ns = time.monotonic_ns()
        if message is not self.last_message:
            self.last_message = message
            self.last_receipt_wall_ns = wall_ns
        if render_array or self.pages.currentIndex() == 0:
            replay_pause = model.replay and wall_ns - self.last_receipt_wall_ns > 1_000_000_000
            self._refresh_array(now, replay_pause)
        if self.equalization_dialog is not None:
            self.equalization_dialog.refresh()
        if self.load_dialog is not None:
            self.load_dialog.refresh()
        if self.serial_dialog is not None:
            self.serial_dialog.refresh_view()

    def _refresh_array(self, now, replay_pause):
        model = self.model
        message = model.message
        valid = model.mask(now)
        mode = self.mode.currentIndex()
        delta = mode == DELTA_MODE
        contact = mode in (CONTACT_MODE, SURFACE_MODE)
        surface = mode == SURFACE_MODE
        self.contour_checkbox.setVisible(surface)
        if not surface:
            self.contour_image.hide()
        self.span.setVisible(delta)
        self.span_label.setVisible(delta)
        source_colors = False
        profile = self.equalization.active if contact else None
        self.png_button.setToolTip("保存均衡前的原始色标图" if profile is not None else "")
        self.png_button.setText("导出未均衡图" if profile is not None
                                else "导出估算图" if surface else "导出图像")
        mode_changed = self._format != mode
        if mode_changed:
            self.image.setLookupTable({RAW_MODE: RAW, DELTA_MODE: DELTA}.get(mode))
            if contact:
                _set_contact_legend(self.scale, surface)
            self._format = mode
        if contact:
            thresholds = None
            if profile is not None or surface:
                amplitude, thresholds = model.contact_surface_inputs(now, confirmed_only=surface)
            elif model.replay:
                amplitude = None  # Preserve the exact recorded RGB8 in the native view.
            else:
                amplitude = model.contact_response(now, allow_rgb_fallback=True)
            if amplitude is not None:
                status = model.current_reference
                span = status.span_adc if model.reference is not None else DEFAULT_CONTACT_SPAN
                measured = amplitude
                if profile is not None:
                    # A ready-to-ready reference switch may have a mixed grid. Only
                    # color rows known to have been sampled after the matched reference.
                    supported = False if thresholds is None else np.isfinite(thresholds)
                    amplitude = apply_equalization(np.where(supported, measured, np.nan), profile)
                frame = self._surface(amplitude, span) if surface else amplitude_rgb(amplitude, span)
                usable = np.isfinite(amplitude)
                active = amplitude > 0
                self.peak.setText(_peak_label(amplitude, usable, span))
            else:
                self.peak.setText("峰值 —")
                frame = model.contact_pixels(now)
                usable = np.any(frame != INVALID_RGB, axis=2)
                unloaded = np.all(frame == UNLOADED_RGB, axis=2)
                legacy_unloaded = np.all(frame == _LEGACY_UNLOADED_RGB, axis=2)
                active = usable & ~(unloaded | legacy_unloaded)
                # RGB8 carries no palette metadata. The new unloaded color
                # identifies current frames; older full-contact images can be ambiguous.
                source_colors = bool(usable.any() and not unloaded.any())
                if source_colors:
                    self.source_scale.setText("源图配色　白色无载　灰色缺测" if legacy_unloaded.any()
                                              else "源图配色　灰色缺测")
            self.image.setImage(frame, autoLevels=False, levels=None)
            summary = (f"有响应 {active.sum()} 点　有效像素 {usable.sum()}/400" if usable.any()
                       else "等待响应数据　有效像素 0/400")
            if surface:
                summary = f"接触 {active.sum()} 点　有效 {usable.sum()}/400"
        else:
            levels = (-self.span.value(), self.span.value()) if delta else (0, 65535)
            self.image.setImage(model.values(now, delta), autoLevels=False, levels=levels)
            summary = f"色域 {levels[0]} 至 {levels[1]} {'ΔADC' if delta else 'ADC'}　灰色为无效单元"
        self.scale.setVisible(contact and not source_colors)
        self.source_scale.setVisible(source_colors)
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

        ref_info = self._update_monitor_status(valid, usable if contact else None, replay_pause, profile is not None)
        reason = message.reason if message is not None and not valid.any() else ""
        self.legend.setText("    ".join(filter(None, (model.sensor_id, summary, ref_info, reason, model.diagnostic,
                                                    model.rejection, model.service_error))))

    def _update_monitor_status(self, valid, contact_coverage, replay_pause, equalized):
        model = self.model
        message = model.message
        fresh = bool(valid.any())
        if message is None:
            state = "等待回放" if model.replay else "等待连接"
        elif not message.connected:
            state = "已断开"
        elif replay_pause:
            state = "回放暂停或等待下一帧"
        elif not fresh:
            state = "等待读数"
        else:
            state = "播放中" if model.replay else "已连接"
            coverage = contact_coverage if contact_coverage is not None and model.reference is not None else valid
            if not coverage.all():
                state += f" · 有效 {coverage.sum()}/{ROWS * COLS}"
        self.status.setText(f"{'回放' if model.replay else '实时'} · {state}")
        status = model.current_reference
        phase = status.phase if status else "unreferenced"
        ref_info = ""
        text = ""
        if phase == "ready" and model.reference is not None:
            covered = sum(model.reference.valid_mask)
            text = "已归零" if covered == ROWS * COLS else f"已归零 {covered}/400 点"
            if equalized:
                text += " · 已均衡"
            ref_info = f"参考 {covered}/400　满量程 {status.span_adc:g} ADC"
        elif phase == "collecting":
            text = f"正在归零 {status.progress:.0%}"
        elif phase == "failed":
            text = "归零未完成"
            ref_info = status.reason
        elif fresh:
            text = "待归零"
        self.ref_status.setText(text)
        can_collect = self.request_reference is not None and fresh and phase != "collecting"
        self.reference_button.setEnabled(can_collect)
        if model.service_error != self._last_reference_error:
            self._last_reference_error = model.service_error
            if model.service_error:
                self.statusBar().showMessage("未能设置空载参考，请在「详情」查看原因后重试。", 6000)

        prompt = ""
        if contact_coverage is not None and not contact_coverage.any():
            if message is None or not message.connected:
                prompt = "等待回放数据" if model.replay else "在「详情」中打开「串口设置」\n连接触觉传感器"
            elif not fresh:
                prompt = "等待下一帧回放数据" if model.replay else "等待传感器读数"
            elif phase == "collecting":
                prompt = "正在回放空载归零" if model.replay else "正在归零，请保持感应面空载"
            elif phase == "ready":
                prompt = "等待读取接触图"
            elif model.replay:
                prompt = "这段回放尚无可用的空载参考"
            elif self.request_reference is None:
                prompt = "等待空载参考"
            elif phase == "failed":
                prompt = "归零未完成\n请让感应面保持空载，再点「空载归零」"
            else:
                prompt = "让感应面保持空载\n然后点「空载归零」"
        if self.map_hint.text() != prompt:
            self.map_hint.setText(prompt)
        self.map_hint.setVisible(bool(prompt))
        return ref_info
