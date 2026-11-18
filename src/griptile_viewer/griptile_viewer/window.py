"""Qt view; axes are native zero-based matrix addresses."""

import os
import time

# Prefer the same Qt binding as the ROS desktop distribution.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from griptile_core.grid import ROWS, COLS
from .storage import load_reference, save_reference, save_snapshot


def _palette(colors):
    stops = np.linspace(0, 1, len(colors))
    return pg.ColorMap(stops, [pg.mkColor(color) for color in colors]).getLookupTable(nPts=256)


RAW = _palette(("#151629", "#383d87", "#157d98", "#45b88c", "#fee681"))
DELTA = _palette(("#2d5eaf", "#8ab9e6", "#f9f8ee", "#f2a272", "#a0373c"))


class ViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, model, clock_ns=time.monotonic_ns):
        super().__init__()
        self.model = model
        self.clock_ns = clock_ns
        self.last_receipt_wall_ns = time.monotonic_ns()
        self.last_message = None
        self.setWindowTitle(f"GripTile Viewer · {model.sensor_id}")
        self.setStyleSheet("""
            QMainWindow, QStatusBar { background: #182129; }
            QLabel, QStatusBar { color: #e0e9ed; font-size: 14px; }
            QLabel#status { font-size: 17px; font-weight: 600; padding: 4px 0; }
            QLabel#reference, QLabel#legend { color: #b9cad2; font-size: 13px; }
            QPushButton, QComboBox, QSpinBox {
                background: #25343f; color: #e5eef2; border: 1px solid #4a5b65;
                border-radius: 4px; padding: 5px 9px; min-height: 19px;
            }
            QPushButton:hover { background: #304752; border-color: #7bcbbd; }
            QPushButton:disabled, QSpinBox:disabled { color: #90a1ab; background: #202b34; }
            QComboBox QAbstractItemView { color: #e5eef2; background: #25343f; }
        """)
        self.resize(840, 740)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(18, 14, 18, 14)
        self.status = QtWidgets.QLabel()
        self.status.setObjectName("status")
        layout.addWidget(self.status)
        controls = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["原始 ADC", "相对无载参考 ΔADC"])
        controls.addWidget(self.mode)
        controls.addWidget(QtWidgets.QLabel("差值色限 ±"))
        self.span = QtWidgets.QSpinBox()
        self.span.setRange(1, 65535)
        self.span.setValue(3000)
        controls.addWidget(self.span)
        self.reference_button = QtWidgets.QPushButton("采集无载参考（3 秒）")
        self.reference_button.clicked.connect(self.start_reference)
        controls.addWidget(self.reference_button)
        controls.addStretch()
        layout.addLayout(controls)
        self.ref_status = QtWidgets.QLabel()
        self.ref_status.setObjectName("reference")
        layout.addWidget(self.ref_status)
        self.map = pg.PlotWidget(background="#333c46")
        self.map.setLabel("bottom", "列")
        self.map.setLabel("left", "行")
        self.map.setAspectLocked()
        self.map.hideButtons()
        self.map.getViewBox().setMenuEnabled(False)
        self.map.setMouseEnabled(False, False)
        self.map.invertY()
        self.map.setXRange(0, COLS, padding=0)
        self.map.setYRange(0, ROWS, padding=0)
        ticks = [(n + .5, str(n)) for n in (0, 4, 9, 14, 19)]
        for axis in ("left", "bottom"):
            self.map.getAxis(axis).setTicks([ticks])
        self.image = pg.ImageItem(axisOrder="row-major")
        self.map.addItem(self.image)
        self.marker = self.map.plot([], [], pen=None, symbol="s", symbolPen=pg.mkPen("w", width=2),
                                    symbolBrush=None, symbolSize=13)
        self.map.scene().sigMouseClicked.connect(self.pick_cell)
        self.map.scene().sigMouseMoved.connect(self.inspect_cell)
        layout.addWidget(self.map, stretch=4)
        self.cell_label = QtWidgets.QLabel("点击单元查看曲线")
        layout.addWidget(self.cell_label)
        self.curve = pg.PlotWidget(background="#171b21")
        self.curve.hideButtons()
        self.curve.setLabel("bottom", "相对最新样本", units="s")
        self.curve.setLabel("left", "原始 ADC")
        self.curve.showGrid(x=True, y=True, alpha=.2)
        self.line = self.curve.plot(pen=pg.mkPen("#72d8c5", width=2), connect="finite")
        layout.addWidget(self.curve, stretch=2)
        self.legend = QtWidgets.QLabel()
        self.legend.setObjectName("legend")
        self.legend.setWordWrap(True)
        layout.addWidget(self.legend)
        footer = QtWidgets.QHBoxLayout()
        self._file_button(footer, "保存参考", "JSON (*.json)", "save_ref")
        self._file_button(footer, "加载参考", "JSON (*.json)", "load_ref")
        self._file_button(footer, "导出 ADC 快照", "CSV (*.csv)", "csv")
        self._file_button(footer, "导出热图", "PNG (*.png)", "png")
        footer.addStretch()
        layout.addLayout(footer)
        self._format = None
        self.refresh()

    def _file_button(self, layout, caption, pattern, action):
        button = QtWidgets.QPushButton(caption)
        button.clicked.connect(lambda: self.file_action(caption, pattern, action))
        layout.addWidget(button)

    def file_action(self, caption, pattern, action):
        dialog = QtWidgets.QFileDialog
        chooser = dialog.getOpenFileName if action == "load_ref" else dialog.getSaveFileName
        path, _ = chooser(self, caption, "", pattern)
        if not path:
            return
        try:
            if action == "load_ref":
                load_reference(self.model, path)
                self.mode.setCurrentIndex(1)
            elif action == "save_ref":
                save_reference(self.model, path)
            elif action == "csv":
                save_snapshot(self.model, path, self.clock_ns())
            elif not self.map.grab().save(path, "PNG"):
                raise OSError("无法保存图像")
        except (ValueError, OSError, TypeError) as error:
            self.statusBar().showMessage(str(error), 6000)

    def start_reference(self):
        try:
            self.model.start_reference(self.clock_ns())
            self.mode.setCurrentIndex(1)
        except ValueError as error:
            self.statusBar().showMessage(str(error), 5000)

    def cell_at(self, position):
        data = self.image.mapFromScene(position)
        if 0 <= data.y() < ROWS and 0 <= data.x() < COLS:
            return int(data.y()), int(data.x())
        return None

    def pick_cell(self, event):
        cell = self.cell_at(event.scenePos())
        if cell is not None:
            self.model.select(*cell)

    def inspect_cell(self, position):
        cell = self.cell_at(position)
        if cell is not None:
            delta = self.mode.currentIndex() == 1
            value = self.model.values(self.clock_ns(), delta)[cell]
            label = f"{value:g} {'ΔADC' if delta else 'ADC'}" if np.isfinite(value) else "无有效值"
            self.statusBar().showMessage(f"行 {cell[0]}  列 {cell[1]}    {label}", 1500)

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
        if (model.response.status.phase == "collecting" and message
                and now - model.receive_clock > max(250_000_000, message.max_age_ns)):
            model.response.cancel_reference("数据中断")
        delta = self.mode.currentIndex() == 1
        levels = (-self.span.value(), self.span.value()) if delta else (0, 65535)
        current_format = (delta, levels)
        if current_format != self._format:
            self.image.setLookupTable(DELTA if delta else RAW)
            self._format = current_format
        self.image.setImage(model.values(now, delta), autoLevels=False, levels=levels)
        row, col = model.selected
        self.marker.setData([col + .5], [row + .5])
        raw = model.values(now)[row, col]
        value = model.values(now, True)[row, col]
        raw_text = f"{raw:g}" if np.isfinite(raw) else "—"
        delta_text = f"{value:+g}" if np.isfinite(value) else "—"
        self.cell_label.setText(f"行 {row}  列 {col}    ADC {raw_text}    ΔADC {delta_text}")
        if model.trend:
            stamps, values = zip(*model.trend)
            self.line.setData([(stamp - stamps[-1]) / 1e9 for stamp in stamps], values)
        else:
            self.line.setData([], [])

        if message is None:
            state = "等待阵列数据"
        elif not message.connected:
            state = f"已断开 · {message.reason}"
        elif replay_pause:
            state = "回放暂停或等待下一帧"
        elif not valid.any():
            state = f"无新鲜采样 · {message.reason}"
        else:
            state = "已连接"
        self.status.setText(f"{'回放' if model.replay else '实时'}   {model.sensor_id}    {state}"
                            f"    有效 {valid.sum()}/400")
        phase = model.response.status
        text = {"unreferenced": "无载参考未采集", "collecting": f"保持阵列无载 · {phase.progress:.0%}",
                "ready": f"无载参考就绪 · {phase.reference_cells}/400",
                "failed": f"参考采集未完成 · {phase.reason}"}.get(phase.phase, phase.phase)
        self.ref_status.setText(text)
        self.span.setEnabled(delta)
        self.reference_button.setEnabled(bool(valid.any()) and phase.phase != "collecting")
        self.legend.setText(f"色域 {levels[0]} 至 {levels[1]} {'ΔADC' if delta else 'ADC'}  ·  灰色为无效单元"
                            + (f"    {model.diagnostic}" if model.diagnostic else "")
                            + (f"    {model.rejection}" if model.rejection else ""))
