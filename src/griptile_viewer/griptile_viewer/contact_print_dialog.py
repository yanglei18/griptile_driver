"""A contact print and its three time-registered single-exposure proofs."""

from concurrent.futures import ThreadPoolExecutor
from time import monotonic

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from griptile_core.contact_print import render_contact_print
from griptile_core.grid import COLS, ROWS


GROUND = QtGui.QColor("#0d1013")
PAPER = QtGui.QColor("#ede8d9")
INK = QtGui.QColor("#232a2c")
SUBTLE = QtGui.QColor("#62675f")
FULL_ARRAY = QtCore.QRect(0, 0, COLS, ROWS)
_COLOR_KEY = "暖铜　响应增强　 ·　 青色　响应回落　 ·　 灰白　三次相近"


def _contact_bounds(frames):
    # Include the union of all three lowest visible contour footprints plus
    # a margin. A weak-only exposure keeps the whole sensor in view.
    rows, cols = np.nonzero(np.any(np.asarray(frames) >= 500, axis=0))
    if not len(rows):
        return FULL_ARRAY
    area = QtCore.QRect(QtCore.QPoint(int(cols.min()), int(rows.min())),
                        QtCore.QPoint(int(cols.max()), int(rows.max())))
    return area.adjusted(-2, -2, 2, 2).intersected(FULL_ARRAY)


def _qimage(pixels):
    data = np.ascontiguousarray(pixels)
    return QtGui.QImage(data.data, data.shape[1], data.shape[0],
                        data.strides[0], QtGui.QImage.Format_RGB888).copy()


def _font(size, *, serif=False):
    font = QtGui.QFont("Noto Serif CJK SC" if serif else "Noto Sans CJK SC")
    font.setPixelSize(size)
    return font


def _text(painter, rect, text, color, size, *, serif=False, right=False):
    painter.setPen(color)
    painter.setFont(_font(size, serif=serif))
    alignment = QtCore.Qt.AlignVCenter | (QtCore.Qt.AlignRight if right else QtCore.Qt.AlignLeft)
    painter.drawText(rect, alignment, text)


def _draw_exposure(painter, rect, image, bounds):
    sx, sy = image.width() / COLS, image.height() / ROWS
    source = QtCore.QRectF(bounds.x() * sx, bounds.y() * sy,
                           bounds.width() * sx, bounds.height() * sy)
    size = source.size().scaled(rect.size(), QtCore.Qt.KeepAspectRatio)
    target = QtCore.QRectF(QtCore.QPointF(), size)
    target.moveCenter(rect.center())
    painter.drawImage(target, image, source)


class _PrintPlate(QtWidgets.QWidget):
    """Paint the same registered composition to a window or an export device."""

    def __init__(self, parent):
        super().__init__(parent)
        self.record = None
        self.images = ()
        self.bounds = FULL_ARRAY
        self.proof_labels = ()
        self.incomplete = False
        self.historical = False
        self.empty_text = "等待有效读数"
        self.zoomed = True
        self.setMinimumSize(300, 320)
        self.setAccessibleName("触印与三张单帧响应图")

    def clear(self, message):
        if self.record is None and self.empty_text == message and self.accessibleDescription() == message:
            return
        self.record = None
        self.images = self.proof_labels = ()
        self.bounds = FULL_ARRAY
        self.incomplete = self.historical = False
        self.empty_text = message
        self.setAccessibleDescription(message)
        self.update()

    def assign(self, record, images, *, historical):
        self.record = record
        self.images = tuple(_qimage(image) for image in images)
        self.bounds = _contact_bounds(record.frames)
        self.historical = historical
        last = record.stamps_ns[-1]
        self.proof_labels = tuple(f"{round((last - t) / 1_000_000)} 毫秒前"
                                  for t in record.stamps_ns[:-1]) + (("取印时" if historical else "此刻"),)
        self.incomplete = not np.isfinite(record.frames).all()
        self.setAccessibleDescription(self.caption() + "。" + "、".join(self.proof_labels)
                                      + "。单帧越亮，响应越强。" + _COLOR_KEY)
        self.update()

    def caption(self):
        kind = "实时接触"
        if self.historical:
            kind = "历史留印 · " + ("自动留印" if self.record.automatic else "手动留印")
        return kind + (" · 已均衡" if self.record.equalized else "")

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        try:
            self.draw(painter, self.width(), self.height())
        finally:
            painter.end()

    def draw(self, painter, width, height):
        rect = QtCore.QRectF
        margin = min(28, max(12, width // 24))
        title = min(48, max(30, width // 14))
        proof_height = min(188, max(116, height // 3))
        hero = height - proof_height
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, not self.incomplete)
        painter.fillRect(rect(0, 0, width, height), GROUND)
        _text(painter, rect(margin, 7, width / 2, title + 12), "触印", PAPER, title, serif=True)
        _text(painter, rect(width / 2, 9, width / 2 - margin, 30),
              "GripTile", PAPER, min(24, title - 8), serif=True, right=True)
        _text(painter, rect(width / 2, 37, width / 2 - margin, 18),
              "20 × 20", QtGui.QColor("#b6b9af"), 12, right=True)
        if self.record is None:
            painter.setFont(_font(16))
            painter.setPen(QtGui.QColor("#abafa8"))
            painter.drawText(rect(0, 55, width, height - 85), QtCore.Qt.AlignCenter, self.empty_text)
            return
        painter.fillRect(rect(0, hero, width, proof_height), PAPER)
        tile_top = hero + 29
        tile_h = max(30, proof_height - 78)
        gap = max(6, margin // 2)
        tile_w = (width - 2 * margin - 2 * gap) / 3
        proof_note = "同一位置 · 越亮，响应越强"
        if self.incomplete:
            proof_note += "　中灰方块为缺测"
        _text(painter, rect(margin, hero + 4, width - 2 * margin, 22), proof_note, INK, 13)
        bounds = self.bounds if self.zoomed else FULL_ARRAY
        if bounds != FULL_ARRAY:
            # Locator shows the selected region in the full physical array.
            locator = rect(width - margin - 130, 17, 33, 33)
            painter.setPen(QtGui.QColor("#586264"))
            painter.drawRect(locator)
            painter.setPen(QtGui.QColor("#c29170"))
            painter.drawRect(rect(locator.x() + bounds.x() / COLS * locator.width(),
                                  locator.y() + bounds.y() / ROWS * locator.height(),
                                  bounds.width() / COLS * locator.width(),
                                  bounds.height() / ROWS * locator.height()))
        view_top = max(57, title + 19)
        view = rect(margin, view_top, width - 2 * margin, hero - view_top - 34)
        _draw_exposure(painter, view, self.images[0], bounds)
        _text(painter, rect(margin, hero - 29, width - 2 * margin, 21), self.caption(), PAPER, 12)
        address = (f"行 {bounds.top()}–{bounds.bottom()}　列 {bounds.left()}–{bounds.right()}"
                   if bounds != FULL_ARRAY else "全阵列")
        _text(painter, rect(width / 2, hero - 29, width / 2 - margin, 21),
              address, QtGui.QColor("#b6b9af"), 12, right=True)
        for i in range(3):
            x = margin + i * (tile_w + gap)
            tile = rect(x, tile_top, tile_w, tile_h)
            painter.fillRect(tile, GROUND)
            _draw_exposure(painter, tile, self.images[i + 1], bounds)
            _text(painter, rect(x, tile_top + tile_h + 1, tile_w, 18),
                  self.proof_labels[i], INK, 12)
        legend = (("#b87649", "暖铜", "增强"), ("#367f92", "青色", "回落"),
                  ("#858981", "灰白", "三次相近"))
        for i, (color, tone, meaning) in enumerate(legend):
            x = margin + i * (tile_w + gap)
            painter.fillRect(rect(x, height - 15, 7, 7), QtGui.QColor(color))
            label = tone + "  " + meaning
            _text(painter, rect(x + 12, height - 23, tile_w - 12, 22), label, SUBTLE,
                  11 if width < 420 else 13)

    def export_image(self):
        image = QtGui.QImage(1280, 1200, QtGui.QImage.Format_RGB32)
        painter = QtGui.QPainter(image)
        try:
            painter.scale(2, 2)
            self.draw(painter, 640, 600)
        finally:
            painter.end()
        return image


class ContactPrintDialog(QtWidgets.QDialog):
    # Keep acquisition and normal Qt painting responsive while showing the latest contact.
    PREVIEW_INTERVAL = 0.1

    def __init__(self, parent, capture):
        super().__init__(parent)
        self.capture = capture
        self._executor = None
        self._work = None
        self._view = None
        self._next_preview = 0.
        self._error = ""
        self.setWindowTitle("GripTile · 触印")
        self.setWindowModality(QtCore.Qt.NonModal)
        self.resize(620, 660)
        self.setStyleSheet("""
            QDialog { background: #ede8d9; }
            QDialog QLabel, QDialog QCheckBox { color: #353e3b; font-size: 13px; }
            QDialog QPushButton { background: #efece4; color: #22292a; border: 1px solid #969b90;
                border-radius: 3px; min-height: 23px; padding: 3px 11px; }
            QPushButton#shutter { background: #26383a; color: #f5f1e5; }
            QDialog QPushButton:disabled, QPushButton#shutter:disabled { color: #777c78; background: #deded4; }
        """)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.setSpacing(6)
        self.plate = _PrintPlate(self)
        layout.addWidget(self.plate, 1)
        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(12, 4, 12, 0)
        self.shutter_button = QtWidgets.QPushButton("拍下当前", self)
        self.shutter_button.setObjectName("shutter")
        self.export_button = QtWidgets.QPushButton("导出触印", self)
        for button in (self.shutter_button, self.export_button):
            button.setAutoDefault(False)
            controls.addWidget(button)
        controls.addStretch()
        self.zoom = QtWidgets.QCheckBox("放大接触区", self)
        self.zoom.setChecked(True)
        self.zoom.setToolTip("三个时刻共用同一取景。取消可查看完整阵列。")
        controls.addWidget(self.zoom)
        layout.addLayout(controls)
        self.status = QtWidgets.QLabel(self)
        self.status.setWordWrap(True)
        info = QtWidgets.QHBoxLayout()
        info.setContentsMargins(12, 0, 12, 0)
        self.history_button = QtWidgets.QPushButton("上次留印", self)
        self.history_button.setCheckable(True)
        self.history_button.setAutoDefault(False)
        info.addWidget(self.history_button)
        info.addWidget(self.status, 1)
        layout.addLayout(info)
        self.shutter_button.clicked.connect(self._snapshot)
        self.export_button.clicked.connect(self._export)
        self.history_button.clicked.connect(self.refresh)
        self.zoom.toggled.connect(self._set_zoom)

    def _set_zoom(self, enabled):
        self.plate.zoomed = enabled
        self.plate.update()

    def showEvent(self, event):
        super().showEvent(event)
        self.history_button.setChecked(False)
        self.refresh()

    def closeEvent(self, event):
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._work = None
        super().closeEvent(event)

    def _update_plate(self, view, ready, empty):
        """Render one frame at a time. A result only belongs to its source contact or saved print."""
        historical, source = view
        if view != self._view:
            self._view, self._error, self._next_preview = view, "", 0.
            self.plate.clear(empty)
        elif self.plate.record is None or (not historical and not ready):
            self.plate.clear(empty)
        if self._work is not None and self._work[2].done():
            work_view, record, future = self._work
            self._work = None
            if work_view == view and (historical or ready):
                try:
                    master, proofs = future.result()
                    self.plate.assign(record, (master, *proofs), historical=historical)
                except (ValueError, RuntimeError) as error:
                    self._error = str(error)
        target = None
        if self._work is None and not self._error:
            if historical:
                target = source
            elif ready and monotonic() >= self._next_preview:
                target = self.capture.preview()
                self._next_preview = monotonic() + self.PREVIEW_INTERVAL
        if target is not None and self.plate.record is not target:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="griptile-print")
            self._work = view, target, self._executor.submit(render_contact_print, target.frames)

    def refresh(self):
        if not self.isVisible():
            return
        capture = self.capture
        if capture.latest is None:
            self.history_button.setChecked(False)
        historical = self.history_button.isChecked()
        present = capture.contact_present
        ready = capture.can_snapshot
        if historical:
            empty = "正在载入历史留印"
        elif present is None:
            empty = "等待有效读数"
        elif not present:
            empty = "无接触"
        else:
            empty = "正在生成当前触印" if ready else "已检测到接触 · 等待连续读数"
        self._update_plate((historical, capture.latest if historical else capture.contact_id), ready, empty)
        self.history_button.setEnabled(capture.latest is not None)
        self.history_button.setText("返回实时" if historical else "上次留印")
        self.shutter_button.setEnabled(ready and not historical)
        self.export_button.setEnabled(historical and self.plate.record is not None)
        self.zoom.setEnabled(self.plate.record is not None and self.plate.bounds != FULL_ARRAY)
        if self._error:
            text = "无法生成触印　" + self._error
        elif historical:
            text = "当前查看已保存的历史留印。"
        elif capture.latest is not None:
            text = "已保存一张触印，可查看或导出。"
        elif present is None:
            text = "等待当前阵列和有效的无载参考。"
        elif not present:
            text = "轻压感应面即可查看接触。"
        else:
            text = "拍下当前，或在接触响应回落后查看自动留印。"
        self.status.setText(text)

    def _snapshot(self):
        self.capture.snapshot()
        self.refresh()

    def _export(self):
        # A native save dialog starts a nested event loop; an incoming release
        # during that loop must not substitute a different print.
        if not self.plate.historical:
            return
        image = self.plate.export_image()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出触印", "griptile-print.png",
                                                        "PNG 图像 (*.png)")
        if path and not image.save(path, "PNG"):
            QtWidgets.QMessageBox.warning(self, "导出触印", "无法保存该图像。")

    def save_png(self, path):
        if not self.plate.historical:
            raise ValueError("请先打开上次留印")
        if not self.plate.export_image().save(str(path), "PNG"):
            raise OSError("无法保存触印")
