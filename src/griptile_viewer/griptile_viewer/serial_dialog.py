"""Live driver serial settings. Device discovery runs on the driver host."""

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


_WAIT_REASONS = {
    "no_data_timeout": "没有收到新字节，请检查阵列供电和数据线。",
    "no_valid_packets": "已收到字节，未解析出有效阵列行。请核对串口设备和波特率。",
}


class SerialDialog(QtWidgets.QDialog):
    def __init__(self, viewer, control):
        super().__init__(viewer)
        self.setWindowTitle("GripTile · 串口设置")
        self.setMinimumWidth(440)
        self.setStyleSheet("QDialog { background: #182129; }")
        self.viewer = viewer
        self.control = control
        self._available = False
        self._configuration = None
        self._draft_port = False
        self._draft_baud = False
        self._local_error = ""
        self._requested = ""

        layout = QtWidgets.QVBoxLayout(self)
        self.status_label = QtWidgets.QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(self.status_label)
        form = QtWidgets.QFormLayout()
        self.port = QtWidgets.QComboBox()
        self.port.setEditable(True)
        self.port.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.port.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.port.setMinimumContentsLength(22)
        self.port.lineEdit().setPlaceholderText("选择或输入设备路径")
        self.port.setAccessibleName("串口设备")
        self.refresh_button = QtWidgets.QPushButton("刷新")
        devices = QtWidgets.QHBoxLayout()
        devices.addWidget(self.port, 1)
        devices.addWidget(self.refresh_button)
        form.addRow("串口设备", devices)
        self.baudrate = QtWidgets.QComboBox()
        self.baudrate.setAccessibleName("波特率")
        self.baudrate.setEditable(True)
        self.baudrate.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.baudrate.setValidator(QtGui.QIntValidator(1, 2**31 - 1, self.baudrate))
        self.baudrate.addItems(["9600", "19200", "38400", "57600", "115200", "230400",
                                "460800", "921600", "1000000"])
        self.baudrate.setCurrentText("115200")
        form.addRow("波特率", self.baudrate)
        layout.addLayout(form)
        self.feedback = QtWidgets.QLabel()
        self.feedback.setWordWrap(True)
        self.feedback.setTextFormat(QtCore.Qt.PlainText)
        layout.addWidget(self.feedback)
        actions = QtWidgets.QHBoxLayout()
        actions.addStretch()
        self.disconnect_button = QtWidgets.QPushButton("断开")
        self.connect_button = QtWidgets.QPushButton("连接")
        for button in (self.disconnect_button, self.connect_button):
            button.setAutoDefault(False)
            actions.addWidget(button)
        layout.addLayout(actions)
        self.port.currentTextChanged.connect(self._edit_port)
        self.baudrate.currentTextChanged.connect(self._edit_baud)
        self.refresh_button.clicked.connect(lambda: self._submit("refresh"))
        self.connect_button.clicked.connect(lambda: self._submit("connect"))
        self.disconnect_button.clicked.connect(lambda: self._submit("disconnect"))

    def _edit_port(self):
        self._draft_port = True
        self._local_error = ""
        self._render()

    def _edit_baud(self):
        self._draft_baud = True
        self._local_error = ""
        self._render()

    def _baud_value(self):
        value = self.baudrate.currentText().strip()
        return int(value) if value.isascii() and value.isdigit() and 0 < len(value) <= 10 else 0

    def _submit(self, action):
        if self.control is None or not self.control.available or self.control.pending:
            return
        self._local_error = ""
        try:
            self._requested = action
            if action == "connect":
                self.control.submit(action, self.port.currentText().strip(), self._baud_value())
            else:
                self.control.submit(action)
        except ValueError as error:
            self._local_error = str(error)
        self._render()

    def open_settings(self):
        self._available = False
        self.show()
        self.raise_()
        self.activateWindow()
        self.refresh_view()

    def refresh_view(self):
        if not self.isVisible():
            return
        available = self.control is not None and self.control.available
        if available and not self._available:
            self._available = True
            self._submit("refresh")
        else:
            self._available = available
            if self.control is not None:
                self.control.poll()
        self._render()

    def _sync_choices(self, status):
        configuration = (status.ports, status.port, status.baudrate)
        if configuration == self._configuration:
            return
        current = self.port.currentText()
        previous_port = self._configuration[1] if self._configuration else ""
        self._configuration = configuration
        if self._draft_port and current == status.port:
            self._draft_port = False
        if previous_port and current == previous_port and not status.port:
            self._draft_port = True
        if self._draft_baud and self._baud_value() == status.baudrate:
            self._draft_baud = False
        if not self._draft_port:
            current = status.port or (current if current in status.ports else "")
            if not current and len(status.ports) == 1:
                current = status.ports[0]
        with QtCore.QSignalBlocker(self.port), QtCore.QSignalBlocker(self.baudrate):
            self.port.clear()
            self.port.addItems(status.ports)
            self.port.setCurrentText(current)
            if not self._draft_baud:
                self.baudrate.setCurrentText(str(status.baudrate or 115200))

    def _render(self):
        control = self.control
        available = control is not None and control.available
        status = control.status if available else None
        busy = bool(available and control.pending)
        if status is not None:
            self._sync_choices(status)
        selected = self.port.currentText().strip()
        baud = self._baud_value()
        active = bool(status and status.port and status.state != "disconnected")
        same = bool(active and selected == status.port and baud == status.baudrate)
        wait_reason = _WAIT_REASONS.get(status.message, "") if status and status.state == "waiting" else ""
        retry = bool(same and (status.state == "error" or wait_reason))
        can_change = bool(status and available and not busy)
        self.refresh_button.setEnabled(available and not busy)
        self.connect_button.setEnabled(can_change and bool(selected) and 0 < baud <= 2**31 - 1
                                       and (not same or retry))
        self.disconnect_button.setEnabled(can_change and active)
        self.connect_button.setText("重试连接" if retry
                                    else "切换连接" if active and not same else "连接")

        if not available:
            headline = "等待实时驱动"
        elif busy and self._requested in ("connect", "disconnect"):
            headline = "正在请求连接" if self._requested == "connect" else "正在请求断开"
        elif status is None:
            headline = "正在读取串口配置" if busy else "等待读取串口配置"
        else:
            headline = {"disconnected": "未连接", "opening": "正在打开串口",
                        "waiting": "串口已打开，等待有效阵列数据", "streaming": "正在接收阵列数据",
                        "error": "串口连接异常"}.get(status.state, "等待串口状态")
            if active:
                headline += f"\n{status.port}　{status.baudrate} 波特"
        self.status_label.setText(headline)
        feedback = self._local_error or (control.error if control is not None else "")
        if not feedback and status is not None:
            if status.state == "error":
                feedback = status.message
            elif wait_reason:
                feedback = wait_reason
            elif not status.ports and not active:
                feedback = "未发现串口。接入阵列后点「刷新」，也可直接输入设备路径。"
            elif active and self.viewer.load_dialog and self.viewer.load_dialog.trial:
                if self.viewer.load_dialog.trial.stage != "done":
                    feedback = "切换或断开串口会结束当前载荷验证。"
        self.feedback.setText(feedback)
        self.feedback.setVisible(bool(feedback))
