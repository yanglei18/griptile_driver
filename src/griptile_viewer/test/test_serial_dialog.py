"""Use the real Qt controls against an asynchronous serial service stand-in."""

import pytest

from griptile_viewer.model import ArrayView
from griptile_viewer.serial_client import SerialStatus


class Control:
    def __init__(self):
        self.available = True
        self.pending = False
        self.error = ""
        self.status = None
        self.calls = []

    def submit(self, action, port="", baudrate=115200):
        self.calls.append((action, port, baudrate))
        self.error = ""
        self.pending = True

    def poll(self):
        pass

    def reply(self, state="disconnected", port="", baud=115200, ports=(), message=""):
        self.pending = False
        self.status = SerialStatus(ports, port, baud, state, message)


@pytest.fixture
def viewer(open_window):
    control = Control()
    return open_window(ArrayView("jaw"), serial_control=control), control


def test_port_controls_appear_on_compact_live_window_and_manage_connection(viewer):
    window, control = viewer
    assert window.serial_button.isVisible() and not window.details.isVisible()
    window.serial_button.click()
    dialog = window.serial_dialog
    assert control.calls == [("refresh", "", 115200)]
    control.reply(ports=("/dev/ttyACM0",))
    window.refresh()
    assert dialog.port.currentText() == "/dev/ttyACM0"
    assert not dialog.disconnect_button.isEnabled()
    dialog.baudrate.setCurrentText("230400")
    dialog.connect_button.click()
    assert control.calls[-1] == ("connect", "/dev/ttyACM0", 230400)
    assert not dialog.refresh_button.isEnabled() and not dialog.connect_button.isEnabled()
    for state, text in (("opening", "正在打开"), ("waiting", "等待有效"), ("streaming", "正在接收")):
        control.reply(state, "/dev/ttyACM0", 230400, ports=("/dev/ttyACM0",))
        window.refresh()
        assert text in dialog.status_label.text() and "/dev/ttyACM0" in dialog.status_label.text()
    assert not dialog.connect_button.isEnabled() and dialog.disconnect_button.isEnabled()
    dialog.disconnect_button.click()
    assert control.calls[-1][0] == "disconnect"
    control.reply()
    window.refresh()
    assert dialog.port.currentText() == "/dev/ttyACM0" and dialog.connect_button.isEnabled()
    assert "未连接" in dialog.status_label.text()


def test_refresh_preserves_manual_input_and_distinguishes_an_error_from_live_samples(viewer):
    window, control = viewer
    window.serial_button.click()
    dialog = window.serial_dialog
    control.reply()
    window.refresh()
    assert "未发现串口" in dialog.feedback.text()
    dialog.port.setEditText("/dev/serial/by-path/jaw")
    dialog.baudrate.setCurrentText("500000")
    dialog.refresh_button.click()
    control.reply(ports=("/dev/ttyUSB0", "/dev/ttyUSB1"))
    window.refresh()
    assert dialog.port.currentText() == "/dev/serial/by-path/jaw"
    assert dialog.baudrate.currentText() == "500000"
    dialog.connect_button.click()
    assert control.calls[-1] == ("connect", "/dev/serial/by-path/jaw", 500000)
    control.reply("error", "/dev/serial/by-path/jaw", 500000, message="设备忙")
    window.refresh()
    assert dialog.connect_button.text() == "重试连接" and dialog.connect_button.isEnabled()
    assert "设备忙" in dialog.feedback.text() and "正在接收" not in dialog.status_label.text()
    dialog.baudrate.setEditText("")
    assert not dialog.connect_button.isEnabled()


def test_wait_for_service_reopen_and_replay(viewer, open_window):
    window, control = viewer
    control.available = False
    window.serial_button.click()
    dialog = window.serial_dialog
    assert "等待实时驱动" == dialog.status_label.text() and not control.calls
    control.available = True
    window.refresh()
    assert len(control.calls) == 1
    control.reply()
    dialog.close()
    window.refresh()
    window.serial_button.click()
    assert window.serial_dialog is dialog and len(control.calls) == 2
    replay = open_window(ArrayView("jaw", replay=True), serial_control=control)
    assert not replay.serial_button.isVisible()
    replay.open_serial_dialog()
    assert replay.serial_dialog is None


@pytest.mark.parametrize("reason,hint", [
    ("no_data_timeout", "新字节"),
    ("no_valid_packets", "波特率"),
])
def test_open_port_reports_why_it_has_no_valid_samples_and_allows_retry(viewer, reason, hint):
    window, control = viewer
    window.serial_button.click()
    control.reply("waiting", "/dev/ttyACM0", message=reason)
    window.refresh()
    dialog = window.serial_dialog
    assert "串口已打开" in dialog.status_label.text() and hint in dialog.feedback.text()
    assert dialog.connect_button.text() == "重试连接" and dialog.connect_button.isEnabled()
    dialog.connect_button.click()
    assert control.calls[-1] == ("connect", "/dev/ttyACM0", 115200)
