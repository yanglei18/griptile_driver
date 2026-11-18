"""Passive inventory and validation of serial devices on the acquisition host."""

import os
from pathlib import Path
import re
import stat
import sys

from serial.tools import list_ports


def available_ports():
    """Return a single preferred path per device, without opening any ports."""
    candidates = {}
    for item in list_ports.comports():
        # Many Linux systems expose dozens of unused 8250 slots. Show USB
        # devices and known serial ports; a real built-in UART can be typed.
        if sys.platform.startswith("linux") and re.fullmatch(r"/dev/ttyS\d+", item.device) and not item.vid:
            continue
        candidates[os.path.realpath(item.device)] = item.device
    if sys.platform.startswith("linux"):
        for alias in sorted(Path("/dev/serial/by-id").glob("*"), reverse=True):
            if alias.is_char_device():
                candidates[os.path.realpath(alias)] = str(alias)
    return sorted(candidates.values(), key=lambda name: (not name.startswith("/dev/serial/by-id/"), name))


def validate_port(port):
    """Accept an explicit local serial device, including a Linux PTY slave."""
    if not port or port != port.strip() or "\x00" in port:
        raise ValueError("请输入有效的串口名称")
    if sys.platform == "win32":
        if not re.fullmatch(r"(?:\\\\\.\\)?COM[1-9][0-9]*", port, re.IGNORECASE):
            raise ValueError("请输入本机串口名称，如 COM3")
        return
    path = Path(port)
    if not path.is_absolute():
        raise ValueError("请输入串口的绝对路径")
    target = path.resolve(strict=True)
    if not target.is_relative_to("/dev") or not stat.S_ISCHR(target.stat().st_mode):
        raise ValueError("所选路径不是串口设备")
    if sys.platform.startswith("linux"):
        if target.parent == Path("/dev/pts") and target.name.isdecimal():
            return
        kernel_tty = Path("/sys/class/tty") / target.name
        if not (kernel_tty / "device").exists():
            raise ValueError("请选择串口设备或 PTY 从端")
        number = target.stat().st_rdev
        if (kernel_tty / "dev").read_text().strip() != f"{os.major(number)}:{os.minor(number)}":
            raise ValueError("设备号与系统串口不匹配")
