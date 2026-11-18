"""Logical sensor names and identity of an opened Linux USB serial device."""
import os
from pathlib import Path
import re
import sys
from urllib.parse import quote


def validate_sensor_id(sensor_id: str) -> str:
    """Allow a single ROS name token so sensor IDs can be used in namespaces."""
    if not isinstance(sensor_id, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", sensor_id):
        raise ValueError("sensor_id must be a nonempty ROS name token, e.g. sensor_0")
    return sensor_id


def usb_identity(device) -> str:
    """Read USB descriptors for the opened file descriptor, if available.

    Following the port path after opening can race udev during hotplug. The
    kernel device number provides the sysfs tty belonging to the descriptor.
    Missing serial numbers (including pseudo-terminals) yield an empty string.
    """
    if not sys.platform.startswith("linux"):
        return ""
    try:
        number = os.fstat(device.fileno()).st_rdev
        link = Path(f"/sys/dev/char/{os.major(number)}:{os.minor(number)}/device")
        tty = link.resolve(strict=True)
        for directory in (tty, *tty.parents):
            if (directory / "idVendor").is_file() and (directory / "idProduct").is_file():
                vendor, product, serial = (
                    (directory / name).read_text().strip()
                    for name in ("idVendor", "idProduct", "serial"))
                if (serial and re.fullmatch("[0-9A-Fa-f]{4}", vendor)
                        and re.fullmatch("[0-9A-Fa-f]{4}", product)):
                    return f"usb:{vendor.lower()}:{product.lower()}:{quote(serial, safe='')}"
                break
    except (AttributeError, OSError, UnicodeError, ValueError):
        pass
    return ""
