import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from griptile_core import serial_ports


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux tty metadata and udev aliases")
def test_inventory_prefers_persistent_names_without_offering_unused_uart_slots(tmp_path, monkeypatch):
    aliases = tmp_path / "by-id"
    aliases.mkdir()
    master, slave = os.openpty()
    try:
        name = os.ttyname(slave)
        links = [aliases / "b-board", aliases / "a-board"]
        for alias in links:
            alias.symlink_to(name)
        (aliases / "departed").symlink_to("/dev/absent-test-serial")
        directory = Path.glob
        monkeypatch.setattr(Path, "glob", lambda self, pattern: directory(
            aliases if self == Path("/dev/serial/by-id") else self, pattern))
        monkeypatch.setattr(serial_ports.list_ports, "comports", lambda: [
            SimpleNamespace(device="/dev/ttyS30", vid=None), SimpleNamespace(device=name, vid=1234)])
        assert serial_ports.available_ports() == [str(links[1])]
        serial_ports.validate_port(name)
        serial_ports.validate_port(str(links[1]))
    finally:
        os.close(slave)
        os.close(master)


@pytest.mark.parametrize("name", ["", "relative", "/dev/null", "/dev/ptmx", "/etc/passwd", "/dev/missing-port"])
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux device path conventions")
def test_manual_selection_rejects_non_serial_devices(name):
    with pytest.raises((OSError, ValueError)):
        serial_ports.validate_port(name)
