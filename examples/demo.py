#!/usr/bin/env python3
"""Run the normal ROS driver against a simulated 20x20 sensor on a private PTY."""
import argparse
import math
import os
import signal
import struct
import subprocess
import time


def scan(t):
    x = 9.5 + 5 * math.sin(t)
    y = 9.5 + 5 * math.cos(t)
    strength = max(0.0, math.sin(t / 2)) * 18000
    return b"".join(
        struct.pack(">2sH20H", b"\xa5\x5a", row, *(
            round(18000 + strength * math.exp(-((col-x)**2 + (row-y)**2) / 9))
            for col in range(20)
        )) for row in range(20)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Start the driver only")
    args = parser.parse_args()
    master, slave = os.openpty()
    path = os.ttyname(slave)
    os.close(slave)
    # The driver flushes and configures the PTY on open. Nonblocking writes keep
    # shutdown responsive when a subscriber or the driver is being stopped.
    os.set_blocking(master, False)
    cmd = ["ros2", "launch", "griptile_driver", "sensor.launch.py",
           "port:=" + path, "sensor_id:=demo", "gui:=" + str(not args.headless).lower()]
    process = None
    try:
        process = subprocess.Popen(cmd, start_new_session=True)
        start = time.monotonic()
        pending = b""
        while process.poll() is None:
            if not pending:
                pending = scan(time.monotonic() - start)
            try:
                pending = pending[os.write(master, pending):]
            except BlockingIOError:
                pass
            time.sleep(0.03)
        return process.returncode
    except KeyboardInterrupt:
        return 0
    finally:
        if process is not None and process.poll() is None:
            # The launch service forwards this signal to each child.
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
        os.close(master)


if __name__ == "__main__":
    raise SystemExit(main())
