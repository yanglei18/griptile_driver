#!/usr/bin/env python3
"""Run the normal ROS driver against a simulated 20x20 sensor on a private PTY."""
import argparse
import math
import os
import signal
import struct
import subprocess
import time


def scan(seconds=None):
    """Emit one complete scan. The unpressed surface has per-cell DC offsets."""
    phase = seconds % 10 if seconds is not None else 0
    ring = 1 <= phase < 4.5
    pads = 5.5 <= phase < 9
    onset = phase - (1 if ring else 5.5)
    strength = min(1.0, max(0.0, onset * 3)) * 3600
    packets = []
    for row in range(20):
        values = []
        for col in range(20):
            idle = 18000 + row * 7 + col * 11
            response = 0.0
            if ring:
                radius = math.hypot(row - 8, col - 11)
                if 3.5 <= radius <= 6.5:
                    response = 1 - abs(radius - 5) / 1.8
            elif pads:
                for y, x, gain in ((6, 5, 1.0), (13, 14, 0.58)):
                    distance = ((row - y) / 3.1)**2 + ((col - x) / 2.1)**2
                    if distance <= 1:
                        response = gain * (1 - 0.65 * distance)
            values.append(idle + round(response * strength))
        packets.append(struct.pack(">2sH20H", b"\xa5\x5a", row, *values))
    return b"".join(packets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Start the driver only")
    args = parser.parse_args()
    import rclpy
    from griptile_msgs.msg import ContactReference
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rclpy.signals import SignalHandlerOptions

    # Keep the simulated sensor unloaded until the actual driver has finished
    # its reference. This also works when DDS discovery or startup takes longer.
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    observer = rclpy.create_node("griptile_demo_source")
    ready_since = None

    def on_reference(message):
        nonlocal ready_since
        if message.phase == "ready" and any(message.valid_mask):
            if ready_since is None:
                ready_since = time.monotonic()
        else:
            ready_since = None

    observer.create_subscription(ContactReference, "/tactile/demo/contact_reference", on_reference,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    master, slave = os.openpty()
    path = os.ttyname(slave)
    os.close(slave)
    # The driver flushes and configures the PTY on open. Nonblocking writes keep
    # shutdown responsive when a subscriber or the driver is being stopped.
    os.set_blocking(master, False)
    cmd = ["ros2", "launch", "griptile_driver", "sensor.launch.py",
           "port:=" + path, "sensor_id:=demo", "contact_auto_reference:=true", "gui:=" + str(not args.headless).lower()]
    process = None
    try:
        process = subprocess.Popen(cmd, start_new_session=True)
        pending = b""
        while process.poll() is None:
            rclpy.spin_once(observer, timeout_sec=0)
            if not pending:
                elapsed = time.monotonic() - ready_since if ready_since is not None else None
                pending = scan(elapsed)
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
        observer.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
