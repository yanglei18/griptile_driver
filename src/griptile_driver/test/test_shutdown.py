"""Exercise graceful stop against real ROS writers and a Linux pseudo-terminal."""

import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time

import pytest
import rclpy
from ament_index_python.packages import get_package_prefix
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile

from griptile_core.protocol import Block
from griptile_core.serial_source import SerialWorker
from griptile_driver.node import TactileNode
from griptile_msgs.msg import SensorInfo, TactileRow44Array


def test_host_context_may_close_first():
    context = Context()
    rclpy.init(context=context, args=[])
    node = TactileNode(context=context)
    context.shutdown()
    node.destroy_node()
    node.destroy_node()


@pytest.mark.parametrize("joined", (True, False))
def test_quiesce_publishes_the_whole_buffer_even_if_join_times_out(monkeypatch, joined):
    context = Context()
    rclpy.init(context=context, args=[])
    node = TactileNode(context=context)
    # Populate the production queue without opening a port. More than one
    # normal timer drain is waiting when the caller requests a stop.
    worker = SerialWorker("fixture", queue_capacity=160)
    node.source = worker
    rows = []
    monkeypatch.setattr(node.sample_publisher, "publish",
                        lambda batch: rows.extend(b.adc[0] for b in batch.blocks))
    monkeypatch.setattr(worker, "stop", lambda timeout: joined)
    try:
        for i in range(150):
            worker._emit(connected=True, reason="ok", blocks=(Block(i % 20, (i,) * 20),),
                         clocks=(i + 1, i + 1, True))
        assert node._stop_source() is joined
        assert rows == list(range(150)) and not worker.drain()
    finally:
        node.destroy_node()
        context.shutdown()


@pytest.mark.skipif(sys.platform != "linux", reason="requires a Linux pseudo-terminal")
@pytest.mark.parametrize("stop_signal", (signal.SIGINT, signal.SIGTERM))
def test_cli_shutdown_drains_confirmed_rows_before_terminal_event(stop_signal, tmp_path):
    context = Context()
    domain = 105 + os.getpid() % 35
    rclpy.init(context=context, domain_id=domain, args=[])
    namespace = f"/close_{os.getpid()}_{stop_signal.value}"
    observer = rclpy.create_node("archive", context=context, namespace=namespace)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(observer)
    infos, batches = [], []
    observer.create_subscription(SensorInfo, "info", infos.append, QoSProfile(
        depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    observer.create_subscription(TactileRow44Array, "samples", batches.append, 128)
    master, slave = os.openpty()
    port = os.ttyname(slave)
    os.close(slave)

    def spin_until(predicate, timeout=7):
        end = time.monotonic() + timeout
        while time.monotonic() < end and not predicate():
            executor.spin_once(timeout_sec=.02)
        assert predicate(), (tmp_path / "driver.log").read_text()

    executable = Path(get_package_prefix("griptile_driver")) / "lib/griptile_driver/griptile_node"
    # One long read holds all rows when the signal arrives. A single queue
    # slot tests that the final status cannot displace the decoded batch.
    args = [str(executable), "--ros-args", "-r", "__ns:=" + namespace,
            "-p", "port:=" + port, "-p", "queue_capacity:=1",
            "-p", "read_timeout_sec:=10.0", "-p", "read_batch_bytes:=4096"]
    env = dict(os.environ, ROS_DOMAIN_ID=str(domain), ROS_LOG_DIR=str(tmp_path / "ros"))
    with (tmp_path / "driver.log").open("w") as log:
        process = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        spin_until(lambda: any(m.connected for m in infos) and
                   observer.count_publishers(namespace + "/samples") == 1)
        wire = b"".join(struct.pack(">2sH20H", b"\xa5\x5a", i % 20, *range(100*i, 100*i + 20))
                        for i in range(41))
        assert os.write(master, wire) == len(wire)
        # Give the outstanding pyserial read time to accumulate these bytes.
        # It cannot complete until its 10 s timeout, cancellation, or 4096 bytes.
        time.sleep(.1)
        process.send_signal(stop_signal)
        spin_until(lambda: any(m.reason == "stopped" for m in batches), timeout=4)
        assert process.wait(timeout=2) == 0
        rows = [row for batch in batches for row in batch.blocks]
        # The 41st row provides lookahead for row 40. It has no lookahead itself.
        assert [(r.row, list(r.adc)) for r in rows] == [
            (i % 20, list(range(100*i, 100*i + 20))) for i in range(40)]
        assert batches[-1].reason == "stopped"
        assert batches[-1].discarded_bytes_total == 44
        assert batches[-1].producer_dropped_total == 0
        assert all(m.discarded_bytes_total == 0 for m in batches if m.blocks)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        os.close(master)
        executor.shutdown()
        observer.destroy_node()
        context.shutdown()
