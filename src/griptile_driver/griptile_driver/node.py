"""One serial owner, bounded worker queue, loss-aware ROS tactile output.

Only decoded sample batches are archival measurements. The timer-driven state
and optional RGB heatmap retain the acquisition timestamp of their source.
"""

from __future__ import annotations

from contextlib import ExitStack
from importlib.metadata import version
import math
from numbers import Integral
import signal
import time
import uuid

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rcl_interfaces.msg import SetParametersResult
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image
from griptile_msgs.msg import SensorInfo, TactileRow44Array, TactileGridState
from griptile_msgs.srv import ManageSerial

from griptile_core.assembler import Assembler
from griptile_core.grid import CELL_COUNT, COLS, LAYOUT_ID, ROWS, SCHEMA_VERSION
from griptile_core.identity import validate_sensor_id
from griptile_core.protocol import PROTOCOL_ID
from griptile_core.serial_ports import available_ports, validate_port
from griptile_core.serial_source import Batch, CLOCK_DOMAIN, MAX_READ_BYTES, SerialWorker, TIMING_SOURCE

from .contact import ContactImaging
from .ros_messages import heatmap_message, sample_message, state_message


_DATA_TIMEOUTS = ("no_data_timeout", "no_valid_packets")


class TactileNode(Node):
    def __init__(self, *, source_factory=SerialWorker, monotonic_ns=time.monotonic_ns,
                 context=None, parameter_overrides=None, **kwargs):
        super().__init__("griptile_driver", context=context,
                         parameter_overrides=parameter_overrides, **kwargs)
        defaults = {
            "port": "", "baudrate": 115200, "sensor_id": "sensor_0", "frame_id": "",
            "queue_capacity": 512, "sample_qos_depth": 128,
            "read_batch_bytes": 1024, "read_timeout_sec": 0.002,
            "state_rate_hz": 30.0, "diagnostics_rate_hz": 1.0,
            "stale_timeout_sec": 2.0, "max_cell_age_ms": 100.0, "reconnect_interval_sec": 1.0,
            "publish_heatmap": False,
        }
        self.declare_parameters("", list(defaults.items()))
        self.config = {name: self.get_parameter(name).value for name in defaults}
        try:
            if self.get_parameter("use_sim_time").value:
                raise ValueError("Live acquisition requires system time. Use griptile_replay for recorded samples.")
            for name in ("queue_capacity", "sample_qos_depth", "baudrate"):
                if self.config[name] < 1:
                    raise ValueError(f"{name} must be positive")
            read_batch_bytes = self.config["read_batch_bytes"]
            if (isinstance(read_batch_bytes, bool) or not isinstance(read_batch_bytes, Integral)
                    or not 1 <= read_batch_bytes <= MAX_READ_BYTES):
                raise ValueError(f"read_batch_bytes must be an integer in [1, {MAX_READ_BYTES}]")
            for name in ("state_rate_hz", "diagnostics_rate_hz", "stale_timeout_sec", "max_cell_age_ms", "reconnect_interval_sec", "read_timeout_sec"):
                if not math.isfinite(self.config[name]) or self.config[name] <= 0:
                    raise ValueError(f"{name} must be finite and positive")
            validate_sensor_id(self.config["sensor_id"])
        except Exception:
            super().destroy_node()
            raise
        self.add_on_set_parameters_callback(self._validate_parameter_change)
        self._monotonic_ns = monotonic_ns
        self.frame_id = self.config["frame_id"] or f"{self.config['sensor_id']}_tactile"
        self.assembler = Assembler(sensor_id=self.config["sensor_id"])
        self._last_batch = None
        self._last_snapshot = None
        self._closing = False
        self._last_diagnostic_summary = None
        self._rate_window_start_ns = self._monotonic_ns()
        self._rate_window_batches = 0
        self._rate_window_packets = 0
        self._batch_rate_hz = 0.0
        self._packet_rate_hz = 0.0
        self.sample_publisher = self.create_publisher(
            TactileRow44Array, "samples",
            QoSProfile(depth=self.config["sample_qos_depth"],
                       reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE),
        )
        self.state_publisher = self.create_publisher(
            TactileGridState, "state",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                       durability=DurabilityPolicy.VOLATILE),
        )
        self.diagnostic_publisher = self.create_publisher(DiagnosticArray, "diagnostics", 1)
        self.heatmap_publisher = self.create_publisher(Image, "heatmap", 1) if self.config["publish_heatmap"] else None
        self._source_factory = source_factory
        self._port = self.config["port"].strip()
        self._baudrate = self.config["baudrate"]
        self._serial_revision = str(uuid.uuid4())
        self._serial_ports = []
        self._serial_error = ""
        self.source = self._make_source(self._port, self._baudrate) if self._port else None
        self.info_publisher = self.create_publisher(
            SensorInfo, "info", QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                           reliability=ReliabilityPolicy.RELIABLE))
        self._driver_version = version("griptile_driver")
        self._last_info = None
        self.contact = ContactImaging(self, self.config["sensor_id"])
        self._publish_info()
        self.serial_service = self.create_service(ManageSerial, "manage_serial", self.manage_serial)
        self._pending_events = []
        self.drain_timer = self.create_timer(0.005, self.drain_samples)
        self.state_timer = self.create_timer(1.0 / self.config["state_rate_hz"], self.publish_state)
        self.diagnostic_timer = self.create_timer(1.0 / self.config["diagnostics_rate_hz"], self.publish_diagnostics)
        if self.source is not None:
            self.source.start()
        else:
            self._pending_events.append(self._boundary("port_not_configured"))

    def _make_source(self, port, baudrate):
        return self._source_factory(
            port=port, baudrate=baudrate, sensor_id=self.config["sensor_id"],
            queue_capacity=self.config["queue_capacity"], read_timeout=self.config["read_timeout_sec"],
            read_batch_bytes=self.config["read_batch_bytes"], no_data_timeout=self.config["stale_timeout_sec"],
            reconnect_interval_sec=self.config["reconnect_interval_sec"],
        )

    def _boundary(self, reason):
        # Administrative boundaries have their own session. SerialWorker's first
        # row/health event starts at sequence zero in a fresh acquisition session.
        return Batch(sensor_id=self.config["sensor_id"], session_id=str(uuid.uuid4()), reason=reason)

    def _serial_state(self):
        if self.source is None:
            return "idle"
        if not self.source.running:
            return "error"
        if self.source.connected:
            return "connected"
        return "error" if self._serial_error else "opening"

    def _stop_source(self):
        """Stop the writer before draining its finite backlog, including its final event."""
        if self.source is None:
            return True
        stopped = self.source.stop(timeout=1.0)
        if self.context.ok():
            self.drain_samples(limit=None)
        return stopped

    def _replace_serial(self, port, baudrate):
        if (port, baudrate) == (self._port, self._baudrate):
            if self._serial_state() != "error" and self._serial_error not in _DATA_TIMEOUTS:
                return
        next_source = self._make_source(port, baudrate) if port else None
        if not self._stop_source():
            raise ValueError("正在释放原串口，请稍后刷新并重试")

        self._port, self._baudrate = port, baudrate
        self.source = next_source
        self._serial_revision = str(uuid.uuid4())
        self._serial_error = ""
        self._pending_events = [self._boundary("opening_serial" if port else "port_not_configured")]
        self.drain_samples()
        self.publish_state()
        if next_source is not None:
            try:
                next_source.start()
            except RuntimeError as exc:
                # No worker acquired a descriptor; keep a retryable idle node.
                self.source, self._port = None, ""
                self._pending_events = [self._boundary("serial_start_failed")]
                self.drain_samples()
                self.publish_state()
                raise ValueError(f"无法启动串口采集：{exc}") from exc

    def manage_serial(self, request, result):
        try:
            if request.action == "refresh":
                self._serial_ports = available_ports()
            elif request.action not in ("connect", "disconnect"):
                raise ValueError("不支持的串口操作")
            elif request.expected_revision != self._serial_revision:
                raise ValueError("串口配置已变化，请刷新后重试")
            elif request.action == "connect":
                if not request.baudrate:
                    raise ValueError("波特率必须大于零")
                validate_port(request.port)
                self._replace_serial(request.port, request.baudrate)
            else:
                self._replace_serial("", self._baudrate)
            result.success = True
            result.message = self._serial_error
        except (OSError, ValueError, RuntimeError) as exc:
            result.success, result.message = False, str(exc)
        result.ports = self._serial_ports
        result.port, result.baudrate = self._port, self._baudrate
        result.state, result.revision = self._serial_state(), self._serial_revision
        result.session_id = self.source.session_id if self.source is not None else ""
        return result

    def _publish_info(self, batch=None):
        connected = batch is not None and batch.connected
        hardware_id = (self.source.identity_for_epoch(batch.stream_epoch)
                       if connected and self.source is not None else "")
        session_id = batch.session_id if batch else ""
        stream_epoch = batch.stream_epoch if batch else 0
        previous = self._last_info
        if (previous is not None and previous.session_id == session_id
                and previous.stream_epoch == stream_epoch and previous.connected == connected
                and previous.hardware_id == hardware_id):
            return
        info = SensorInfo(
            schema_version=1, sensor_id=self.config["sensor_id"], session_id=session_id,
            stream_epoch=stream_epoch, connected=connected, port=self._port,
            hardware_id=hardware_id, protocol_id=PROTOCOL_ID, layout_id=LAYOUT_ID,
            rows=ROWS, cols=COLS, unit="adc_counts", driver_version=self._driver_version,
            clock_domain=CLOCK_DOMAIN, timing_source=TIMING_SOURCE,
            time_description="Host clocks at serial read completion. Each row retains its acquisition time.",
            has_documented_checksum=False, has_hardware_sequence=False, has_hardware_timestamp=False,
        )
        info.header.stamp = self.get_clock().now().to_msg()
        info.header.frame_id = self.frame_id
        self.info_publisher.publish(info)
        self._last_info = info
        self.contact.update_info(info)

    def _validate_parameter_change(self, parameters):
        for parameter in parameters:
            if parameter.name == "use_sim_time" and parameter.value:
                return SetParametersResult(successful=False, reason="Live tactile acquisition requires system time.")
            if parameter.name in self.config and parameter.value != self.config[parameter.name]:
                return SetParametersResult(successful=False, reason="Capture parameters are startup-only; restart the node to change them.")
        return SetParametersResult(successful=True)

    def drain_samples(self, limit=128):
        # This queue is independent of the DDS KEEP_LAST history. A blocked DDS
        # publish never makes the serial thread wait on the ROS executor.
        batches = self._pending_events
        self._pending_events = []
        if self.source is not None:
            batches = batches + self.source.drain(limit=limit)
        for batch in batches:
            self._last_batch = batch
            if batch.reason.startswith("serial_error:") or batch.reason in (*_DATA_TIMEOUTS, "device_identity_changed"):
                self._serial_error = batch.reason
            elif batch.reason != "stopped":
                # A terminal "stopped" batch follows the underlying open/read error.
                self._serial_error = ""
            self._publish_info(batch)
            self.assembler.apply(batch)
            self.sample_publisher.publish(sample_message(batch, self.frame_id))
            self._rate_window_batches += 1
            self._rate_window_packets += len(batch.blocks)

    def publish_state(self):
        self._last_snapshot = self.assembler.snapshot(
            self._monotonic_ns(), int(self.config["max_cell_age_ms"] * 1e6))
        message = state_message(self._last_snapshot, self.frame_id)
        self.state_publisher.publish(message)
        self.contact.update(message)
        if self.heatmap_publisher is not None:
            self.heatmap_publisher.publish(heatmap_message(message))

    def publish_diagnostics(self):
        snapshot = self._last_snapshot
        batch = self._last_batch
        now_ns = self._monotonic_ns()
        elapsed_ns = now_ns - self._rate_window_start_ns
        if elapsed_ns > 0:
            self._batch_rate_hz = self._rate_window_batches * 1e9 / elapsed_ns
            self._packet_rate_hz = self._rate_window_packets * 1e9 / elapsed_ns
            self._rate_window_start_ns = now_ns
            self._rate_window_batches = self._rate_window_packets = 0
        valid_cell_count = sum(snapshot.valid_mask) if snapshot else 0
        source_age_ns = (max(0, now_ns - snapshot.host_monotonic_ns)
                         if snapshot and snapshot.host_monotonic_ns else -1)
        reason = snapshot.reason if snapshot else "waiting_for_state"
        if not snapshot or not snapshot.connected:
            level = DiagnosticStatus.ERROR
        elif not snapshot.time_valid or not any(snapshot.valid_mask):
            level = DiagnosticStatus.WARN
        else:
            level = DiagnosticStatus.OK
        dropped = batch.producer_dropped_total if batch else 0
        if dropped:
            level = max(level, DiagnosticStatus.WARN)
        summary = (level, reason, dropped)
        status = DiagnosticStatus(
            level=level, name=f"tactile/{self.config['sensor_id']}",
            hardware_id=self._last_info.hardware_id, message=reason or "ok",
            values=[KeyValue(key=key, value=str(value)) for key, value in {
                "port": self._port, "baudrate": self._baudrate, "unit": "adc_counts",
                "source_sequence": snapshot.source_sequence if snapshot else 0,
                "producer_dropped_total": dropped,
                "discarded_bytes_total": batch.discarded_bytes_total if batch else 0,
                "malformed_packets_total": batch.malformed_packets_total if batch else 0,
                "max_age_ns": snapshot.max_age_ns if snapshot else 0,
                "batch_rate_hz": f"{self._batch_rate_hz:.6f}",
                "packet_rate_hz": f"{self._packet_rate_hz:.6f}",
                "rate_scope": "host_publish; batches_include_status_events; not_hardware_scan_rate",
                "valid_cell_count": valid_cell_count,
                "cell_count": CELL_COUNT, "rows": ROWS, "cols": COLS,
                "state_schema_version": SCHEMA_VERSION,
                "source_age_ns": source_age_ns,
                "state_is_rolling": True,
                "checksum_verified": False,
            }.items()],
        )
        array = DiagnosticArray(status=[status])
        array.header.stamp = self.get_clock().now().to_msg()
        self.diagnostic_publisher.publish(array)
        if summary != self._last_diagnostic_summary:
            if level != DiagnosticStatus.OK:
                self.get_logger().warning(f"Tactile status: {status.message}; dropped batches={dropped}")
            self._last_diagnostic_summary = summary

    def destroy_node(self):
        if self._closing:
            return
        self._closing = True
        try:
            if not self._stop_source():
                self.get_logger().error("Tactile serial worker did not stop within 1 second.")
            if self.context.ok():
                # The recorder subscribes reliably to these three streams. All
                # acknowledgements share one deadline before destroying the writers.
                deadline = time.monotonic() + 1.0
                for name, publisher in (("samples", self.sample_publisher), ("info", self.info_publisher),
                                        ("contact_reference", self.contact.reference_publisher)):
                    remaining = max(0., deadline - time.monotonic())
                    if not publisher.wait_for_all_acked(Duration(seconds=remaining)):
                        self.get_logger().warning(f"Timed out waiting for {name} acknowledgements.")
        except RuntimeError:
            # A host application can shut down its context independently. Once
            # it has done so, stop the serial worker and release resources only.
            if self.context.ok():
                raise
        finally:
            super().destroy_node()


def main(args=None):
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    # Let the current callback finish and keep DDS valid through the final
    # publish. A second launch/terminal signal only repeats the request.
    with ExitStack() as cleanup:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous = signal.signal(number, request_stop)
            cleanup.callback(signal.signal, number, previous)
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
        cleanup.callback(rclpy.try_shutdown)
        node = TactileNode()
        cleanup.callback(node.destroy_node)
        executor = SingleThreadedExecutor(context=node.context)
        cleanup.callback(executor.shutdown)
        executor.add_node(node)
        try:
            while not stopping and node.context.ok():
                executor.spin_once(timeout_sec=.1)
        except ExternalShutdownException:
            pass
