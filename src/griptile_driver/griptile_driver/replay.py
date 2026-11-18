"""Rebuild rolling state from recorded decoded samples without serial hardware."""

import math

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.event_handler import PublisherEventCallbacks
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from griptile_msgs.msg import SensorInfo, TactileRow44Array, TactileGridState

from griptile_core.assembler import Assembler
from griptile_core.identity import validate_sensor_id
from .lifecycle import shutdown_ros
from .ros_messages import batch_from_message, heatmap_message, state_message


class TactileReplayNode(Node):
    def __init__(self, **kwargs):
        super().__init__("griptile_replay", **kwargs)
        self.declare_parameters("", [("sensor_id", "sensor_0"), ("frame_id", ""),
                                     ("state_rate_hz", 30.0), ("max_cell_age_ms", 100.0),
                                     ("publish_heatmap", False)])
        self.sensor_id = validate_sensor_id(self.get_parameter("sensor_id").value)
        self.frame_id = self.get_parameter("frame_id").value
        rate = self.get_parameter("state_rate_hz").value
        max_age = self.get_parameter("max_cell_age_ms").value
        if not math.isfinite(rate) or rate <= 0 or not math.isfinite(max_age) or max_age <= 0:
            super().destroy_node()
            raise ValueError("state_rate_hz and max_cell_age_ms must be finite and positive")
        if self.resolve_topic_name("source_info") == self.resolve_topic_name("info"):
            super().destroy_node()
            raise ValueError("Recorded source_info and replay info require distinct topic names")
        self.max_age_ns = int(max_age * 1e6)
        self.assembler = Assembler(sensor_id=self.sensor_id)
        self._last_batch = None
        self._last_clock_ns = None
        self._needs_refresh = False
        self.rejected_samples = 0
        matched = PublisherEventCallbacks(matched=self._on_reader_matched)
        self.state_publisher = self.create_publisher(
            TactileGridState, "state", QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                                durability=DurabilityPolicy.VOLATILE),
            event_callbacks=matched)
        self.heatmap_publisher = (self.create_publisher(Image, "heatmap", 1, event_callbacks=matched)
                                  if self.get_parameter("publish_heatmap").value else None)
        info_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.info_publisher = self.create_publisher(SensorInfo, "info", info_qos)
        self.info_subscription = self.create_subscription(SensorInfo, "source_info", self.on_info, info_qos)
        self.sample_subscription = self.create_subscription(
            TactileRow44Array, "samples", self.on_samples,
            QoSProfile(depth=30, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.VOLATILE))
        # EOF can stop /clock between the final sample and the next ROS-time tick.
        # Steady time schedules presentation only. Freshness uses recorded clocks.
        self.timer = self.create_timer(1.0 / rate, self.publish_state,
                                       clock=Clock(clock_type=ClockType.STEADY_TIME))
        if not self.get_parameter("use_sim_time").value:
            self.get_logger().warning("Replay should use use_sim_time:=true with ros2 bag play --clock.")

    def _on_reader_matched(self, event):
        if event.current_count_change > 0:
            self._needs_refresh = True

    def on_info(self, message):
        # Keep the original metadata available after the bag player exits.
        if message.schema_version == 1 and message.sensor_id == self.sensor_id:
            self.info_publisher.publish(message)

    def on_samples(self, message):
        try:
            batch = batch_from_message(message, self.sensor_id)
        except ValueError:
            self.rejected_samples += 1
            self.assembler = Assembler(sensor_id=self.sensor_id)
            self._last_batch = None
            return
        now_ros_ns = self.get_clock().now().nanoseconds
        if self._last_clock_ns is not None and now_ros_ns < self._last_clock_ns:
            # A seek or loop starts a new timeline. A duplicate packet on a
            # forward timeline is handled by the assembler's sequence checks.
            self.assembler = Assembler(sensor_id=self.sensor_id)
        self._last_clock_ns = now_ros_ns
        self.assembler.apply(batch)
        self._last_batch = message
        self._needs_refresh = True

    def publish_state(self):
        if self._last_batch is None:
            return
        batch = self._last_batch
        now_ros_ns = self.get_clock().now().nanoseconds
        if self._last_clock_ns is not None and now_ros_ns < self._last_clock_ns:
            # Wait for data at the new playback position instead of restamping
            # data from the future after a clock rewind.
            self.assembler = Assembler(sensor_id=self.sensor_id)
            self._last_batch = None
            self._last_clock_ns = now_ros_ns
            return
        if not self._needs_refresh and now_ros_ns == self._last_clock_ns:
            return
        self._needs_refresh = False
        self._last_clock_ns = now_ros_ns
        now_recorded_monotonic_ns = batch.host_monotonic_ns + max(0, now_ros_ns - batch.host_realtime_ns)
        state = state_message(self.assembler.snapshot(now_recorded_monotonic_ns, self.max_age_ns),
                              self.frame_id or batch.header.frame_id)
        self.state_publisher.publish(state)
        if self.heatmap_publisher is not None:
            self.heatmap_publisher.publish(heatmap_message(state))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TactileReplayNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        shutdown_ros(node)
