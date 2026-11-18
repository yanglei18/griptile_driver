"""Rebuild rolling state from recorded decoded samples without serial hardware."""

from collections import deque
import math

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.event_handler import PublisherEventCallbacks
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from griptile_msgs.msg import ContactReference, SensorInfo, TactileRow44Array, TactileGridState

from griptile_core.assembler import Assembler
from griptile_core.identity import validate_sensor_id
from .contact import ContactImaging
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
        if self.resolve_topic_name("recorded_contact_reference") == self.resolve_topic_name("contact_reference"):
            super().destroy_node()
            raise ValueError("Recorded and reconstructed references require distinct topics")
        self._pending_references = deque(maxlen=128)
        self.max_age_ns = int(max_age * 1e6)
        self.assembler = Assembler(sensor_id=self.sensor_id)
        self._last_batch = None
        self._last_clock_ns = None
        self._needs_refresh = False
        self.rejected_samples = 0
        matched = PublisherEventCallbacks(matched=self._on_reader_matched)
        self.contact = ContactImaging(self, self.sensor_id, image_events=matched)
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
        self.reference_subscription = self.create_subscription(
            ContactReference, "recorded_contact_reference", self.on_reference, info_qos)
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

    def _check_rewind(self):
        now = self.get_clock().now().nanoseconds
        if self._last_clock_ns is not None and now < self._last_clock_ns:
            self._reset_timeline("playback_rewind")
            self._last_clock_ns = now
        return now

    def on_info(self, message):
        # Keep the original metadata available after the bag player exits.
        if message.schema_version == 1 and message.sensor_id == self.sensor_id:
            self._check_rewind()
            self.info_publisher.publish(message)
            self.contact.update_info(message)

    def on_reference(self, message):
        if message.sensor_id == self.sensor_id and message.session_id:
            self._check_rewind()
            self._pending_references.append(message)
            self._needs_refresh = True

    def _reference_at(self, state):
        # DDS may deliver metadata before the samples that established it.
        # Apply only the latest matching change at or before this source stamp.
        due = None
        future = deque(maxlen=self._pending_references.maxlen)
        for ref in self._pending_references:
            stamp = ref.header.stamp.sec * 1_000_000_000 + ref.header.stamp.nanosec
            if (ref.session_id, ref.stream_epoch) == (state.session_id, state.stream_epoch) and stamp <= state.host_realtime_ns:
                due = ref
            else:
                future.append(ref)
        self._pending_references = future
        return due

    def _reset_timeline(self, reason):
        self.contact.reset_timeline(reason)
        self._pending_references.clear()
        self.assembler = Assembler(sensor_id=self.sensor_id)
        self._last_batch = None

    def on_samples(self, message):
        try:
            batch = batch_from_message(message, self.sensor_id)
        except ValueError:
            self.rejected_samples += 1
            self._reset_timeline("invalid_sample")
            return
        self._last_clock_ns = self._check_rewind()
        self.assembler.apply(batch)
        self._last_batch = message
        self._needs_refresh = True

    def publish_state(self):
        now_ros_ns = self._check_rewind()
        if self._last_batch is None:
            return
        batch = self._last_batch
        if not self._needs_refresh and now_ros_ns == self._last_clock_ns:
            return
        self._needs_refresh = False
        self._last_clock_ns = now_ros_ns
        now_recorded_monotonic_ns = batch.host_monotonic_ns + max(0, now_ros_ns - batch.host_realtime_ns)
        state = state_message(self.assembler.snapshot(now_recorded_monotonic_ns, self.max_age_ns),
                              self.frame_id or batch.header.frame_id)
        self.state_publisher.publish(state)
        self.contact.update(state, self._reference_at(state))
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
