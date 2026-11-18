"""Exercise generated messages, the ROS nodes and their clock/identity boundaries."""

from dataclasses import replace
import time

import pytest
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time

from griptile_msgs.msg import SensorInfo, TactileGridState
from griptile_core.protocol import Block
from griptile_core.serial_source import Batch
from griptile_driver.node import TactileNode
from griptile_driver.replay import TactileReplayNode
from griptile_driver.ros_messages import batch_from_message, sample_message


@pytest.fixture
def context():
    ros = Context()
    rclpy.init(context=ros, domain_id=193, args=[])
    try:
        yield ros
    finally:
        ros.shutdown()


class Collector:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def measurement():
    return Batch(
        sensor_id="pad_a", session_id="recorded", stream_epoch=1, sequence=4,
        first_packet_index=10, host_realtime_ns=10_000_000_000,
        host_monotonic_ns=6_000_000_000, connected=True, time_valid=True,
        reason="ok", blocks=(Block(19, tuple(range(20))),),
    )


def test_latched_info_reflects_actual_connection(context):
    node = TactileNode(context=context, namespace="/tactile/pad_a",
                       parameter_overrides=[Parameter("sensor_id", value="pad_a")])
    try:
        assert node.source is None
        assert node.info_publisher.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL
        assert node.sample_publisher.qos_profile.reliability == ReliabilityPolicy.RELIABLE
        assert node.state_publisher.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT

        class Source:
            def identity_for_epoch(self, epoch):
                return "usb:fixture" if epoch == 1 else ""

            def stop(self, timeout):
                return True

        node.source = Source()
        collector = Collector()
        node.info_publisher = collector
        first = measurement()
        node._publish_info(first)
        node._publish_info(replace(first, sequence=5))
        node._publish_info(replace(first, sequence=6, connected=False))
        node._publish_info(replace(first, sequence=7, stream_epoch=2))
        assert [(m.session_id, m.stream_epoch, m.connected, m.hardware_id) for m in collector.messages] == [
            ("recorded", 1, True, "usb:fixture"),
            ("recorded", 1, False, ""),
            ("recorded", 2, True, ""),
        ]
    finally:
        node.destroy_node()


def test_recorded_source_time_controls_replay_expiry_and_seek(context):
    node = TactileReplayNode(context=context, parameter_overrides=[
        Parameter("use_sim_time", value=True), Parameter("sensor_id", value="pad_a")])
    try:
        collector = Collector()
        node.state_publisher = collector
        first = measurement()
        message = sample_message(first, "tip_surface")
        assert batch_from_message(message, "pad_a") == first
        clock = node.get_clock()
        clock.set_ros_time_override(Time(nanoseconds=10_001_000_000))
        node.on_samples(message)
        node.publish_state()
        state = collector.messages[-1]
        assert state.header == message.header and state.sensor_id == "pad_a"
        assert state.valid_mask[380] and state.adc[380] == 0
        assert state.adc[399] == 19 and sum(state.valid_mask) == 20
        assert state.row_monotonic_ns[19] == first.host_monotonic_ns

        clock.set_ros_time_override(Time(nanoseconds=10_200_000_000))
        node.publish_state()
        assert not any(collector.messages[-1].valid_mask)
        node.on_samples(message)
        node.publish_state()
        assert not any(collector.messages[-1].valid_mask)
        clock.set_ros_time_override(Time(nanoseconds=10_001_000_000))
        node.on_samples(message)
        node.publish_state()
        assert sum(collector.messages[-1].valid_mask) == 20
        message.header.stamp.nanosec = 1
        node.on_samples(message)
        assert node.rejected_samples == 1
    finally:
        node.destroy_node()


def test_short_bag_finishes_between_ticks_and_late_viewer_receives_state_and_info(context):
    replay = TactileReplayNode(context=context, namespace="/short_bag", parameter_overrides=[
        Parameter("use_sim_time", value=True), Parameter("sensor_id", value="pad_a")])
    viewer = Node("late_viewer", context=context, namespace="/short_bag")
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(replay)
    executor.add_node(viewer)
    try:
        # A player delivers its only rows and exits. No further /clock tick will occur.
        first = measurement()
        final = replace(first, sequence=5, first_packet_index=11,
                        host_monotonic_ns=first.host_monotonic_ns + 1_000_000,
                        host_realtime_ns=first.host_realtime_ns + 1_000_000,
                        blocks=(Block(18, tuple(range(20, 40))),))
        replay.get_clock().set_ros_time_override(Time(nanoseconds=final.host_realtime_ns))
        replay.on_samples(sample_message(first, "tip_surface"))
        replay.on_samples(sample_message(final, "tip_surface"))
        info = SensorInfo(schema_version=1, sensor_id="pad_a", hardware_id="usb:fixture",
                          session_id=first.session_id, stream_epoch=first.stream_epoch)
        replay.on_info(info)

        # Wait for the first presentation to finish *before* opening the viewer.
        def spin_until(condition):
            deadline = time.monotonic() + 3
            while not condition() and time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.05)
            assert condition()

        spin_until(lambda: not replay._needs_refresh)
        states, metadata = [], []
        viewer.create_subscription(TactileGridState, "state", states.append,
                                   QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        viewer.create_subscription(SensorInfo, "info", metadata.append, replay.info_publisher.qos_profile)
        spin_until(lambda: bool(states and metadata))
        assert metadata == [info]
        state = states[-1]
        assert state.header == sample_message(final, "tip_surface").header
        assert state.source_sequence == final.sequence
        assert state.row_monotonic_ns[18] == final.host_monotonic_ns
        assert sum(state.valid_mask) == 40
        assert list(state.adc[360:380]) == list(range(20, 40))

        # An unchanged paused clock emits no repeated snapshots.
        collector = Collector()
        replay.state_publisher = collector
        replay.publish_state()
        replay.publish_state()
        assert not collector.messages
    finally:
        executor.shutdown()
        viewer.destroy_node()
        replay.destroy_node()
