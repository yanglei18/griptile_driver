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

from sensor_msgs.msg import Image

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
        # A late image-only subscriber also triggers a final update after EOF.
        images = []
        viewer.create_subscription(Image, "contact_image", images.append, 1)
        spin_until(lambda: bool(images))
        assert images[-1].header == sample_message(final, "tip_surface").header
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


def test_contact_capture_real_rows_and_session_bound_import(context):
    from griptile_core.contact_image import INVALID_RGB, STRONG_RGB, UNLOADED_RGB
    from griptile_core.grid import LAYOUT_ID
    from griptile_core.protocol import PROTOCOL_ID
    from griptile_msgs.srv import SetContactReference

    current = [measurement().host_monotonic_ns]
    node = TactileNode(context=context, monotonic_ns=lambda: current[0],
                       parameter_overrides=[Parameter("sensor_id", value="pad_a")])
    node.contact.image_publisher = images = Collector()
    node.contact.reference_publisher = references = Collector()
    first = measurement()
    index = 0

    def send(adc, *, epoch=first.stream_epoch):
        nonlocal index
        age = index * 100_000_000
        current[0] = first.host_monotonic_ns + age
        batch = replace(first, sequence=first.sequence+index, first_packet_index=first.first_packet_index+index,
                        stream_epoch=epoch, host_monotonic_ns=current[0], host_realtime_ns=first.host_realtime_ns+age,
                        blocks=(Block(3, tuple(adc)),))
        index += 1
        node.assembler.apply(batch)
        node.publish_state()
        return images.messages[-1]

    def pixel(message, row, col):
        start = 3 * (row*20 + col)
        return tuple(message.data[start:start+3])

    def call(**kwargs):
        return node.contact.set_reference(SetContactReference.Request(**kwargs), SetContactReference.Response())

    try:
        zero = [20000] * 20
        initial = send(zero)
        assert (initial.height, initial.width, initial.step, initial.encoding) == (20, 20, 60, "rgb8")
        assert bytes(initial.data) == bytes(INVALID_RGB)*400
        assert references.messages[-1].phase == "unreferenced"
        assert call().success
        assert not call().success  # Concurrent clients cannot indefinitely restart a capture.
        for _ in range(30):
            last = send(zero)
        profile = references.messages[-1]
        assert profile.phase == "ready" and sum(profile.valid_mask) == 20
        assert pixel(last, 3, 4) == UNLOADED_RGB and pixel(last, 4, 3) == INVALID_RGB
        loaded = zero.copy()
        loaded[4] -= 5000
        loaded[18] += 5000
        image = send(loaded)
        assert image.header == node.contact.state.header
        assert pixel(image, 3, 4) == pixel(image, 3, 18) == STRONG_RGB
        assert bytes(send(zero).data) == bytes(last.data)
        args = dict(sensor_id=first.sensor_id, session_id=first.session_id, stream_epoch=first.stream_epoch,
                    baseline_adc=profile.baseline_adc, activation_threshold_adc=profile.activation_threshold_adc,
                    valid_mask=profile.valid_mask, load_profile=True)
        assert not call(hardware_id="", **args).success
        node.contact.update_info(SensorInfo(
            schema_version=1, sensor_id=first.sensor_id, session_id=first.session_id,
            stream_epoch=first.stream_epoch, hardware_id="usb:fixture", connected=True,
            rows=20, cols=20, unit="adc_counts", layout_id=LAYOUT_ID, protocol_id=PROTOCOL_ID))
        assert not call(hardware_id="wrong", **args).success
        assert call(hardware_id="usb:fixture", **args).success
        assert references.messages[-1].hardware_id == "usb:fixture"
        current[0] += 200_000_000
        node.publish_state()
        assert bytes(images.messages[-1].data) == bytes(INVALID_RGB)*400
        assert not call().success
        send(zero, epoch=2)
        assert references.messages[-1].phase == "unreferenced"
        assert not call(hardware_id="usb:fixture", **args).success
    finally:
        node.destroy_node()


def test_replay_keeps_reference_source_order_at_pause_and_rewind(context):
    from griptile_core.contact_image import INVALID_RGB, STRONG_RGB, UNLOADED_RGB
    from griptile_msgs.msg import ContactReference

    replay = TactileReplayNode(context=context, parameter_overrides=[
        Parameter("use_sim_time", value=True), Parameter("sensor_id", value="pad_a")])
    replay.contact.image_publisher = images = Collector()
    replay.contact.reference_publisher = profiles = Collector()
    try:
        first = measurement()
        second = replace(first, sequence=5, first_packet_index=11, host_realtime_ns=first.host_realtime_ns+10,
                         host_monotonic_ns=first.host_monotonic_ns+10)
        def received(batch):
            replay.get_clock().set_ros_time_override(Time(nanoseconds=batch.host_realtime_ns))
            replay.on_samples(sample_message(batch, "jaw"))
            replay.publish_state()

        def recorded(batch, **kwargs):
            fields = dict(header=sample_message(batch, "jaw").header, sensor_id=first.sensor_id,
                          session_id=first.session_id, stream_epoch=first.stream_epoch, span_adc=8.0,
                          direction="increase", phase="ready", progress=1.0, baseline_adc=[0.0]*400,
                          activation_threshold_adc=[3.0]*400, valid_mask=[True]*400)
            fields.update(kwargs)
            return ContactReference(**fields)

        replay.on_reference(recorded(second))
        received(first)
        assert bytes(images.messages[-1].data) == bytes(INVALID_RGB)*400
        received(second)
        image = images.messages[-1]
        assert image.header == sample_message(second, "jaw").header
        assert tuple(image.data[1140:1143]) == UNLOADED_RGB and tuple(image.data[-3:]) == STRONG_RGB
        assert profiles.messages[-1].span_adc == 8.0
        replay.on_reference(recorded(second, direction="decrease"))
        replay.publish_state()  # Late metadata after EOF updates pixels while time is stopped.
        assert tuple(images.messages[-1].data[-3:]) == UNLOADED_RGB
        # New playback position starts with this session's first measurement, no recorded reference yet.
        received(first)
        assert profiles.messages[-1].phase == "unreferenced"
        assert bytes(images.messages[-1].data) == bytes(INVALID_RGB)*400
    finally:
        replay.destroy_node()
