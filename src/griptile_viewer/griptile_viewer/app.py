"""Single-threaded Qt/ROS subscriber for state, source info and diagnostics."""

from contextlib import ExitStack
import signal
import sys
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from griptile_msgs.msg import ContactReference, SensorInfo, TactileGridState
from griptile_msgs.srv import ManageSerial, SetContactReference
from sensor_msgs.msg import Image
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from .model import ArrayView
from .serial_client import SerialControl
from .window import QtCore, QtWidgets, ViewerWindow


class ViewerNode(Node):
    def __init__(self):
        super().__init__("griptile_viewer")
        sensor_id = self.declare_parameter("sensor_id", "sensor_0").value
        replay = self.get_parameter("use_sim_time").value
        self.model = ArrayView(sensor_id, replay=replay)
        self.view_clock = (lambda: self.get_clock().now().nanoseconds) if replay else time.monotonic_ns
        self.serial_control = None if replay else SerialControl(
            self.create_client(ManageSerial, "manage_serial"), ManageSerial.Request, self.model, self.view_clock)
        self.create_subscription(TactileGridState, "state", self._on_state,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        metadata_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                  durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(SensorInfo, "info", self.model.accept_info, metadata_qos)
        self.create_subscription(ContactReference, "contact_reference", self._on_reference, metadata_qos)
        self.create_subscription(Image, "contact_image", self._on_image, 1)
        self.create_subscription(Image, "contact_response", self._on_response, 1)
        self.reference_client = self.create_client(SetContactReference, "set_contact_reference")
        self.pending_reference = None
        self.create_subscription(DiagnosticArray, "diagnostics", self._on_diagnostics, 1)

    def _on_state(self, message):
        try:
            self.model.accept(message, self.view_clock())
        except (ValueError, TypeError, OverflowError) as error:
            self.model.invalidate(str(error))

    def _on_image(self, message):
        try:
            self.model.accept_image(message)
        except ValueError as error:
            self.model.invalidate(str(error))

    def _on_response(self, message):
        try:
            self.model.accept_response(message)
        except ValueError as error:
            self.model.invalidate(str(error))

    def _on_reference(self, message):
        try:
            self.model.accept_reference(message)
        except (TypeError, ValueError) as error:
            self.model.invalidate(str(error))

    def request_reference(self, profile=None):
        model = self.model
        state = model.message
        if state is None or not model.mask(self.view_clock()).any():
            raise ValueError("请等到有新鲜触觉数据后再设置参考")
        if not self.reference_client.service_is_ready():
            raise ValueError("等待驱动的无载参考服务")
        if self.pending_reference and not self.pending_reference.done():
            raise ValueError("参考请求正在处理中")
        request = SetContactReference.Request(sensor_id=model.sensor_id, session_id=state.session_id,
                                              stream_epoch=state.stream_epoch)
        if profile is not None:
            binding = model.binding()
            if not binding:
                raise ValueError("加载参考需要已核验的设备标识")
            request.load_profile = True
            request.hardware_id = binding["hardware_id"]
            request.baseline_adc = list(profile.baseline_adc)
            request.activation_threshold_adc = list(profile.activation_threshold_adc)
            request.valid_mask = list(profile.valid_mask)
        model.service_error = ""
        future = self.reference_client.call_async(request)
        self.pending_reference = future
        future.add_done_callback(self._reference_result)

    def _reference_result(self, future):
        try:
            result = future.result()
            self.model.service_error = "" if result.success else result.message
        except Exception as error:
            self.model.service_error = str(error)

    def _on_diagnostics(self, message):
        if message.status:
            values = {item.key: item.value for item in message.status[0].values}
            keys = (("producer_dropped_total", "丢批"), ("malformed_packets_total", "结构错误"))
            self.model.diagnostic = "  ".join(f"{title} {values[key]}" for key, title in keys if key in values)


def main(args=None):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([sys.argv[0]])
    stopping = False

    def request_stop(signum, frame):
        # Let Qt's next timer tick exit normally. Raising KeyboardInterrupt
        # during a Python slot can escape through the C++ event dispatcher.
        nonlocal stopping
        stopping = True

    with ExitStack() as cleanup:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous = signal.signal(sig, request_stop)
            cleanup.callback(signal.signal, sig, previous)
        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
        cleanup.callback(rclpy.try_shutdown)
        node = ViewerNode()
        cleanup.callback(node.destroy_node)
        executor = SingleThreadedExecutor()
        cleanup.callback(executor.shutdown)
        executor.add_node(node)
        window = ViewerWindow(node.model, node.view_clock, node.request_reference, serial_control=node.serial_control)
        window.show()
        ros_timer = QtCore.QTimer()

        def poll():
            if stopping or not rclpy.ok():
                app.quit()
                return
            try:
                executor.spin_once(timeout_sec=0)
            except ExternalShutdownException:
                app.quit()

        ros_timer.timeout.connect(poll)
        ros_timer.start(5)
        cleanup.callback(ros_timer.stop)
        draw_timer = QtCore.QTimer()
        draw_timer.timeout.connect(window.refresh)
        draw_timer.start(33)
        cleanup.callback(draw_timer.stop)
        return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
