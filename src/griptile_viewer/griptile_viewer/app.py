"""Single-threaded Qt/ROS subscriber for state, source info and diagnostics."""

from contextlib import ExitStack
import signal
import sys
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from griptile_msgs.msg import SensorInfo, TactileGridState
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from .model import ArrayView
from .window import QtCore, QtWidgets, ViewerWindow


class ViewerNode(Node):
    def __init__(self):
        super().__init__("griptile_viewer")
        sensor_id = self.declare_parameter("sensor_id", "sensor_0").value
        replay = self.get_parameter("use_sim_time").value
        self.model = ArrayView(sensor_id, replay=replay)
        self.view_clock = (lambda: self.get_clock().now().nanoseconds) if replay else time.monotonic_ns
        self.create_subscription(TactileGridState, "state", self._on_state,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(SensorInfo, "info", self.model.accept_info,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                            durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(DiagnosticArray, "diagnostics", self._on_diagnostics, 1)

    def _on_state(self, message):
        try:
            self.model.accept(message, self.view_clock())
        except (ValueError, TypeError, OverflowError) as error:
            self.model.invalidate(str(error))

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
        window = ViewerWindow(node.model, node.view_clock)
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
