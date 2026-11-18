"""Nonblocking serial control for the viewer. Only the driver opens the port."""

from dataclasses import dataclass, replace
import time


@dataclass(frozen=True)
class SerialStatus:
    ports: tuple[str, ...]
    port: str
    baudrate: int
    state: str
    message: str = ""


class SerialControl:
    def __init__(self, client, make_request, model, view_clock, monotonic=time.monotonic):
        self._client = client
        self._make_request = make_request
        self._model = model
        self._view_clock = view_clock
        self._clock = monotonic
        self._snapshot = None
        self._session = self._revision = ""
        self._future = None
        self._action = ""
        self._sent_at = float("-inf")
        self.error = ""
        self._transport_failed = False

    @property
    def available(self):
        return self._client.service_is_ready()

    @property
    def pending(self):
        return self._future is not None and (self._action != "refresh" or self._snapshot is None)

    @property
    def status(self):
        status = self._snapshot
        if status is None or status.state != "waiting":
            return status
        message = self._model.message
        if (self._session and message is not None and message.session_id == self._session
                and self._model.mask(self._view_clock()).any()):
            return replace(status, state="streaming")
        return status

    def _cancel(self):
        future, self._future = self._future, None
        if future is not None:
            future.cancel()

    def _forget_source(self):
        self._snapshot = None
        self._session = self._revision = ""

    def submit(self, action, port="", baudrate=115200):
        if action not in ("refresh", "connect", "disconnect"):
            raise ValueError("未知的串口操作")
        if not self.available:
            self._forget_source()
            raise ValueError("等待采集后台连接")
        if self._future is not None:
            if self._action != "refresh":
                raise ValueError("正在处理串口操作")
            if action == "refresh":
                return
            self._cancel()
        if action != "refresh" and not self._revision:
            raise ValueError("正在读取串口配置，请稍候")
        self.error = ""
        self._send(action, port, baudrate)

    def _send(self, action, port="", baudrate=115200):
        request = self._make_request(action=action, port=port, baudrate=baudrate,
                                     expected_revision=self._revision if action != "refresh" else "")
        self._action = action
        self._sent_at = self._clock()
        self._future = self._client.call_async(request)
        self._future.add_done_callback(self._received)

    def _received(self, future):
        if self._future is not future:
            return
        self._future = None
        try:
            reply = future.result()
        except Exception as error:
            self._forget_source()
            self._transport_failed = True
            self.error = f"无法读取串口配置：{error}"
            return
        if self._transport_failed:
            self.error = ""
            self._transport_failed = False
        state = {"idle": "disconnected", "connected": "waiting"}.get(reply.state, reply.state)
        self._snapshot = SerialStatus(tuple(reply.ports), reply.port, reply.baudrate, state, reply.message)
        self._session, self._revision = reply.session_id, reply.revision
        if not reply.success:
            self.error = reply.message or "串口操作未完成，请刷新"

    def poll(self):
        """Called while the dialog is visible. Background refreshes never queue up."""
        now = self._clock()
        if not self.available:
            self._cancel()
            self._forget_source()
            return
        if self._future is not None and now - self._sent_at > 5:
            self._cancel()
            self._forget_source()
            self._transport_failed = True
            self.error = "串口后台响应超时，正在重新读取配置"
        if self._future is None and now - self._sent_at >= 1:
            self._send("refresh")
