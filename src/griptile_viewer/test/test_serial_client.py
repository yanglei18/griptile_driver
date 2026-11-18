from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from griptile_viewer.serial_client import SerialControl


class Service:
    def __init__(self):
        self.requests = []
        self.ready = True

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        self.requests.append(request)
        self.future = Future()
        return self.future

    def reply(self, revision="first", state="idle", success=True, message=""):
        self.future.set_result(SimpleNamespace(
            state=state, success=success, message=message, ports=["COM3"], baudrate=115200,
            port="COM3" if state != "idle" else "", revision=revision, session_id="source"))


def test_mutation_uses_last_revision_and_waits_for_data_from_the_new_source():
    clock = [0.]
    model = SimpleNamespace(message=SimpleNamespace(session_id="previous"),
                            mask=lambda _: np.asarray([True]))
    service = Service()
    control = SerialControl(service, SimpleNamespace, model, lambda: 0, lambda: clock[0])
    with pytest.raises(ValueError):
        control.submit("connect", "COM3")
    control.poll()
    service.reply()
    clock[0] = 2
    control.poll()
    background_read = service.future
    assert not control.pending
    control.submit("connect", "COM3", 57600)
    assert background_read.cancelled() and control.pending
    assert service.requests[-1].expected_revision == "first"
    assert service.requests[-1].baudrate == 57600
    service.reply("second", "connected")
    assert control.status.state == "waiting"
    model.message.session_id = "source"
    assert control.status.state == "streaming"
    model.mask = lambda _: np.asarray([False])
    assert control.status.state == "waiting"
    service.ready = False
    control.poll()
    assert control.status is None and not control.available


def test_timeout_recovers_and_background_poll_preserves_a_rejected_action():
    clock = [0.]
    service = Service()
    control = SerialControl(service, SimpleNamespace, SimpleNamespace(message=None), lambda: 0, lambda: clock[0])
    control.poll()
    lost_reply = service.future
    clock[0] = 6.
    control.poll()
    assert lost_reply.cancelled() and "超时" in control.error
    service.reply()
    assert control.error == "" and not control.pending
    control.submit("connect", "COM3")
    service.reply("second", success=False, message="配置已变更")
    clock[0] += 2
    control.poll()
    service.reply("second")
    assert control.error == "配置已变更"
    control.submit("refresh")
    assert not control.error
