"""An explicit live zero follows only the same verified device across sessions."""

from contextlib import ExitStack
import json
from unittest.mock import Mock

import numpy as np
import pytest
import rclpy
from rclpy.context import Context
from rclpy.parameter import Parameter

from griptile_core.grid import LAYOUT_ID
from griptile_core.protocol import PROTOCOL_ID
from griptile_driver.node import TactileNode
from griptile_driver.replay import TactileReplayNode
from griptile_msgs.msg import SensorInfo, TactileGridState
from griptile_msgs.srv import SetContactReference


@pytest.fixture
def contact(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    ros = Context()
    rclpy.init(context=ros, domain_id=204, args=[])
    with ExitStack() as cleanup:
        cleanup.callback(ros.shutdown)

        def create(cls=TactileNode, **overrides):
            parameters = {"sensor_id": "pad_a"} | overrides
            node = cls(context=ros, parameter_overrides=[
                Parameter(name, value=value) for name, value in parameters.items()])
            cleanup.callback(node.destroy_node)
            return node.contact

        yield create


def info(*, session="test", epoch=1, **overrides):
    fields = dict(schema_version=1, sensor_id="pad_a", session_id=session, stream_epoch=epoch,
                  connected=True, hardware_id="usb:test-board", protocol_id=PROTOCOL_ID,
                  layout_id=LAYOUT_ID, unit="adc_counts", rows=20, cols=20)
    return SensorInfo(**(fields | overrides))


def frame(*, session="test", epoch=1, tick=0, adc=12000):
    stamp = 1_000_000_000 + tick * 100_000_000
    return TactileGridState(schema_version=3, sensor_id="pad_a", session_id=session,
                            stream_epoch=epoch, connected=True, time_valid=True,
                            layout_id=LAYOUT_ID, unit="adc_counts", rows=20, cols=20,
                            snapshot_monotonic_ns=stamp, row_monotonic_ns=[stamp] * 20,
                            valid_mask=[True] * 400, adc=[adc] * 400)


def request(imaging, **fields):
    return imaging.set_reference(SetContactReference.Request(**fields), SetContactReference.Response())


def import_profile(imaging):
    imaging.update_info(info())
    imaging.update(frame())
    assert request(imaging, sensor_id="pad_a", session_id="test", stream_epoch=1,
                   hardware_id="usb:test-board", load_profile=True, baseline_adc=[12000.] * 400,
                   activation_threshold_adc=[3.] * 400, valid_mask=[True] * 400).success


def test_capture_then_restore_on_first_loaded_frame_of_new_session(contact, tmp_path, monkeypatch):
    first = contact()
    first.update_info(info())
    first.update(frame())
    assert first.processor.status.phase == "unreferenced"
    assert request(first).success
    for tick in range(1, 31):
        first.update(frame(tick=tick))
    measured = first.processor.export_reference()
    assert first.processor.status.reference_source == "measured"
    file, = tmp_path.rglob("*.json")
    saved = json.loads(file.read_bytes())
    assert saved["reference"] == measured.to_dict() and saved["direction"] == "both"
    assert saved["binding"] == dict(sensor_id="pad_a", hardware_id="usb:test-board",
                                     layout_id=LAYOUT_ID, protocol_id=PROTOCOL_ID)

    second = contact()
    read = Mock(wraps=second.reference_cache.load)
    write = Mock(side_effect=AssertionError("restoring a reference must not rewrite it"))
    monkeypatch.setattr(second.reference_cache, "load", read)
    monkeypatch.setattr(second.reference_cache, "save", write)
    second.response_publisher = responses = Mock()
    second.reference_publisher = references = Mock()
    second.update_info(info(session="next", epoch=3))
    for tick in range(3):
        second.update(frame(session="next", epoch=3, tick=tick, adc=14000))
        response = responses.publish.call_args.args[0]
        assert np.all(np.frombuffer(response.data, dtype="<f4") == 1998.5)
    read.assert_called_once()
    assert second.processor.export_reference() == measured
    report = references.publish.call_args.args[0]
    assert (report.session_id, report.stream_epoch, report.reason) == ("next", 3, "saved_reference")

    # Reaching the same USB port is insufficient if its hardware changed.
    second.update_info(info(session="next", epoch=3, hardware_id="usb:other-board"))
    second.update(frame(session="next", epoch=3, tick=4))
    assert second.processor.status.phase == "unreferenced"
    assert read.call_count == 2


def test_waits_for_matching_metadata_and_fresh_rows_but_never_adopts_during_collection(contact):
    import_profile(contact())
    later = contact()
    later.update(frame())
    assert later.processor.status.phase == "unreferenced"
    for mismatch in (dict(sensor_id="other"), dict(session="other"), dict(epoch=7),
                     dict(hardware_id=""), dict(layout_id="other"), dict(protocol_id="other"),
                     dict(unit="Pa"), dict(schema_version=42), dict(rows=19)):
        later.update_info(info(**mismatch))
        later.update(frame(tick=1))
        assert later.processor.status.phase == "unreferenced"
    later.update_info(info())
    stale = frame(tick=2)
    stale.time_valid = False
    later.update(stale)
    assert later.processor.status.phase == "unreferenced"
    later.update(frame(tick=3))
    assert later.processor.status.reference_source == "saved"

    unknown = contact()
    unknown.update(frame())
    assert request(unknown).success
    unknown.update_info(info())  # Identity was missing when acquisition started.
    for tick in range(1, 31):
        unknown.update(frame(tick=tick, adc=15000))
    assert unknown.processor.export_reference().baseline_adc[0] == 15000.
    recovered = contact()
    recovered.update_info(info())
    recovered.update(frame())
    assert recovered.processor.export_reference().baseline_adc[0] == 12000.


def test_corrupt_legacy_and_incompatible_files_do_not_become_automatic_zero(contact, tmp_path, monkeypatch):
    import_profile(contact())
    file, = tmp_path.rglob("*.json")
    document = json.loads(file.read_bytes())
    for changes in (None, {"direction": "decrease"}, {"version": 1},
                    {"binding": document["binding"] | {"hardware_id": "usb:other"}},
                    {"reference": document["reference"] | {"baseline_adc": [-1] * 400}}):
        file.write_text("{" if changes is None else json.dumps(document | changes))
        first = contact()
        read = Mock(wraps=first.reference_cache.load)
        monkeypatch.setattr(first.reference_cache, "load", read)
        first.update_info(info())
        first.update(frame())
        first.update(frame(tick=1))
        read.assert_called_once()
        assert first.processor.status.phase == "unreferenced"
    file.write_text(json.dumps(document))
    reverse = contact(contact_direction="decrease")
    reverse.update_info(info())
    reverse.update(frame())
    assert reverse.processor.status.phase == "unreferenced"
    other_sensor = contact(sensor_id="pad_b")
    other_sensor.update_info(info(sensor_id="pad_b"))
    other_frame = frame()
    other_frame.sensor_id = "pad_b"
    other_sensor.update(other_frame)
    assert other_sensor.processor.status.phase == "unreferenced"
    replay = contact(TactileReplayNode)
    replay.update_info(info())
    replay.update(frame())
    assert replay.processor.status.phase == "unreferenced"


def test_storage_error_does_not_break_session_or_overwrite_previous_zero(contact, monkeypatch):
    import_profile(contact())
    next_session = contact()
    write = Mock(side_effect=OSError("read-only fixture directory"))
    monkeypatch.setattr(next_session.reference_cache, "save", write)
    next_session.update_info(info())
    next_session.update(frame())
    assert request(next_session).success
    for tick in range(1, 32):
        next_session.update(frame(tick=tick, adc=15000))
    write.assert_called_once()
    assert next_session.processor.status.reference_source == "measured"
    assert next_session.processor.export_reference().baseline_adc[0] == 15000.
    recovered = contact()
    recovered.update_info(info())
    recovered.update(frame())
    assert recovered.processor.export_reference().baseline_adc[0] == 12000.


def test_demo_auto_collection_never_creates_an_operator_cache(contact, tmp_path):
    demo = contact(contact_auto_reference=True)
    demo.update_info(info())
    for tick in range(32):
        demo.update(frame(tick=tick, adc=28000))
    assert demo.processor.status.reference_source == "measured"
    assert not list(tmp_path.rglob("*.json"))
    import_profile(contact())
    next_demo = contact(contact_auto_reference=True)
    next_demo.update_info(info())
    next_demo.update(frame(adc=28000))
    assert next_demo.processor.status.reference_source == "saved"
    assert next_demo.processor.export_reference().baseline_adc[0] == 12000.
