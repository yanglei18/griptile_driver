import csv
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from griptile_core.assembler import Assembler
from griptile_core.grid import LAYOUT_ID
from griptile_core.protocol import Block, PROTOCOL_ID
from griptile_core.serial_source import Batch
from griptile_viewer.model import ArrayView
from griptile_viewer.storage import load_reference, save_reference, save_snapshot


SENSOR = "grip_pad"
START = 9_000_000_000


def snapshot(assembler, n, t, rows=range(20), *, epoch=1, offset=0):
    packet = Batch(sensor_id=SENSOR, session_id="fixture", stream_epoch=epoch,
                   sequence=n, first_packet_index=(n - 1) * len(rows), host_realtime_ns=t + 10**12,
                   host_monotonic_ns=t, connected=True, time_valid=True,
                   blocks=tuple(Block(row, tuple(row * 20 + c + offset for c in range(20))) for row in rows))
    assembler.apply(packet)
    return assembler.snapshot(t, 100_000_000)


def info(hardware_id="test-board", epoch=1):
    return SimpleNamespace(sensor_id=SENSOR, session_id="fixture", stream_epoch=epoch, hardware_id=hardware_id,
                           layout_id=LAYOUT_ID, protocol_id=PROTOCOL_ID, unit="adc_counts")


def collect(view, *, rows=range(20)):
    assembler = Assembler(sensor_id=SENSOR)
    message = snapshot(assembler, 1, START, rows)
    view.accept(message, 0)
    view.start_reference(0)
    for index in range(1, 31):
        elapsed = index * 100_000_000
        message = snapshot(assembler, index + 1, START + elapsed, rows)
        view.accept(message, elapsed)
    return message


def test_remote_clock_mask_orientation_and_export(tmp_path):
    model = ArrayView(SENSOR)
    assembler = Assembler(sensor_id=SENSOR)
    message = snapshot(assembler, 1, START, (0, 3))
    model.accept(message, 0)
    values = model.values(0)
    assert values[0, 0] == 0
    assert values[3, 7] == 67
    assert np.isnan(values[1, 0])
    assert not model.mask(100_000_001).any()
    assert not model.accept(message, 100_000_002)  # duplicated UI frame cannot extend validity
    path = tmp_path / "frame.csv"
    save_snapshot(model, path, 100_000_002)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 400
    assert rows[67]["adc_counts"] == "67"
    assert (rows[67]["source_valid"], rows[67]["display_valid"]) == ("1", "0")
    assert rows[67]["row_monotonic_ns"] == str(START)


def test_unloaded_reference_requires_recent_distinct_rows_and_signed_delta(tmp_path):
    model = ArrayView(SENSOR)
    model.accept_info(info())
    message = collect(model, rows=(2,))
    assert model.reference is not None
    assert sum(model.reference.valid_mask) == 20
    adc = list(message.adc)
    adc[44] = 5
    changed = replace(message, adc=adc, source_sequence=100, snapshot_monotonic_ns=message.snapshot_monotonic_ns+1)
    model.accept(changed, 3_000_000_001)
    assert model.values(3_000_000_001, True)[2, 4] == -39
    assert np.isnan(model.values(3_000_000_001, True)[0, 0])
    path = tmp_path / "reference.json"
    save_reference(model, path)
    before = model.reference
    model.accept_info(info("another-board"))
    with pytest.raises(ValueError):
        load_reference(model, path)
    assert model.reference is None
    model.accept_info(info())
    load_reference(model, path)
    assert model.reference == before
    model.accept_info(info(""))
    with pytest.raises(ValueError):
        save_reference(model, path)


def test_repeated_row_clock_never_becomes_new_reference():
    model = ArrayView(SENSOR)
    message = snapshot(Assembler(sensor_id=SENSOR), 1, START)
    model.accept(message, 0)
    model.start_reference(0)
    for index in range(1, 32):
        elapsed = index * 100_000_000
        model.accept(replace(message, source_sequence=index + 1, snapshot_monotonic_ns=START+elapsed), elapsed)
    assert model.reference is None
    assert model.response.status.phase == "failed"


def test_replay_seek_clears_reference_and_history():
    model = ArrayView(SENSOR, replay=True)
    collect(model)
    assert model.reference is not None
    model.track_clock(0)
    assert model.reference is None and not model.trend
    assert not model.mask(0).any()


def test_qt_matrix_smoke(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from griptile_viewer.window import QtWidgets, ViewerWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    model = ArrayView(SENSOR)
    model.accept(snapshot(Assembler(sensor_id=SENSOR), 1, START, (0, 2)), 0)
    window = ViewerWindow(model, clock_ns=lambda: 0)
    window.show()
    app.processEvents()
    model.select(2, 3)
    window.refresh()
    assert window.image.axisOrder == "row-major"
    assert window.image.image[2, 3] == 43
    assert np.isnan(window.image.image[1, 0])
    assert "ADC 43" in window.cell_label.text()
    assert window.map.grab().save(str(tmp_path / "matrix.png"), "PNG")
    window.close()
    app.processEvents()
