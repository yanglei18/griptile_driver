"""Operator exports. A snapshot preserves source times and display freshness."""

import csv
import json
import os
from pathlib import Path
import tempfile

from griptile_core.display_response import ReferenceProfile
from griptile_core.grid import ROWS, COLS


def save_reference(view, path):
    binding = view.binding()
    if not binding or view.reference is None:
        raise ValueError("保存参考需要已核验的设备标识和完成的无载参考")
    document = {"version": 1, "binding": binding, "reference": view.reference.to_dict()}
    destination = Path(path)
    with tempfile.NamedTemporaryFile("w", dir=destination.parent, encoding="utf-8", delete=False) as stream:
        temporary = stream.name
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(document, stream, allow_nan=False)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_reference(view, path):
    binding = view.binding()
    if not binding or view.message is None or not view.message.connected:
        raise ValueError("加载参考需要已连接且具有可核验标识的传感器")
    with open(path, "rb") as stream:
        raw = stream.read(131073)
    if len(raw) > 131072:
        raise ValueError("参考文件过大")
    doc = json.loads(raw)
    if not isinstance(doc, dict) or doc.get("version") != 1 or doc.get("binding") != binding:
        raise ValueError("参考文件与当前传感器、协议或布局不匹配")
    return ReferenceProfile.from_dict(doc.get("reference"))


def save_snapshot(view, path, clock_ns):
    message = view.message
    if message is None:
        raise ValueError("尚未收到阵列快照")
    fresh = view.mask(clock_ns)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sensor_id", "session_id", "stream_epoch", "source_sequence", "row", "column",
                         "adc_counts", "source_valid", "display_valid", "updated", "row_realtime_ns",
                         "row_monotonic_ns", "snapshot_monotonic_ns"))
        for row in range(ROWS):
            for col in range(COLS):
                index = row * COLS + col
                writer.writerow((view.sensor_id, message.session_id, message.stream_epoch, message.source_sequence,
                                 row, col, message.adc[index], int(message.valid_mask[index]), int(fresh[row, col]),
                                 int(message.updated_mask[index]), message.row_realtime_ns[row],
                                 message.row_monotonic_ns[row], message.snapshot_monotonic_ns))
