"""ROS conversion shared by live and hardware-free replay; no serial or Qt."""

from dataclasses import fields

from sensor_msgs.msg import Image
from griptile_msgs.msg import TactileRow44, TactileRow44Array, TactileGridState

from griptile_core.grid import CELL_COUNT, COLS, ROWS, normalize_state
from griptile_core.protocol import Block
from griptile_core.serial_source import Batch, CLOCK_DOMAIN, TIMING_SOURCE


def set_stamp(header, realtime_ns: int, frame_id: str) -> None:
    header.stamp.sec, header.stamp.nanosec = divmod(int(realtime_ns), 1_000_000_000)
    header.frame_id = frame_id


def sample_message(batch, frame_id: str) -> TactileRow44Array:
    result = TactileRow44Array()
    for field in fields(batch):
        if field.name != "blocks":
            setattr(result, field.name, getattr(batch, field.name))
    result.blocks = [TactileRow44(row=block.row, adc=list(block.adc))
                     for block in batch.blocks]
    set_stamp(result.header, batch.host_realtime_ns, frame_id)
    return result


def batch_from_message(message, sensor_id: str) -> Batch:
    """Validate the ROS envelope before forwarding recorded rows to the core."""
    stamp = message.header.stamp
    stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
    if (message.schema_version != 2 or message.sensor_id != sensor_id
            or message.host_realtime_ns != stamp_ns or message.clock_domain != CLOCK_DOMAIN
            or message.timing_source != TIMING_SOURCE):
        raise ValueError("Sample schema, device identity or receive clock mismatch")
    if any(not 0 <= block.row < ROWS or len(block.adc) != COLS for block in message.blocks):
        raise ValueError("Invalid recorded row")
    values = {field.name: getattr(message, field.name) for field in fields(Batch)
              if field.name != "blocks"}
    # ROS fixed arrays can contain NumPy scalar integers. Normalize at the boundary.
    blocks = tuple(Block(row=int(row.row), adc=tuple(map(int, row.adc))) for row in message.blocks)
    return Batch(**values, blocks=blocks)


def state_message(snapshot, frame_id: str) -> TactileGridState:
    result = TactileGridState()
    for field in fields(snapshot):
        value = getattr(snapshot, field.name)
        setattr(result, field.name, list(value) if isinstance(value, tuple) else value)
    set_stamp(result.header, snapshot.host_realtime_ns, frame_id)
    return result


def _preview_lut():
    # Fixed plasma-style color scale for the optional RGB image topic.
    anchors = [(13, 8, 135), (75, 3, 161), (126, 3, 168), (168, 34, 150),
               (204, 71, 120), (229, 107, 93), (248, 148, 65), (253, 195, 40), (240, 249, 33)]
    table = []
    for index in range(256):
        position = index * (len(anchors) - 1) / 255
        lo = min(int(position), len(anchors) - 2)
        fraction = position - lo
        table.append(bytes(round(a + fraction * (b - a))
                           for a, b in zip(anchors[lo], anchors[lo + 1])))
    return table


_PREVIEW_LUT = _preview_lut()


def heatmap_message(state: TactileGridState) -> Image:
    """Native 20x20 ADC preview over the unsigned 16-bit scale; no force units."""
    state = normalize_state(state)
    if len(state.adc) != CELL_COUNT or len(state.valid_mask) != CELL_COUNT:
        raise ValueError("Heatmap requires 400 ADC values and validity flags")
    result = Image()
    result.header = state.header
    result.height, result.width = state.rows, state.cols
    result.encoding = "rgb8"
    result.is_bigendian = 0
    result.step = state.cols * 3
    result.data = b"".join(
        _PREVIEW_LUT[max(0, min(255, int(adc) * 255 // 65535))]
        if valid else b"\0\0\0"
        for adc, valid in zip(state.adc, state.valid_mask)
    )
    return result
