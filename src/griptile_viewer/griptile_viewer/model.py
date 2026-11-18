"""Time-aware view of rolling source snapshots. No Qt or ROS dependency."""

from collections import deque

import numpy as np

from griptile_core.display_response import TactileResponseProcessor
from griptile_core.grid import CELL_COUNT, COLS, LAYOUT_ID, ROWS, SHAPE, normalize_state
from griptile_core.identity import validate_sensor_id
from griptile_core.protocol import PROTOCOL_ID


class ArrayView:
    def __init__(self, sensor_id, *, replay=False):
        validate_sensor_id(sensor_id)
        self.sensor_id = sensor_id
        self.replay = replay
        self.selected = (0, 0)
        self.retired = deque(maxlen=16)
        self.trend = deque(maxlen=600)
        self.response = TactileResponseProcessor()
        self.info = None
        self.diagnostic = ""
        self.receive_clock = 0
        self.last_clock = None
        self.identity = None
        self.message = None
        self.reference = None
        self.rejection = ""

    def reset_stream(self):
        self.response.reset()
        self.reference = None
        self.trend.clear()
        self.message = None
        self.identity = None

    def track_clock(self, clock_ns):
        """ROS time may rewind on a bag seek. Discard the previous playback position."""
        if self.replay and self.last_clock is not None and clock_ns < self.last_clock:
            self.retired.clear()
            self.reset_stream()
        self.last_clock = clock_ns

    def accept_info(self, info):
        if (info.sensor_id == self.sensor_id and info.layout_id == LAYOUT_ID
                and info.protocol_id == PROTOCOL_ID and info.unit == "adc_counts"):
            previous = self.binding()
            self.info = info
            if (previous is not None and (info.session_id, info.stream_epoch) == self.identity
                    and self.binding() != previous):
                self.response.reset("sensor identity changed")
                self.reference = None

    def accept(self, message, clock_ns):
        if message.sensor_id != self.sensor_id:
            self.rejection = "传感器名称不匹配"
            return False
        normalize_state(message)
        if (not message.session_id and message.connected) or message.unit != "adc_counts":
            raise ValueError("invalid acquisition identity or unit")
        if (any(len(vector) != CELL_COUNT for vector in
                (message.adc, message.valid_mask, message.updated_mask))
                or len(message.row_monotonic_ns) != ROWS or len(message.row_realtime_ns) != ROWS):
            raise ValueError("invalid row or cell array length")

        self.track_clock(clock_ns)
        identity = (message.session_id, int(message.stream_epoch))
        if identity != self.identity:
            if (identity in self.retired or (self.identity and identity[0] == self.identity[0]
                                             and identity[1] < self.identity[1])):
                return False
            if self.identity:
                self.retired.append(self.identity)
            self.reset_stream()
            self.identity = identity
        elif self.message is not None:
            if (message.source_sequence < self.message.source_sequence
                    or message.snapshot_monotonic_ns <= self.message.snapshot_monotonic_ns):
                return False

        self.rejection = ""
        self.message = message
        self.receive_clock = clock_ns
        instant = int(message.snapshot_monotonic_ns)
        if not message.connected or not message.time_valid:
            self.response.cancel_reference(message.reason or "no source")
            self._break_trend()
            return True
        if self.response.status.phase == "collecting":
            sample_times = np.repeat(message.row_monotonic_ns, COLS).astype(np.int64)
            self.response.observe(message.adc, message.valid_mask, sample_times, identity, instant)
            if self.response.status.phase == "ready":
                self.reference = self.response.export_reference()
        self._sample_trend()
        return True

    def invalidate(self, reason):
        self.reset_stream()
        self.rejection = reason

    def mask(self, clock_ns):
        message = self.message
        if message is None or not message.connected or not message.time_valid:
            return np.zeros(SHAPE, dtype=bool)
        # Subtract only timestamps from the same machine. Driver supplies age at
        # production; local elapsed time since receipt advances that lower bound.
        elapsed = max(0, int(clock_ns) - self.receive_clock)
        now_source = int(message.snapshot_monotonic_ns) + elapsed
        row_fresh = [
            stamp > 0 and 0 <= now_source - int(stamp) <= message.max_age_ns
            for stamp in message.row_monotonic_ns
        ]
        return np.asarray(message.valid_mask, dtype=bool).reshape(SHAPE) & np.asarray(row_fresh)[:, None]

    def values(self, clock_ns, delta=False):
        message = self.message
        if message is None:
            return np.full(SHAPE, np.nan)
        data = np.asarray(message.adc, dtype=float).reshape(SHAPE).copy()
        data[~self.mask(clock_ns)] = np.nan
        if delta:
            if self.reference is None:
                data.fill(np.nan)
            else:
                data -= np.asarray(self.reference.baseline_adc).reshape(SHAPE)
                data[~np.asarray(self.reference.valid_mask).reshape(SHAPE)] = np.nan
        return data

    def start_reference(self, clock_ns):
        if not self.mask(clock_ns).any():
            raise ValueError("请等到有新鲜触觉数据时再采集参考")
        self.reference = None
        self.response.begin_reference(self.identity, self.message.snapshot_monotonic_ns)

    def binding(self):
        info = self.info
        if (self.identity and info and (info.session_id, info.stream_epoch) == self.identity
                and info.hardware_id):
            return {"sensor_id": self.sensor_id, "hardware_id": info.hardware_id,
                    "protocol_id": PROTOCOL_ID, "layout_id": LAYOUT_ID}
        return None

    def adopt_reference(self, profile):
        self.response.load_reference(profile, self.identity, self.message.snapshot_monotonic_ns)
        self.reference = profile

    def select(self, row, column):
        if self.selected != (row, column) and 0 <= row < ROWS and 0 <= column < COLS:
            self.selected = row, column
            self.trend.clear()
            if self.message:
                self._sample_trend()

    def _break_trend(self):
        if self.trend and np.isfinite(self.trend[-1][1]):
            self.trend.append((self.trend[-1][0], np.nan))

    def _sample_trend(self):
        row, column = self.selected
        message = self.message
        stamp = int(message.row_monotonic_ns[row])
        if not self.mask(self.receive_clock)[row, column]:
            self._break_trend()
            return
        if self.trend and stamp <= self.trend[-1][0]:
            return
        if self.trend and stamp - self.trend[-1][0] > message.max_age_ns:
            self._break_trend()
        self.trend.append((stamp, int(message.adc[row * COLS + column])))
