"""Time-aware view of rolling source snapshots. No Qt or ROS dependency."""

from collections import deque

import numpy as np

from griptile_core.contact_image import INVALID_RGB
from griptile_core.display_response import ReferenceProfile
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
        # Independent DDS topics can arrive in either order. Retain only a few
        # snapshots and images to match source stamps without blocking Qt.
        self._states = deque(maxlen=4)
        self._images = deque(maxlen=4)
        self.reference_status = None
        self.service_error = ""
        self._profile = None
        self.info = None
        self.diagnostic = ""
        self.receive_clock = 0
        self.last_clock = None
        self.identity = None
        self.message = None
        self.rejection = ""

    def reset_stream(self):
        self._states.clear()
        self._images.clear()
        self.trend.clear()
        self.message = None
        self.identity = None

    def track_clock(self, clock_ns):
        """ROS time may rewind on a bag seek. Discard the previous playback position."""
        if self.replay and self.last_clock is not None and clock_ns < self.last_clock:
            self.retired.clear()
            self.reset_stream()
            self.reference_status = self._profile = None
            self.info = None
        self.last_clock = clock_ns

    def accept_info(self, info):
        if (info.sensor_id == self.sensor_id and info.layout_id == LAYOUT_ID
                and info.protocol_id == PROTOCOL_ID and info.unit == "adc_counts"):
            self.info = info

    @property
    def current_reference(self):
        status = self.reference_status
        if status is None or (status.session_id, status.stream_epoch) != self.identity:
            return None
        if (self.info and (self.info.session_id, self.info.stream_epoch) == self.identity
                and self.info.hardware_id != status.hardware_id):
            return None
        return status

    @property
    def reference(self):
        return self._profile if self.current_reference is not None else None

    def accept_reference(self, status):
        if (status.sensor_id != self.sensor_id or (status.session_id, status.stream_epoch) in self.retired):
            return
        profile = None
        if status.phase == "ready":
            profile = ReferenceProfile(np.asarray(status.baseline_adc), np.asarray(status.activation_threshold_adc),
                                       np.asarray(status.valid_mask, dtype=bool))
        self.reference_status, self._profile = status, profile

    @staticmethod
    def _key(header):
        return (header.frame_id, header.stamp.sec, header.stamp.nanosec)

    def accept_image(self, image):
        if (image.encoding != "rgb8" or (image.height, image.width) != SHAPE
                or image.step != COLS * 3 or len(image.data) != CELL_COUNT * 3):
            raise ValueError("接触图需要 20×20 个 RGB8 像素")
        pixels = np.frombuffer(bytes(image.data), dtype=np.uint8).reshape(ROWS, COLS, 3)
        self._images.append((self._key(image.header), pixels))

    def contact_pixels(self, clock_ns):
        unknown = np.full((ROWS, COLS, 3), INVALID_RGB, dtype=np.uint8)
        if self.reference is None or not self._images:
            return unknown
        for key, state, received in reversed(self._states):
            pixels = next((rgb for stamp, rgb in reversed(self._images) if stamp == key), None)
            if pixels is not None:
                fresh = self._mask_at(state, received, clock_ns) & self.mask(clock_ns)
                valid = fresh & np.asarray(self.reference.valid_mask).reshape(SHAPE)
                unknown[valid] = pixels[valid]
                break
        return unknown

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
        if not message.connected or not message.time_valid:
            self._states.clear()
            self._images.clear()
            self._break_trend()
            return True
        self._states.append((self._key(message.header), message, clock_ns))
        self._sample_trend()
        return True

    def invalidate(self, reason):
        self.reset_stream()
        self.rejection = reason

    def mask(self, clock_ns):
        return self._mask_at(self.message, self.receive_clock, clock_ns)

    @staticmethod
    def _mask_at(message, received, clock_ns):
        if message is None or not message.connected or not message.time_valid:
            return np.zeros(SHAPE, dtype=bool)
        # Subtract only timestamps from the same machine. Driver supplies age at
        # production; local elapsed time since receipt advances that lower bound.
        elapsed = max(0, int(clock_ns) - received)
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

    def binding(self):
        info = self.info
        if (self.identity and info and (info.session_id, info.stream_epoch) == self.identity
                and info.hardware_id):
            return {"sensor_id": self.sensor_id, "hardware_id": info.hardware_id,
                    "protocol_id": PROTOCOL_ID, "layout_id": LAYOUT_ID}
        return None

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
