"""Time-aware view of rolling source snapshots. No Qt or ROS dependency."""

from collections import deque
from dataclasses import replace

import numpy as np

from griptile_core.contact_image import INVALID_RGB
from griptile_core.display_response import ReferenceProfile
from griptile_core.grid import CELL_COUNT, COLS, LAYOUT_ID, ROWS, SHAPE, normalize_state
from griptile_core.identity import validate_sensor_id
from griptile_core.protocol import PROTOCOL_ID

from .load_capture import LoadContext, LoadObservation


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
        self._responses = deque(maxlen=4)
        self._load_reference = None
        self._load_revision = 0
        self._load_watermark = None
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
        self._responses.clear()
        self.trend.clear()
        self.message = None
        self.identity = None

    def track_clock(self, clock_ns):
        """ROS time may rewind on a bag seek. Discard the previous playback position."""
        if self.replay and self.last_clock is not None and clock_ns < self.last_clock:
            self.retired.clear()
            self.reset_stream()
            self.reference_status = self._profile = None
            self._load_reference = self._load_watermark = None
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
            if not 0 < status.span_adc <= 65535:
                raise ValueError("接触参考量程超出 ADC 范围")
            profile = ReferenceProfile(np.asarray(status.baseline_adc), np.asarray(status.activation_threshold_adc),
                                       np.asarray(status.valid_mask, dtype=bool))
        self.reference_status, self._profile = status, profile
        self._bind_load_reference(status, profile)

    def _bind_load_reference(self, status, profile):
        # image headers identify an acquisition, not which reference generated it.
        # Track the ready event and coefficients, including direct ready->ready loads.
        context = LoadContext.from_reference(status, profile, self._load_revision)
        header = getattr(status, "header", None)
        if header is not None:
            stream = status.session_id, int(status.stream_epoch)
            stamp = int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)
            if self._load_watermark and self._load_watermark[0] == stream and stamp < self._load_watermark[1]:
                self._load_reference = None
                return
            self._load_watermark = stream, stamp
        if self._load_reference != context:
            self._load_revision += 1
            self._load_reference = replace(context, generation=self._load_revision) if context else None

    @property
    def load_context(self):
        message = self.message
        if (self.reference is not None and message is not None and message.connected
                and message.time_valid and self._load_reference is not None):
            return self._load_reference
        return None

    @staticmethod
    def _key(header):
        return (header.frame_id, header.stamp.sec, header.stamp.nanosec)

    def accept_image(self, image):
        if (image.encoding != "rgb8" or (image.height, image.width) != SHAPE
                or image.step != COLS * 3 or len(image.data) != CELL_COUNT * 3):
            raise ValueError("接触图需要 20×20 个 RGB8 像素")
        pixels = np.frombuffer(bytes(image.data), dtype=np.uint8).reshape(ROWS, COLS, 3)
        self._images.append((self._key(image.header), pixels))

    def accept_response(self, image):
        if (image.encoding != "32FC1" or (image.height, image.width) != SHAPE
                or image.step != COLS * 4 or len(image.data) != CELL_COUNT * 4):
            raise ValueError("接触响应用 20×20 个单通道浮点数表示")
        dtype = ">f4" if image.is_bigendian else "<f4"
        amplitude = np.frombuffer(image.data, dtype=dtype).astype(np.float32).reshape(SHAPE)
        if np.isinf(amplitude).any() or (amplitude < 0).any():
            raise ValueError("接触响应应为非负数或缺测值")
        self._responses.append((self._key(image.header), amplitude))

    def _paired_sample(self, frames, clock_ns):
        reference = self.reference
        if reference is not None and frames:
            for key, state, received in reversed(self._states):
                frame = next((data for stamp, data in reversed(frames) if stamp == key), None)
                if frame is not None:
                    fresh = self._mask_at(state, received, clock_ns) & self.mask(clock_ns)
                    return frame, fresh & np.asarray(reference.valid_mask).reshape(SHAPE), state
        return None, None, None

    def load_observation(self, clock_ns):
        """Observe the same native response as the colors, before display saturation."""
        context = self.load_context
        if context is None:
            return None
        frame, valid, state = self._paired_sample(self._responses, clock_ns)
        if frame is None or not valid.all() or not np.isfinite(frame).all():
            return None
        # New and previous references can publish different images with exactly
        # the same state header at the switch. Start after all rows cross it.
        if (state.header.frame_id != context.frame_id
                or min(state.row_realtime_ns) <= context.reference_stamp_ns):
            return None
        baseline = np.asarray(self.reference.baseline_adc).reshape(SHAPE)
        signed = frame * np.sign(np.asarray(state.adc).reshape(SHAPE) - baseline)
        source_now = int(self.message.snapshot_monotonic_ns) + max(0, clock_ns - self.receive_clock)
        return LoadObservation(context, tuple(map(int, state.row_monotonic_ns)), source_now,
                               float(frame.sum(dtype=np.float64)), float(signed.sum(dtype=np.float64)),
                               float(frame.max()), int(np.count_nonzero(frame)))

    def contact_pixels(self, clock_ns):
        image = np.full((ROWS, COLS, 3), INVALID_RGB, dtype=np.uint8)
        frame, valid, _ = self._paired_sample(self._images, clock_ns)
        if frame is not None:
            image[valid] = frame[valid]
        return image

    def contact_response(self, clock_ns):
        amplitude = np.full(SHAPE, np.nan, dtype=np.float32)
        frame, valid, _ = self._paired_sample(self._responses, clock_ns)
        if frame is not None:
            amplitude[valid] = frame[valid]
        return amplitude

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
            if not message.connected:
                self._load_reference = None
            self._states.clear()
            self._images.clear()
            self._responses.clear()
            self._break_trend()
            return True
        self._states.append((self._key(message.header), message, clock_ns))
        self._sample_trend()
        return True

    def invalidate(self, reason):
        self.reset_stream()
        self._load_reference = None
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
