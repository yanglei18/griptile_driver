"""Time-aware view of rolling source snapshots. No Qt or ROS dependency."""

from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from griptile_core.contact_image import INVALID_RGB, contact_footprint
from griptile_core.display_response import ReferenceProfile
from griptile_core.grid import CELL_COUNT, COLS, LAYOUT_ID, ROWS, SHAPE, normalize_state
from griptile_core.identity import validate_sensor_id
from griptile_core.light_contact import LightContactTracker
from griptile_core.protocol import PROTOCOL_ID

from .load_capture import LoadContext, LoadObservation
from .print_capture import ContactPrintCapture


@dataclass(frozen=True)
class UniformObservation:
    context: LoadContext
    row_stamps: tuple
    source_now_ns: int
    raw_adc: np.ndarray


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
        self.contact_prints = ContactPrintCapture()
        self._reset_light_contact()
        self._reset_prints()
        self._has_measured_response = False
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
        self._reset_light_contact()
        self._reset_prints()
        self._states.clear()
        self._images.clear()
        self._responses.clear()
        self._has_measured_response = False
        self.trend.clear()
        self.message = None
        self.identity = None

    def _reset_light_contact(self):
        self._light_tracker = LightContactTracker()
        self._light_context = self._light_stamp = self._light_mask = None

    def _reset_prints(self):
        self.contact_prints.reset()
        self._print_context = self._print_profile = None
        self._print_stamp = self._print_received = None
        self._print_rows = np.zeros(ROWS, dtype=np.int64)

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
        if info.sensor_id != self.sensor_id or (info.session_id, info.stream_epoch) in self.retired:
            return
        if info.layout_id == LAYOUT_ID and info.protocol_id == PROTOCOL_ID and info.unit == "adc_counts":
            self.info = info
        elif (info.session_id, info.stream_epoch) == self.identity:
            # Superseded metadata must not authorize a stored calibration on
            # an acquisition that now reports an incompatible layout/protocol.
            self.info = None

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

    @property
    def equalization_context(self):
        """The live reference and sensor info must identify the same acquisition."""
        context = self.load_context
        info = self.info
        if (self.replay or context is None or info is None or not getattr(info, "connected", True)
                or (info.session_id, info.stream_epoch) != self.identity
                or (context.session_id, context.stream_epoch) != self.identity
                or context.frame_id != self.message.header.frame_id
                or info.hardware_id != context.hardware_id):
            return None
        return context

    def uniform_observation(self, clock_ns):
        context = self.equalization_context
        state = self.message
        if (context is None or not all(self.reference.valid_mask) or not self.mask(clock_ns).all()
                or not self._rows_after_reference(state, context).all()):
            return None
        source_now = int(state.snapshot_monotonic_ns) + max(0, clock_ns - self.receive_clock)
        return UniformObservation(context, tuple(map(int, state.row_monotonic_ns)), source_now,
                                  np.asarray(state.adc, dtype=np.uint16).reshape(SHAPE).copy())

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

    @staticmethod
    def _rows_after_reference(state, context):
        # The driver can publish the old and new response with the same header
        # at a ready-to-ready reference switch. Only later rows are unambiguous.
        if state.header.frame_id != context.frame_id:
            return np.zeros(ROWS, dtype=bool)
        return np.asarray(state.row_realtime_ns) > context.reference_stamp_ns

    def load_observation(self, clock_ns):
        """Observe the same native response as the colors, before display saturation."""
        context = self.load_context
        if context is None:
            return None
        frame, valid, state = self._paired_sample(self._responses, clock_ns)
        if frame is None or not valid.all() or not np.isfinite(frame).all():
            return None
        if not self._rows_after_reference(state, context).all():
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

    def _response_with_state(self, clock_ns, *, allow_rgb_fallback=False):
        frame, valid, state = self._paired_sample(self._responses, clock_ns)
        if frame is not None and (valid & np.isfinite(frame)).any():
            self._has_measured_response = True
            return np.where(valid, frame, np.nan), state
        if allow_rgb_fallback and not self._has_measured_response:
            return None, None
        return np.full(SHAPE, np.nan, dtype=np.float32), None

    def contact_response(self, clock_ns, *, allow_rgb_fallback=False):
        """Prefer native measurements for the rest of an acquisition once observed."""
        amplitude, _ = self._response_with_state(clock_ns, allow_rgb_fallback=allow_rgb_fallback)
        return amplitude

    def contact_surface_inputs(self, clock_ns, *, confirmed_only=False):
        """Pair response and thresholds, optionally retaining only measured contact."""
        amplitude, state = self._response_with_state(clock_ns)
        context = self.load_context
        if state is None or context is None:
            return (np.full_like(amplitude, np.nan) if confirmed_only else amplitude), None
        rows = self._rows_after_reference(state, context)
        thresholds = np.asarray(self.reference.activation_threshold_adc).reshape(SHAPE)
        known = rows[:, None] & np.isfinite(amplitude)
        thresholds = np.where(known, thresholds, np.nan)
        if confirmed_only:
            stamp = state.header.stamp.sec * 1_000_000_000 + state.header.stamp.nanosec
            seeds = self.contact_prints.active_taxels_at(stamp) if self._print_context is context else None
            footprint = contact_footprint(amplitude, thresholds, seeds)
            if self._light_context is context and self._light_stamp == stamp:
                footprint |= self._light_mask
            amplitude = np.where(known, np.where(footprint, amplitude, 0.), np.nan)
        return amplitude, thresholds

    def update_contacts(self, clock_ns, profile=None):
        """Feed each new measured scan once, independent of Qt refresh or DDS arrival order."""
        if self.message is not None and (not self.message.connected or not self.message.time_valid):
            # The stream ended. Its finished print is still available for export.
            # accept() interrupted the shutter; a new stream or reference resets it.
            return
        context = self.load_context
        if context is not self._light_context:
            self._reset_light_contact()
            self._light_context = context
        if context is not self._print_context or profile is not self._print_profile:
            self._reset_prints()
            self._print_context, self._print_profile = context, profile
        if context is None or not self._responses:
            return
        source = dict(self._responses)
        reference = self.reference
        valid_now = self.mask(clock_ns) & np.asarray(reference.valid_mask).reshape(SHAPE)
        thresholds = np.asarray(reference.activation_threshold_adc).reshape(SHAPE)
        baseline = np.asarray(reference.baseline_adc).reshape(SHAPE)
        for key, state, received in self._states:
            if key not in source:
                continue
            stamp = int(key[1]) * 1_000_000_000 + int(key[2])
            new_light = self._light_stamp is None or stamp > self._light_stamp
            new_print = self._print_stamp is None or stamp > self._print_stamp
            if not (new_light or new_print):
                continue
            valid = valid_now & self._mask_at(state, received, clock_ns)
            valid &= self._rows_after_reference(state, context)[:, None]
            field = np.where(valid, source[key], np.nan)
            rows = np.asarray(state.row_monotonic_ns)
            if new_light:
                signed = np.asarray(state.adc).reshape(SHAPE) - baseline
                self._light_mask = self._light_tracker.update(
                    stamp, rows, signed, field, thresholds)
                self._light_stamp = stamp
            if not new_print:
                continue
            if not any(state.valid_mask):
                # This observed scan was already empty at acquisition. A later
                # quiet scan cannot turn its missing measurements into a release.
                self.contact_prints.interrupt()
                self._print_stamp, self._print_received = stamp, None
                continue
            valid_rows = valid.any(axis=1)
            # A rolling state can publish the same physical row repeatedly.
            # Wait until every participating row has a new observation.
            if not valid_rows.any() or (rows[valid_rows] <= self._print_rows[valid_rows]).any():
                continue
            self.contact_prints.observe(stamp, field, thresholds,
                                        profile, span_adc=context.span_adc)
            self._print_rows[valid_rows] = rows[valid_rows]
            self._print_stamp, self._print_received = stamp, received
        # A delayed draw may still have a continuous queue of fresh source frames.
        # Only expire the shutter after draining the matched observations.
        if self._print_received is not None and clock_ns - self._print_received > self.message.max_age_ns:
            self.contact_prints.interrupt()
            self._print_received = None

    update_contact_print = update_contacts

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
            self._reset_light_contact()
            self.contact_prints.interrupt()
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
