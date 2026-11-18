"""Session-bound observations and short holds for the known-load bench test."""

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from statistics import median

import numpy as np

from griptile_core.grid import LAYOUT_ID


@dataclass(frozen=True)
class LoadContext:
    sensor_id: str
    session_id: str
    stream_epoch: int
    hardware_id: str
    layout_id: str
    direction: str
    span_adc: float
    frame_id: str
    reference_stamp_ns: int
    reference_digest: str
    generation: int

    @classmethod
    def from_reference(cls, status, profile, generation):
        header = getattr(status, "header", None)
        direction = getattr(status, "direction", None)
        if profile is None or header is None or direction not in ("both", "increase", "decrease"):
            return None
        if not header.frame_id or not status.session_id:
            return None
        stamp = int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)
        if stamp <= 0:
            return None
        digest = sha256()
        for values, dtype in ((profile.baseline_adc, "<f8"), (profile.activation_threshold_adc, "<f8"),
                              (profile.valid_mask, "u1")):
            digest.update(np.asarray(values, dtype=dtype).tobytes())
        return cls(status.sensor_id, status.session_id, int(status.stream_epoch), status.hardware_id,
                   LAYOUT_ID, direction, float(status.span_adc), header.frame_id, stamp,
                   digest.hexdigest(), generation)


@dataclass(frozen=True)
class LoadObservation:
    context: LoadContext
    row_stamps: tuple
    source_now_ns: int   # Current source clock estimated at this observation, used only as an arm fence.
    total_adc: float
    signed_total_adc: float
    peak_adc: float
    active_cells: int


@dataclass(frozen=True)
class HeldLoad:
    total_adc: float
    signed_total_adc: float
    peak_adc: float
    active_cells: int
    samples: int
    duration_s: float
    spread_adc: float
    signed_spread_adc: float
    stable: bool
    first_row_monotonic_ns: int
    last_row_monotonic_ns: int


class HoldCapture:
    """Capture independent complete scans spanning a short hold.

    Absolute ADC spread is relevant near zero; at larger loads use fractional spread.
    These checks describe within-hold movement, not force accuracy. Only a fitted
    anchor needs a stationary interval. Independent checks retain the first window,
    including release chatter, so their measurements are not selected for stability.
    """

    WINDOW_NS = 400_000_000
    MAX_SCAN_GAP_NS = 1_000_000_000
    TIMEOUT_NS = 10_000_000_000

    def __init__(self, initial, now_ns, *, require_stable=True):
        self.require_stable = require_stable
        self.context = initial.context
        self.start_ns = now_ns
        self._after_ns = initial.source_now_ns
        self._samples = deque()
        self._last_received_ns = now_ns
        self.progress_text = "保持当前载荷，等待新的完整阵列读数"

    def update(self, observation, now_ns):
        if observation is not None and observation.context != self.context:
            raise ValueError("设备或无载参考已更换，请重新开始载荷标定")
        if now_ns - self.start_ns >= self.TIMEOUT_NS:
            raise ValueError(f"{self.progress_text}。本次采集超时，请重试此步")
        if observation is None:
            self._samples.clear()
            self.progress_text = "等待 400 个测点的有效读数"
            return None

        # A published snapshot may contain mostly old rows. The last contribution
        # to one accepted scan must precede every row of the next accepted scan.
        if min(observation.row_stamps) <= self._after_ns:
            return None
        end_ns = max(observation.row_stamps)
        self._after_ns = end_ns
        if self._samples and (end_ns - self._samples[-1][0] > self.MAX_SCAN_GAP_NS
                              or now_ns - self._last_received_ns > self.MAX_SCAN_GAP_NS):
            # A suspended GUI may never have polled during the missing interval.
            self._samples.clear()
        self._last_received_ns = now_ns
        self._samples.append((end_ns, observation))
        while len(self._samples) > 3 and self._samples[1][0] <= end_ns - self.WINDOW_NS:
            self._samples.popleft()
        duration = end_ns - self._samples[0][0]
        if len(self._samples) < 3 or duration < self.WINDOW_NS:
            self.progress_text = f"已采集 {len(self._samples)} 帧，请保持当前载荷"
            return None

        values = [sample for _, sample in self._samples]
        totals = [sample.total_adc for sample in values]
        signed = [sample.signed_total_adc for sample in values]
        center = median(totals)
        spread = max(totals) - min(totals)
        signed_spread = max(signed) - min(signed)
        stable = max(spread, signed_spread) <= max(20.0, center * 0.10)
        if self.require_stable and not stable:
            self.progress_text = "保持同一位置和载荷，等待读数稳定"
            return None
        return HeldLoad(center, median(signed), median(s.peak_adc for s in values),
                        round(median(s.active_cells for s in values)), len(values),
                        duration / 1e9, spread, signed_spread, stable,
                        min(values[0].row_stamps), end_ns)
