"""Source-timed evidence for light contact on the operator's map."""

from collections import deque

import numpy as np

from .contact_image import CONTACT_SEED_RATIO
from .grid import ROWS, SHAPE


_MAX_GAP_NS = 100_000_000
# Compare against earlier measurements, clear of both onset observations.
_HISTORY_MIN_NS = 85_000_000
_HISTORY_MAX_NS = 215_000_000
_REBOUND_NS = 225_000_000
# All amplitudes below are in multiples of each taxel's measured noise threshold.
_ISOLATED_RISE = 3.
_ADJACENT_RISE = 1.25
_HOLD_RISE = 1.
_HOLD_RESPONSE = .75


def _neighbor_latest(values):
    """Maximum among the four edge-adjacent taxels."""
    neighbors = np.full(SHAPE, -1, np.int64)
    neighbors[1:] = values[:-1]
    neighbors[:-1] = np.maximum(neighbors[:-1], values[1:])
    neighbors[:, 1:] = np.maximum(neighbors[:, 1:], values[:, :-1])
    neighbors[:, :-1] = np.maximum(neighbors[:, :-1], values[:, 1:])
    return neighbors


def _nearby_latest(stamps):
    # A strong contact can release into adjacent taxels as the film recovers.
    padded = np.pad(stamps, 2, constant_values=-1)
    return np.lib.stride_tricks.sliding_window_view(padded, (5, 5)).max(axis=(-2, -1))


class LightContactTracker:
    """Track changes relative to a local pre-touch level, without changing the zero.

    Supply one physical acquisition and fixed reference per instance. Row timestamps
    must share a monotonic timeline; the enclosing frame time may use another epoch.
    Returned masks can be retained and are immutable.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Forget motion evidence across a stream or reference discontinuity."""
        self._stamp = None
        self._rows = np.full(ROWS, -1, np.int64)
        self._since = np.full(ROWS, -1, np.int64)
        self._history = deque(maxlen=128)
        self._last_rise = np.full(SHAPE, -np.inf)
        self._repeat_stamp = np.full(SHAPE, -1, np.int64)
        self._positive = np.full(SHAPE, -1, np.int64)
        self._negative = self._positive.copy()
        self._anchor = np.zeros(SHAPE)
        self._sign = np.zeros(SHAPE)
        self._active = np.zeros(SHAPE, dtype=bool)

    def update(self, stamp_ns, row_stamps, signed_delta, amplitude, thresholds):
        """Return additional weak-contact taxels with fresh supporting measurements."""
        rows = np.asarray(row_stamps, dtype=np.int64)
        signed, response, noise = (np.asarray(x, dtype=float)
                                   for x in (signed_delta, amplitude, thresholds))
        if rows.shape != (ROWS,) or any(x.shape != SHAPE for x in (signed, response, noise)):
            raise ValueError("light contact requires 20 row timestamps and three 20x20 fields")
        if self._stamp is not None and stamp_ns < self._stamp:
            return np.frombuffer(bytes(self._active.size), dtype=bool).reshape(SHAPE)
        if self._stamp is not None and stamp_ns - self._stamp > _MAX_GAP_NS:
            self.reset()
        self._stamp = stamp_ns
        valid = np.isfinite(signed) & np.isfinite(response) & np.isfinite(noise) & (noise > 0)
        self._active &= valid
        self._last_rise[~valid] = -np.inf
        self._repeat_stamp[~valid] = -1
        fresh = (rows > self._rows) & (rows >= 0) & valid.any(axis=1)
        if fresh.any():
            gap = fresh & ((self._rows < 0) | (rows - self._rows > _MAX_GAP_NS))
            self._active[gap] = False
            self._last_rise[gap] = -np.inf
            self._repeat_stamp[gap] = -1
            self._since[gap] = rows[gap]
            self._rows[fresh] = rows[fresh]
            signed = np.divide(signed, noise, out=np.full(SHAPE, np.nan), where=valid)
            response = np.divide(response, noise, out=np.zeros(SHAPE), where=valid)
            self._observe(stamp_ns, rows, fresh, signed, response)
        output = self._active & (rows == self._rows)[:, None]
        return np.frombuffer(output.tobytes(), dtype=bool).reshape(SHAPE)

    def _observe(self, stamp, rows, fresh, signed, response):
        physical_rows = rows[:, None]
        measured = fresh[:, None] & np.isfinite(signed)
        magnitude = np.abs(signed)
        # Frame time bounds storage; only distinct physical readings vote per row.
        while self._history and self._history[0][0] < stamp - _HISTORY_MAX_NS - _MAX_GAP_NS:
            self._history.popleft()
        reference = np.full(SHAPE, np.nan)
        if self._history:
            stamps = np.stack([h[1] for h in self._history])
            age = rows - stamps
            eligible = ((age >= _HISTORY_MIN_NS) & (age <= _HISTORY_MAX_NS)
                        & (stamps >= self._since) & fresh)
            if eligible.any():
                values = np.stack([h[2] for h in self._history])
                history = np.ma.masked_array(values, mask=~(eligible[:, :, None] & np.isfinite(values)))
                reference = np.ma.median(history, axis=0).filled(np.nan)
                reference[history.count(axis=0) < 3] = np.nan
        rise = np.where(measured & np.isfinite(reference), magnitude - reference, -np.inf)
        repeated = np.minimum(rise, self._last_rise)
        adjacent_rise = repeated >= _ADJACENT_RISE
        # A neighboring row can arrive in another frame. Its vote must have
        # come from two different readings of its own taxel and still be recent.
        self._repeat_stamp[fresh] = np.where(adjacent_rise, physical_rows, -1)[fresh]
        neighbor_stamp = _neighbor_latest(self._repeat_stamp)
        corroborated = (neighbor_stamp >= 0) & (np.abs(physical_rows - neighbor_stamp) <= _MAX_GAP_NS)
        candidates = (repeated >= _ISOLATED_RISE) | (adjacent_rise & corroborated)
        strong = measured & (response >= CONTACT_SEED_RATIO)
        self._positive = np.where(strong & (signed > 0), physical_rows, self._positive)
        self._negative = np.where(strong & (signed < 0), physical_rows, self._negative)
        opposite = np.where(signed > 0, _nearby_latest(self._negative), _nearby_latest(self._positive))
        rebound = (opposite >= 0) & (physical_rows - opposite <= _REBOUND_NS)
        start = measured & ~self._active & candidates & (response >= .5) & ~rebound
        self._anchor[start] = reference[start]
        self._sign[start] = np.sign(signed[start])
        held = (self._sign * signed - self._anchor >= _HOLD_RISE) & (response >= _HOLD_RESPONSE)
        self._active[fresh] = (start | (self._active & held & ~rebound))[fresh]
        self._last_rise[fresh] = rise[fresh]
        self._history.append((stamp, np.where(fresh, rows, -1), np.where(measured, magnitude, np.nan)))
