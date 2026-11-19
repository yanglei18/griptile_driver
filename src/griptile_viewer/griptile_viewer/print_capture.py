"""Retain three source-timed tactile exposures and capture one print per release."""

from collections import deque
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from griptile_core.contact_image import CONTACT_SEED_RATIO, CONTACT_SUPPORT_RATIO
from griptile_core.equalization import apply_equalization
from griptile_core.grid import SHAPE


@dataclass(frozen=True, eq=False)
class ContactPrint:
    """Three retained display-response grids in chronological order, with real source stamps."""

    stamps_ns: tuple
    frames: tuple
    automatic: bool
    equalized: bool


@dataclass(frozen=True, eq=False)
class _Sample:
    stamp_ns: int
    field: np.ndarray
    # Sum of retained, noise-relative per-taxel evidence before equalization.
    # Missing evidence is deliberately different from an observed absence of strong contact.
    strength: float | None


class ContactPrintCapture:
    STEP_NS = 300_000_000
    MAX_GAP_NS = 150_000_000
    ARM_NS = 100_000_000
    # Wait one exposure interval after the strong response disappears. Even a
    # short press can then contribute the middle exposure of a genuine release.
    RELEASE_NS = STEP_NS
    # Look back one triplet before release for a gradual descending edge.
    LOOKBACK_NS = 2 * STEP_NS
    HISTORY_NS = 2 * STEP_NS + LOOKBACK_NS + RELEASE_NS + 2 * MAX_GAP_NS

    def __init__(self):
        self._samples = deque(maxlen=2048)
        self.reset()

    def _clear_event(self):
        self._pressed_since = None
        self._quiet_since = None
        self._armed = False
        self._active_taxels = np.zeros(SHAPE, dtype=bool)

    def interrupt(self):
        """Drop in-flight evidence on a gap; keep already taken prints and the time watermark."""
        self._samples.clear()
        self._automatic_since = None
        self._clear_event()

    def reset(self):
        """Start a different source/reference/display profile, including its local print collection."""
        self.interrupt()
        self.latest = None
        self._profile = None
        self._last_stamp = None

    def observe(self, stamp_ns, measured, thresholds, profile=None):
        """Accept one matched source frame. The caller owns stream and zero-reference identity."""
        if isinstance(stamp_ns, bool) or not isinstance(stamp_ns, Integral):
            raise ValueError("触印需要整数源时间戳")
        stamp_ns = int(stamp_ns)
        if self._last_stamp is not None and stamp_ns <= self._last_stamp:
            return
        field = np.asarray(measured, dtype=np.float32)
        thresholds = np.asarray(thresholds)
        if field.shape != SHAPE or thresholds.shape != SHAPE:
            raise ValueError("触印需要 20×20 响应与逐点门限")
        if self._profile is not profile:
            self.reset()
            self._profile = profile
        if self._last_stamp is not None and stamp_ns - self._last_stamp > self.MAX_GAP_NS:
            self.interrupt()
        self._last_stamp = stamp_ns

        valid = np.isfinite(field) & (field >= 0) & np.isfinite(thresholds) & (thresholds > 0)
        if not valid.any():
            self.interrupt()
            return
        strength = None
        if valid.all():
            relative = field / thresholds
            # Re-use the contour thresholds over time at each measured location.
            # A lighter hold stays active until its original taxel falls below
            # the support floor. Unrelated background cannot keep the event open.
            self._active_taxels = ((relative >= CONTACT_SEED_RATIO)
                                   | (self._active_taxels & (relative >= CONTACT_SUPPORT_RATIO)))
            strength = float(np.sum(relative, where=self._active_taxels, dtype=np.float64))
        else:
            field = np.where(valid, field, np.nan)
        if profile is not None:
            field = apply_equalization(field, profile)
        # Bytes-backed storage cannot be mutated by the subscriber or presentation code.
        stored = np.frombuffer(field.tobytes(), dtype=np.float32).reshape(SHAPE)
        self._samples.append(_Sample(stamp_ns, stored, strength))
        oldest = stamp_ns - self.HISTORY_NS
        while self._samples[0].stamp_ns < oldest:
            self._samples.popleft()
        self._advance_event(stamp_ns, strength)

    def _triplet(self, end):
        """Choose actual observations nearest both previous exposure times."""
        middle = end.stamp_ns - self.STEP_NS
        first = middle - self.STEP_NS
        if self._samples[0].stamp_ns > first:
            return None
        first_sample = min(self._samples, key=lambda s: abs(s.stamp_ns - first))
        middle_sample = min(self._samples, key=lambda s: abs(s.stamp_ns - middle))
        return (first_sample, middle_sample, end)

    def _manual_triplet(self):
        if self._samples:
            triplet = self._triplet(self._samples[-1])
            if triplet and np.logical_and.reduce([np.isfinite(s.field) for s in triplet]).any():
                return triplet
        return None

    @property
    def can_snapshot(self):
        return self._manual_triplet() is not None

    def _take(self, triplet, *, automatic):
        self.latest = ContactPrint(tuple(s.stamp_ns for s in triplet), tuple(s.field for s in triplet),
                                   automatic, self._profile is not None)
        return self.latest

    def snapshot(self):
        """Capture the most recent three exposures on operator request."""
        triplet = self._manual_triplet()
        return self._take(triplet, automatic=False) if triplet else None

    def _advance_event(self, stamp_ns, strength):
        if strength is None:
            self._clear_event()
            self._automatic_since = None
            return
        if self._automatic_since is None:
            self._automatic_since = stamp_ns
        if strength > 0:
            self._quiet_since = None
            if self._pressed_since is None:
                self._pressed_since = stamp_ns
            if stamp_ns - self._pressed_since >= self.ARM_NS:
                self._armed = True
            return
        self._pressed_since = None
        if not self._armed:
            return
        if self._quiet_since is None:
            self._quiet_since = stamp_ns
        elif stamp_ns - self._quiet_since >= self.RELEASE_NS:
            self._take_release()
            self._automatic_since = self._quiet_since
            self._clear_event()

    def _take_release(self):
        # Rank only complete triplets close to this release. Maximize the loss
        # of measured contact evidence from the two past exposures to the last.
        # Early ties keep the actual edge instead of a later quiet frame.
        best = None
        best_drop = 0.0
        earliest = self._quiet_since - self.LOOKBACK_NS
        for end in self._samples:
            if end.stamp_ns < earliest:
                continue
            triplet = self._triplet(end)
            if not triplet or triplet[0].stamp_ns < self._automatic_since:
                continue
            first, middle, _ = triplet
            drop = (first.strength + middle.strength) / 2 - end.strength
            if drop > best_drop:
                best, best_drop = triplet, drop
        if best:
            self._take(best, automatic=True)
