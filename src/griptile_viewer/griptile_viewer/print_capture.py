"""Retain three source-timed tactile exposures and capture one print per release."""

from collections import deque
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from griptile_core.contact_image import (
    CONTACT_SEED_RATIO, CONTACT_SUPPORT_RATIO, DEFAULT_CONTACT_SPAN, contact_footprint,
)
from griptile_core.equalization import apply_equalization
from griptile_core.grid import SHAPE


@dataclass(frozen=True, eq=False)
class ContactPrint:
    """Source-timed response maps and footprints. A saved print has three exposures."""

    stamps_ns: tuple
    frames: tuple
    automatic: bool
    equalized: bool
    span_adc: float = DEFAULT_CONTACT_SPAN
    footprints: tuple | None = None


@dataclass(frozen=True, eq=False)
class _Sample:
    stamp_ns: int
    field: np.ndarray
    footprint: np.ndarray
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

    def interrupt(self):
        """Drop in-flight evidence on a gap; keep already taken prints and the time watermark."""
        self._samples.clear()
        self._automatic_since = None
        self._active_taxels = np.zeros(SHAPE, dtype=bool)
        self.contact_present = None  # False requires a complete measured absence of contact evidence.
        self.contact_id = None       # A render from an earlier contact cannot share this token.
        self._preview_since = None
        self._preview = None
        self._clear_event()

    def reset(self):
        """Start a different source/reference/display profile, including its local print collection."""
        self.interrupt()
        self.latest = None
        self._profile = None
        self._span_adc = DEFAULT_CONTACT_SPAN
        self._last_stamp = None

    def observe(self, stamp_ns, measured, thresholds, profile=None, *, span_adc=DEFAULT_CONTACT_SPAN):
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
        if self._profile is not profile or self._span_adc != span_adc:
            if not np.isfinite(span_adc) or span_adc <= 0:
                raise ValueError("触印色标量程必须为有限正数")
            self.reset()
            self._profile = profile
            self._span_adc = float(span_adc)
        if self._last_stamp is not None and stamp_ns - self._last_stamp > self.MAX_GAP_NS:
            self.interrupt()
        self._last_stamp = stamp_ns

        valid = np.isfinite(field) & (field >= 0) & np.isfinite(thresholds) & (thresholds > 0)
        if not valid.any():
            self.interrupt()
            return
        complete = bool(valid.all())
        relative = np.divide(field, thresholds, out=np.zeros(SHAPE), where=valid)
        seeded = relative >= CONTACT_SEED_RATIO
        # The same measured taxel can follow a lighter hold all the way down
        # to the support floor. A different weak background taxel cannot.
        self._active_taxels = seeded | (self._active_taxels & (relative >= CONTACT_SUPPORT_RATIO))
        present = bool(self._active_taxels.any())
        if present and self.contact_id is None:
            self.contact_id = object()
        elif not present and self.contact_id is not None:
            self.contact_id = None
            self._preview_since = stamp_ns
        self.contact_present = present if complete or present else None
        self._preview = None
        strength = None
        if complete:
            strength = float(np.sum(relative, where=self._active_taxels, dtype=np.float64))
        else:
            field = np.where(valid, field, np.nan)
        # Determine the shape in sensor units. Display gains may change its
        # colors; they cannot add or remove contact from an archived exposure.
        footprint = contact_footprint(field, thresholds, self._active_taxels)
        if profile is not None:
            field = apply_equalization(field, profile)
        # Bytes-backed storage cannot be mutated by the subscriber or presentation code.
        stored = np.frombuffer(field.tobytes(), dtype=np.float32).reshape(SHAPE)
        stored_footprint = np.frombuffer(footprint.tobytes(), dtype=np.bool_).reshape(SHAPE)
        self._samples.append(_Sample(stamp_ns, stored, stored_footprint, strength))
        oldest = stamp_ns - self.HISTORY_NS
        while self._samples[0].stamp_ns < oldest:
            self._samples.popleft()
        self._advance_event(stamp_ns, strength, bool(seeded.any()))

    def _triplet(self, end):
        """Choose actual observations nearest both previous exposure times."""
        middle = end.stamp_ns - self.STEP_NS
        first = middle - self.STEP_NS
        if self._samples[0].stamp_ns > first:
            return None
        first_sample = min(self._samples, key=lambda s: abs(s.stamp_ns - first))
        middle_sample = min(self._samples, key=lambda s: abs(s.stamp_ns - middle))
        return (first_sample, middle_sample, end)

    def live_preview(self):
        """Show the current contact face as soon as it is measured; add history when available."""
        if self.contact_present is not True:
            return None
        if self._preview is None:
            current = self._samples[-1]
            exposures = (current,)
            triplet = self._triplet(current)
            if triplet and (self._preview_since is None or triplet[0].stamp_ns >= self._preview_since):
                supported = np.logical_and.reduce([np.isfinite(s.field) for s in triplet])
                if (supported & self._active_taxels).any():
                    exposures = triplet
            self._preview = self._print(exposures, automatic=False)
        return self._preview

    def preview(self):
        """Return the complete, time-registered print currently available to the shutter."""
        record = self.live_preview()
        return record if record is not None and len(record.frames) == 3 else None

    @property
    def can_snapshot(self):
        return self.preview() is not None

    def _print(self, exposures, *, automatic):
        return ContactPrint(tuple(s.stamp_ns for s in exposures), tuple(s.field for s in exposures),
                            automatic, self._profile is not None, self._span_adc,
                            footprints=tuple(s.footprint for s in exposures))

    def snapshot(self):
        """Capture the most recent three exposures on operator request."""
        record = self.preview()
        if record is not None:
            self.latest = record
        return record

    def _advance_event(self, stamp_ns, strength, seeded):
        if strength is None:
            self._clear_event()
            self._automatic_since = None
            return
        if self._automatic_since is None:
            self._automatic_since = stamp_ns
        if strength > 0:
            self._quiet_since = None
            if self._pressed_since is None and seeded:
                self._pressed_since = stamp_ns
            if self._pressed_since is not None and stamp_ns - self._pressed_since >= self.ARM_NS:
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
            self.latest = self._print(best, automatic=True)
