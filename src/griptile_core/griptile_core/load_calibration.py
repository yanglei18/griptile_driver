"""Single-point calibration of the summed, noise-gated native tactile response.

An existing unloaded reference supplies the origin. The two readings at the
intermediate load and the final zero-load reading are independent checks.
"""

from math import isclose, isfinite
from numbers import Real


_STAGES = ("check_up", "calibration", "check_down", "release")


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
        raise ValueError(f"{name}必须是有限的非负数")
    return float(value)


def _dominant_sign(total_adc, signed_adc):
    total = _nonnegative(total_adc, "触觉总响应")
    if isinstance(signed_adc, bool) or not isinstance(signed_adc, Real) or not isfinite(signed_adc):
        raise ValueError("有符号响应必须是有限数")
    magnitude = abs(signed_adc)
    if magnitude > total and not isclose(magnitude, total, rel_tol=1e-12):
        raise ValueError("有符号响应的幅值超出触觉总响应")
    if total == 0 or magnitude < 0.5 * total:
        return 0
    return 1 if signed_adc > 0 else -1


def relative_load_polarity(anchor_total_adc, anchor_signed_adc, total_adc, signed_adc):
    """Describe the sign of the measured response relative to the loaded anchor.

    With unsigned total P+N and signed total P-N, |P-N| >= (P+N)/2
    means at least three quarters of the response has the same sign. A
    mixed-sign anchor or hold gives no directional conclusion. These labels
    are descriptive and independent of the calculation of force and error.
    """
    anchor_sign = _dominant_sign(anchor_total_adc, anchor_signed_adc)
    hold_sign = _dominant_sign(total_adc, signed_adc)
    if total_adc == 0:
        return "no_response"
    if not hold_sign or not anchor_sign:
        return "mixed"
    return "same" if anchor_sign == hold_sign else "opposite"


class LoadTrial:
    """Fit and check a local estimate of total normal force on one sensor face.

    Each submitted value is the representative sum of the native 20×20 ADC
    response during one hold, before RGB clipping or spatial interpolation.
    """

    def __init__(self, check_n, calibration_n):
        self._check_n = _nonnegative(check_n, "验证载荷")
        self._calibration_n = _nonnegative(calibration_n, "标定载荷")
        if not 0 < self._check_n < self._calibration_n:
            raise ValueError("验证载荷必须大于零且小于标定载荷")
        self._totals = {}

    @property
    def check_n(self):
        return self._check_n

    @property
    def calibration_n(self):
        return self._calibration_n

    @property
    def stage(self):
        return _STAGES[len(self._totals)] if len(self._totals) < len(_STAGES) else "done"

    @property
    def scale_n_per_adc(self):
        anchor = self._totals.get("calibration")
        return None if anchor is None else self.calibration_n / anchor

    def record(self, total_adc):
        stage = self.stage
        if stage == "done":
            raise ValueError("本轮载荷验证已完成")
        total = _nonnegative(total_adc, "触觉总响应")
        if stage == "calibration":
            if not total:
                raise ValueError("标定载荷下没有触觉响应，请检查接触后重新采集这一步")
            gain = self.calibration_n / total
            if not isfinite(gain) or gain == 0:
                raise ValueError("载荷与触觉响应超出数值范围")
        self._totals[stage] = total

    def estimate(self, total_adc):
        """Return N, or None before calibration. Values above the anchor extrapolate."""
        total = _nonnegative(total_adc, "触觉总响应")
        gain = self.scale_n_per_adc
        if gain is None:
            return None
        force = total * gain
        if not isfinite(force):
            raise ValueError("估计载荷超出数值范围")
        return force

    def verify(self, known_n, total_adc):
        """Evaluate another known load without altering the fit or original cycle."""
        if self.stage != "done":
            raise ValueError("请先完成加载、卸载和松开采集")
        known_n = _nonnegative(known_n, "验证载荷")
        if known_n > self.calibration_n:
            raise ValueError("验证载荷超出本轮标定范围")
        total = _nonnegative(total_adc, "触觉总响应")
        estimate = self.estimate(total)
        error = estimate - known_n
        return {
            "total_adc": total,
            "known_n": known_n,
            "estimated_n": estimate,
            "error_n": error,
            "relative_error_pct": error / known_n * 100 if known_n else None,
            "in_calibrated_range": total <= self._totals["calibration"],
        }

    def report(self):
        """Return independent loading, unloading and released-hold measurements."""
        if self.stage != "done":
            raise ValueError("请先完成加载、卸载和松开采集")
        anchor = self._totals["calibration"]
        known = (self.check_n, self.calibration_n, self.check_n, 0.0)
        holds = {stage: self.verify(known_n, self._totals[stage]) for stage, known_n in zip(_STAGES, known)}
        return {
            "schema_version": 1,
            "model": "single_point_through_unloaded_origin",
            "scope": "estimated total normal force on one sensor face with the same mounting and contact geometry",
            "check_n": self.check_n,
            "calibration_n": self.calibration_n,
            "scale_n_per_adc": self.scale_n_per_adc,
            "calibrated_range_adc": [0.0, anchor],
            "holds": holds,
            "unloading_minus_loading_n": holds["check_down"]["estimated_n"] - holds["check_up"]["estimated_n"],
            "release_residual_n": holds["release"]["estimated_n"],
        }
