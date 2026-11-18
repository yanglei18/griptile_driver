"""Two-level spatial equalization for an externally uniform pressure on the 20×20 face.

The target is the median response of the array in the fit hold. Both fitting
and independent validation use the driver's active amplitude (signed excursion
from unloaded, less its release floor). Gains calibrate relative display
response in the tested range; the two external pressures need not be known in
physical units. Uniformity of the applied pressure is a fixture assumption.
"""
from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from .display_response import RELEASE_RATIO, ReferenceProfile
from .grid import CELL_COUNT, SHAPE


MIN_HOLD_FRAMES = 10
MIN_GAIN, MAX_GAIN = 0.25, 4.0
MIN_LEVEL_RATIO = 1.25
MAX_CHECK_CV = 0.10
MAX_CELL_ERROR = 0.20


def _require_direction(direction):
    if direction not in ("both", "increase", "decrease"):
        raise ValueError("无效的接触响应方向")


def _reject_cells(mask, message):
    count = np.count_nonzero(mask)
    if count:
        row, col = np.unravel_index(int(np.argmax(mask)), SHAPE)
        raise ValueError(f"{count}/400 点{message}（例如行 {row} 列 {col}）")


@dataclass(frozen=True)
class UniformHold:
    reference: ReferenceProfile
    direction: str
    signed_response_adc: tuple


def measure_uniform_hold(raw_frames, reference, direction):
    """Summarize independent complete raw scans from one stable, loaded hold.

    The collector supplies fresh, timestamp-distinct scans within one source
    session and one ready reference. Signed amplitudes retain polarity for the
    independent check even when the live contact direction accepts either sign.
    """
    _require_direction(direction)
    if not isinstance(reference, ReferenceProfile):
        raise ValueError("等待就绪的无载参考")
    _reject_cells(~np.asarray(reference.valid_mask), "缺少无载参考")
    raw = np.asarray(raw_frames)
    if raw.ndim != 3 or raw.shape[1:] != SHAPE or len(raw) < MIN_HOLD_FRAMES:
        raise ValueError(f"每档需要至少 {MIN_HOLD_FRAMES} 帧完整的 20×20 读数")
    if raw.dtype.kind not in "iuf" or not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 65535):
        raise ValueError("采集包含无效的原始 ADC 读数")
    _reject_cells(((raw == 0) | (raw == 65535)).any(axis=0), "到达 ADC 边界，请调整载荷后重采")
    threshold = np.asarray(reference.activation_threshold_adc).reshape(SHAPE)
    delta = raw.astype(np.float64) - np.asarray(reference.baseline_adc).reshape(SHAPE)
    if direction == "both":
        polarity = np.sign(np.median(delta, axis=0))
    else:
        polarity = 1 if direction == "increase" else -1
    low, center, high = np.percentile(delta * polarity, (10, 50, 90), axis=0)
    _reject_cells((center < 2 * threshold) | (low <= threshold),
                  "响应过弱或方向不符，请检查整面均匀受压、设备方向和无载参考")
    # Allow baseline-scale jitter locally. A coherent whole-face ramp is
    # checked separately so it cannot hide inside the per-cell noise allowance.
    _reject_cells(high - low > np.maximum(0.15 * center, threshold),
                  "在保持期间波动过大，请稳住当前载荷后重采")
    frame_levels = np.median(delta * polarity, axis=(1, 2))
    lo_level, hi_level = np.percentile(frame_levels, (10, 90))
    if hi_level - lo_level > 0.10 * np.median(frame_levels):
        raise ValueError("整面载荷在变化，请稳住当前载荷后重采")
    response = (center - RELEASE_RATIO * threshold) * polarity
    return UniformHold(reference, direction, tuple(response.ravel()))


def _fit_response(signed):
    magnitude = np.abs(signed)
    level = float(np.median(magnitude))
    gains = level / magnitude
    _reject_cells((gains < MIN_GAIN) | (gains > MAX_GAIN),
                  "所需补偿超出 0.25–4 倍，请检查均压覆盖、安装或异常触点")
    return level, tuple(gains)


@dataclass(frozen=True)
class EqualizationFit:
    hold: UniformHold
    fit_level: float
    gains: tuple


def fit_equalization(hold):
    """Fit the low level; a candidate becomes usable only after validation."""
    level, gains = _fit_response(_signed_vector(hold.signed_response_adc, hold.direction))
    return EqualizationFit(hold, level, gains)


def _signed_vector(values, direction):
    if not isinstance(values, (list, tuple, np.ndarray)) or np.shape(values) != (CELL_COUNT,):
        raise ValueError("均衡档案需要每个阵元对应的 400 个有符号响应")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in values):
        raise ValueError("阵元响应必须为有限数值")
    if any(not -65535 < v < 65535 or v == 0 for v in values):
        raise ValueError("阵元响应超出有效范围")
    vec = np.asarray(values, dtype=np.float64)
    if (direction == "increase" and (vec < 0).any()) or (direction == "decrease" and (vec > 0).any()):
        raise ValueError("阵元响应与设定的受压方向不符")
    return vec


@dataclass(frozen=True)
class EqualizationProfile:
    """A fully checked profile. Restore derives gains and rechecks both recorded holds."""

    direction: str
    fit_signed_adc: tuple
    check_signed_adc: tuple
    gains: tuple = field(init=False, repr=False)
    fit_level: float = field(init=False)
    check_level: float = field(init=False)
    before_cv: float = field(init=False)
    after_cv: float = field(init=False)
    after_max_error: float = field(init=False)
    worst_cell: tuple = field(init=False)

    def __post_init__(self):
        _require_direction(self.direction)
        fit = _signed_vector(self.fit_signed_adc, self.direction)
        check = _signed_vector(self.check_signed_adc, self.direction)
        level, gains = _fit_response(fit)
        _reject_cells(np.sign(fit) != np.sign(check), "两档受压的响应方向不同，请检查均压和安装")
        raw = np.abs(check)
        check_level = float(np.median(raw))
        if check_level < MIN_LEVEL_RATIO * level:
            raise ValueError("两档响应过近，请提高第二档均匀载荷，使中位响应至少增加 25%")
        _reject_cells(raw <= np.abs(fit), "在第二档未随载荷升高，请检查局部饱和或接触变化")
        corrected = raw * gains
        before_cv = float(raw.std() / raw.mean())
        after_cv = float(corrected.std() / corrected.mean())
        errors = np.abs(corrected / np.median(corrected) - 1)
        _reject_cells(errors > MAX_CELL_ERROR, "在独立验证中与中位响应偏差超过 20%")
        # CV is the population standard deviation divided by the mean. Allow
        # half a percentage point of repeat noise on an already uniform face.
        if after_cv > MAX_CHECK_CV or after_cv > before_cv + 0.005:
            raise ValueError(f"验证均匀性未达标：变异系数校正前 {before_cv:.1%}，校正后 {after_cv:.1%}")
        object.__setattr__(self, "fit_signed_adc", tuple(fit))
        object.__setattr__(self, "check_signed_adc", tuple(check))
        object.__setattr__(self, "gains", gains)
        object.__setattr__(self, "fit_level", level)
        object.__setattr__(self, "check_level", check_level)
        object.__setattr__(self, "before_cv", before_cv)
        object.__setattr__(self, "after_cv", after_cv)
        object.__setattr__(self, "after_max_error", float(errors.max()))
        object.__setattr__(self, "worst_cell", tuple(int(i) for i in np.unravel_index(np.argmax(errors), SHAPE)))

    def to_dict(self):
        return {"schema_version": 1, "direction": self.direction, "fit_signed_adc": list(self.fit_signed_adc),
                "check_signed_adc": list(self.check_signed_adc)}

    @classmethod
    def from_dict(cls, doc):
        keys = {"schema_version", "direction", "fit_signed_adc", "check_signed_adc"}
        if (not isinstance(doc, dict) or doc.keys() != keys or type(doc["schema_version"]) is not int
                or doc["schema_version"] != 1):
            raise ValueError("无法读取该版本的阵元均衡档案")
        return cls(doc["direction"], doc["fit_signed_adc"], doc["check_signed_adc"])


def validate_equalization(fit, second_hold):
    if fit.hold.reference != second_hold.reference or fit.hold.direction != second_hold.direction:
        raise ValueError("两档使用了不同的无载参考或响应方向，请重新开始均衡")
    return EqualizationProfile(fit.hold.direction, fit.hold.signed_response_adc, second_hold.signed_response_adc)


def apply_equalization(field, profile):
    """Scale native amplitudes or noise thresholds; preserve missing cells and above-range values."""
    values = np.asarray(field, dtype=np.float32)
    if values.shape != SHAPE:
        raise ValueError("均衡输入必须为 20×20")
    return values * np.asarray(profile.gains, dtype=np.float32).reshape(SHAPE)
