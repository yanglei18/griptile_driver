import json

import numpy as np
import pytest

from griptile_core.display_response import ReferenceProfile
from griptile_core.equalization import (
    EqualizationProfile, apply_equalization, fit_equalization, measure_uniform_hold, validate_equalization,
)

SHAPE = (20, 20)
SENSITIVITY = np.linspace(0.6, 1.4, 400).reshape(SHAPE)
BASELINE = np.arange(400, dtype=float).reshape(SHAPE) + 18000
THRESHOLD = np.linspace(12, 60, 400).reshape(SHAPE)
REFERENCE = ReferenceProfile(BASELINE.ravel(), THRESHOLD.ravel(), (True,) * 400)


def scans(level, sensitivity=SENSITIVITY, signs=1):
    frame = (BASELINE + np.rint(level * sensitivity) * signs).astype(np.uint16)
    return np.broadcast_to(frame, (12, *SHAPE)).copy()


def loaded(level, direction="increase", signs=1):
    return measure_uniform_hold(scans(level, signs=signs), REFERENCE, direction)


def validated():
    return validate_equalization(fit_equalization(loaded(1200)), loaded(2200))


def test_independent_uniform_load_corrects_response_and_preserves_cv_inputs():
    fit = fit_equalization(loaded(1200))
    second = loaded(2200)
    profile = validate_equalization(fit, second)
    raw = np.asarray(second.signed_response_adc, dtype=np.float32).reshape(SHAPE)
    before = raw.copy()
    corrected = apply_equalization(raw, profile)
    np.testing.assert_array_equal(raw, before)
    assert not np.shares_memory(raw, corrected)
    assert corrected.dtype == np.float32
    assert len(profile.gains) == 400
    assert profile.before_cv > 0.20 and profile.after_cv < 0.01
    assert profile.after_max_error < 0.02
    assert tuple(np.unravel_index(np.abs(corrected / np.median(corrected) - 1).argmax(), SHAPE)) == profile.worst_cell
    assert profile.fit_level == pytest.approx(np.median(np.abs(fit.hold.signed_response_adc)))
    assert profile.check_level == pytest.approx(np.median(raw))
    np.testing.assert_allclose(apply_equalization(np.array(fit.hold.signed_response_adc).reshape(SHAPE), profile),
                               fit.fit_level, rtol=1e-6)
    # Comparing both in the same units leaves the display contour decisions unchanged.
    for multiple in (2, 8):
        np.testing.assert_array_equal(corrected >= multiple * apply_equalization(THRESHOLD, profile),
                                      raw >= multiple * THRESHOLD)
    raw[0, 0], raw[-1, -1] = np.nan, 60000
    out = apply_equalization(raw, profile)
    assert np.isnan(out[0, 0]) and out[-1, -1] > 3000
    assert EqualizationProfile.from_dict(json.loads(json.dumps(profile.to_dict()))) == profile


@pytest.mark.parametrize("direction,signs", [
    ("decrease", -1), ("both", np.where(np.indices(SHAPE)[1] % 2, 1, -1)),
])
def test_direction_is_stable_per_cell_not_assumed_uniform(direction, signs):
    sample = loaded(1200, direction, signs)
    result = validate_equalization(fit_equalization(sample), loaded(2200, direction, signs))
    assert result.direction == direction and result.after_cv < 0.01
    np.testing.assert_array_equal(np.sign(result.check_signed_adc), np.broadcast_to(signs, SHAPE).ravel())


def test_a_flipped_taxel_fails_bidirectional_verification():
    check = scans(2100)
    check[:, 8, 13] = 2 * BASELINE[8, 13] - check[:, 8, 13]
    hold = measure_uniform_hold(check, REFERENCE, "both")
    with pytest.raises(ValueError, match="1/400.*方向.*行 8 列 13"):
        validate_equalization(fit_equalization(loaded(1200, "both")), hold)


@pytest.mark.parametrize("value,reason", [(0, "ADC 边界"), (65535, "ADC 边界"), (BASELINE[3, 9], "响应过弱")])
def test_unreliable_cell_cannot_be_hidden_or_have_gain_invented(value, reason):
    frames = scans(1000)
    frames[:, 3, 9] = value
    with pytest.raises(ValueError, match=f"1/400.*{reason}.*行 3 列 9"):
        measure_uniform_hold(frames, REFERENCE, "increase")


def test_partial_contact_and_incomplete_reference_are_not_uniform_holds():
    local = scans(1500, sensitivity=np.zeros(SHAPE))
    local[:, 10:13, 8:14] += 1500
    with pytest.raises(ValueError, match="382/400.*过弱"):
        measure_uniform_hold(local, REFERENCE, "both")
    mask = np.ones(SHAPE, bool)
    mask[11, 0] = False
    partial = ReferenceProfile(BASELINE.ravel(), THRESHOLD.ravel(), mask.ravel())
    with pytest.raises(ValueError, match="1/400.*行 11 列 0"):
        measure_uniform_hold(scans(1000), partial, "increase")


def test_temporal_noise_clipping_and_global_ramp_rejected():
    noisy = scans(1200)
    noisy[::2, 2, 12] += 370
    with pytest.raises(ValueError, match="1/400.*波动.*行 2 列 12"):
        measure_uniform_hold(noisy, REFERENCE, "increase")
    ramp = np.stack([scans(level)[0] for level in np.linspace(900, 1350, 12)])
    with pytest.raises(ValueError, match="波动|变化"):
        measure_uniform_hold(ramp, REFERENCE, "increase")
    clipped = scans(1200)
    clipped[0, 4, 3] = 65535
    with pytest.raises(ValueError, match="ADC 边界.*行 4 列 3"):
        measure_uniform_hold(clipped, REFERENCE, "increase")


def test_extreme_multiplier_and_load_that_is_too_close_rejected():
    frames = scans(1400)
    frames[:, 0, 2] = int(BASELINE[0, 2]) + 100
    hold = measure_uniform_hold(frames, REFERENCE, "increase")
    with pytest.raises(ValueError, match="1/400.*补偿.*行 0 列 2"):
        fit_equalization(hold)
    with pytest.raises(ValueError, match="25%"):
        validate_equalization(fit_equalization(loaded(1400)), loaded(1500))


def test_one_bad_taxel_fails_even_with_excellent_whole_array_cv():
    fit = fit_equalization(loaded(1000))
    check = scans(2000)
    check[:, 1, 15] += 560
    with pytest.raises(ValueError, match="1/400.*20%.*行 1 列 15"):
        validate_equalization(fit, measure_uniform_hold(check, REFERENCE, "increase"))
    check[:, 1, 15] = scans(1000)[:, 1, 15]
    with pytest.raises(ValueError, match="1/400.*未随载荷升高.*行 1 列 15"):
        validate_equalization(fit, measure_uniform_hold(check, REFERENCE, "increase"))


def test_independent_load_can_reject_worsened_previously_uniform_response():
    first = scans(1200, sensitivity=np.linspace(.95, 1.05, 400).reshape(SHAPE))
    second = scans(2400, sensitivity=np.ones(SHAPE))
    fit = fit_equalization(measure_uniform_hold(first, REFERENCE, "increase"))
    with pytest.raises(ValueError, match="变异系数"):
        validate_equalization(fit, measure_uniform_hold(second, REFERENCE, "increase"))


def test_bad_direction_raw_shapes_and_reference_replacement_rejected():
    with pytest.raises(ValueError, match="方向"):
        measure_uniform_hold(scans(900), REFERENCE, "auto")
    with pytest.raises(ValueError, match="方向"):
        measure_uniform_hold(scans(900), REFERENCE, "decrease")
    with pytest.raises(ValueError, match="至少"):
        measure_uniform_hold(scans(900)[:3], REFERENCE, "both")
    with pytest.raises(ValueError, match="20×20"):
        measure_uniform_hold(scans(900).reshape(12, 400), REFERENCE, "increase")
    changed = ReferenceProfile((BASELINE + 5).ravel(), THRESHOLD.ravel(), (True,) * 400)
    with pytest.raises(ValueError, match="不同的无载参考"):
        validate_equalization(fit_equalization(loaded(1100)), measure_uniform_hold(scans(2000), changed, "increase"))


@pytest.mark.parametrize("bad", [None, True, float("nan"), "1200", 0, 65535, 1e30, 10**400])
def test_restored_profiles_check_measurements_and_rederive_gains(bad):
    doc = validated().to_dict()
    doc["fit_signed_adc"][10] = bad
    with pytest.raises(ValueError):
        EqualizationProfile.from_dict(doc)


def test_profile_schema_and_independent_results_checked_on_load():
    original = validated().to_dict()
    for payload in (dict(original, schema_version=True), dict(original, gains=[1] * 400), {}, []):
        with pytest.raises(ValueError):
            EqualizationProfile.from_dict(payload)
    original["check_signed_adc"][55] *= 1.55
    with pytest.raises(ValueError, match="20%"):
        EqualizationProfile.from_dict(original)
