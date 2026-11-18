"""Independent force checks use the same unclipped response as the RGB surface."""

import json

import numpy as np
import pytest

from griptile_core.contact_image import DEFAULT_CONTACT_SPAN, STRONG_RGB, amplitude_rgb
from griptile_core.load_calibration import LoadTrial, relative_load_polarity


def test_independent_checks_expose_loading_error_hysteresis_and_release_residual():
    trial = LoadTrial(check_n=4, calibration_n=10)
    assert trial.estimate(800) is None
    stages = []
    for total in (600, 2000, 900, 80):
        stages.append(trial.stage)
        trial.record(total)
    assert stages == ["check_up", "calibration", "check_down", "release"]
    assert trial.stage == "done"
    report = json.loads(json.dumps(trial.report(), allow_nan=False))
    assert report["scale_n_per_adc"] == .005
    assert report["calibrated_range_adc"] == [0, 2000]
    holds = report["holds"]
    assert holds["check_up"] == {
        "known_n": 4, "estimated_n": 3, "total_adc": 600,
        "error_n": -1, "relative_error_pct": -25, "in_calibrated_range": True,
    }
    assert holds["check_down"]["estimated_n"] == 4.5
    assert holds["check_down"]["relative_error_pct"] == 12.5
    assert report["unloading_minus_loading_n"] == 1.5
    assert report["release_residual_n"] == holds["release"]["error_n"] == .4
    assert holds["release"]["relative_error_pct"] is None


def test_overrange_is_reported_and_counts_above_rgb_saturation_still_contribute_to_fit():
    trial = LoadTrial(3, 8)
    # The one active taxel shows the same red at both loads. Numerical totals
    # must still distinguish a 4000-count anchor from a 6000-count response.
    for total in (0, 4000, 6000, 0):
        field = np.zeros((20, 20))
        field[14, 2] = total
        if total:
            assert tuple(amplitude_rgb(field, DEFAULT_CONTACT_SPAN)[14, 2]) == STRONG_RGB
        trial.record(float(field.sum()))
    holds = trial.report()["holds"]
    assert trial.estimate(6000) == 12
    assert holds["check_down"]["in_calibrated_range"] is False
    assert holds["check_down"]["total_adc"] == 6000
    assert holds["check_down"]["estimated_n"] == 12
    assert holds["check_up"]["error_n"] == -3
    assert holds["release"]["estimated_n"] == 0


@pytest.mark.parametrize("check,high", [(0, 2), (1, 1), (2, 1), (-1, 2), (np.nan, 4),
                                        (1, np.inf), (True, 2), (1, None)])
def test_invalid_loads(check, high):
    with pytest.raises(ValueError):
        LoadTrial(check, high)


def test_an_invalid_hold_can_be_retried_without_losing_prior_results():
    trial = LoadTrial(2, 4)
    trial.record(160)
    with pytest.raises(ValueError):
        trial.report()
    for bad in (0, -1, np.nan, None):
        with pytest.raises(ValueError):
            trial.record(bad)
        assert trial.stage == "calibration"
        assert trial.scale_n_per_adc is None
    trial.record(400)
    assert trial.estimate(160) == 1.6
    trial.record(170)
    trial.record(0)
    with pytest.raises(ValueError):
        trial.record(4)
    assert trial.report()["holds"]["check_up"]["total_adc"] == 160


def test_additional_check_with_a_different_response_keeps_original_calibration():
    trial = LoadTrial(2, 4)
    with pytest.raises(ValueError):
        trial.verify(2, 100)
    for total in (100, 200, 100, 0):
        trial.record(total)
    before = trial.report()
    center = trial.verify(2, 100)
    edge = trial.verify(2, 60)
    assert center["error_n"] == 0
    assert edge["estimated_n"] == 1.2
    assert edge["error_n"] == -.8
    assert trial.verify(4, 250)["in_calibrated_range"] is False
    for known_n, total in [(4.1, 200), (np.nan, 0), (-1, 0), (2, -10)]:
        with pytest.raises(ValueError):
            trial.verify(known_n, total)
    assert trial.report() == before


@pytest.mark.parametrize("anchor, signed, label", [
    (2000, 800, "same"), (-2000, -800, "same"),
    (2000, -800, "opposite"), (-2000, 800, "opposite"),
    (2000, -400, "opposite"), (2000, -399, "mixed"),
    (-2, 800, "mixed"), (0, -800, "mixed"), (-2000, 1, "mixed"),
])
def test_only_dominant_signal_directions_are_compared(anchor, signed, label):
    assert relative_load_polarity(2000, anchor, 800, signed) == label


def test_unloaded_and_invalid_signed_totals():
    assert relative_load_polarity(400, -399.99, 0, 0) == "no_response"
    for invalid in (np.nan, np.inf, -800.1, 801, True):
        with pytest.raises(ValueError):
            relative_load_polarity(2000, 2000, 800, invalid)
