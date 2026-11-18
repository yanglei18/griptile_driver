from dataclasses import replace

import pytest

from griptile_viewer.load_capture import HoldCapture, LoadContext, LoadObservation


CONTEXT = LoadContext("pad", "fixture", 1, "", "layout", "both", 3000., "pad_frame", 1, "abc", 1)
MS = 1_000_000


def observation(t, total=80., *, signed=None):
    return LoadObservation(CONTEXT, (t * MS,) * 20, t * MS, total, total if signed is None else signed,
                           total / 2, int(total > 0) * 2)


def test_hold_skips_cached_and_partially_advanced_rows_then_returns_fresh_medians():
    # The view clock at the click is ahead of the last matched measurement.
    initial = replace(observation(1), source_now_ns=10 * MS)
    hold = HoldCapture(initial, 0)
    for t in (1, 5, 10):
        assert hold.update(observation(t), t * MS) is None
    for t in (200, 400, 600):
        partial = replace(observation(t), row_stamps=(t * MS,) * 19 + (10 * MS,))
        assert hold.update(partial, t * MS) is None
    assert hold.update(observation(700, 79), 700 * MS) is None
    assert hold.update(observation(900, 81), 900 * MS) is None
    assert hold.update(observation(900, 81), 1050 * MS) is None
    reading = hold.update(observation(1100, 80), 1100 * MS)
    assert (reading.total_adc, reading.peak_adc, reading.active_cells) == (80, 40, 2)
    assert (reading.samples, reading.spread_adc, reading.duration_s) == (3, 2, .4)
    assert (reading.first_row_monotonic_ns, reading.last_row_monotonic_ns) == (700 * MS, 1100 * MS)


def test_hold_uses_fresh_stable_series_and_accepts_negative_release_residual():
    hold = HoldCapture(observation(1), 0)
    for t, signed in [(100, -80), (300, 80), (500, -80), (700, -80)]:
        assert hold.update(observation(t, signed=signed), t * MS) is None
    reading = hold.update(observation(900, signed=-80), 900 * MS)
    assert reading.signed_total_adc == -80 and reading.total_adc == 80
    assert reading.stable and reading.signed_spread_adc == 0


def test_missing_epoch_restarts_window_and_zero_remains_a_measured_result():
    hold = HoldCapture(observation(1), 0)
    for t in (100, 300):
        assert hold.update(observation(t, 0), t * MS) is None
    assert hold.update(None, 400 * MS) is None
    for t in (500, 700):
        assert hold.update(observation(t, 0), t * MS) is None
    assert hold.update(observation(900, 0), 900 * MS).total_adc == 0


def test_context_change_and_timeout_need_a_new_hold():
    hold = HoldCapture(observation(1), 0)
    other = replace(observation(2), context=replace(CONTEXT, generation=2))
    with pytest.raises(ValueError, match="参考"):
        hold.update(other, MS)
    with pytest.raises(ValueError, match="超时"):
        hold.update(None, HoldCapture.TIMEOUT_NS)


@pytest.mark.parametrize("values, expected_total, expected_signed, spread, signed_spread", [
    ([(50, -50), (100, 100), (30, -30)], 50, -30, 70, 150),
    ([(50, -50), (50, 50), (50, -50)], 50, -50, 0, 100),
])
def test_validation_records_first_complete_window_even_if_releasing_chatter_or_reversing(
        values, expected_total, expected_signed, spread, signed_spread):
    hold = HoldCapture(observation(1), 0, require_stable=False)
    for index, (total, signed) in enumerate(values):
        t = 100 + index * 200
        result = hold.update(observation(t, total=total, signed=signed), t * MS)
        if index < 2:
            assert result is None
    assert result.samples == 3 and result.duration_s == .4
    assert not result.stable
    assert result.total_adc == expected_total and result.signed_total_adc == expected_signed
    assert (result.spread_adc, result.signed_spread_adc) == (spread, signed_spread)


@pytest.mark.parametrize("source_after_gap", [370, 6500])
def test_pause_without_intermediate_polls_starts_a_new_measurement_window(source_after_gap):
    hold = HoldCapture(observation(1), 0, require_stable=False)
    for t in (70, 170):
        assert hold.update(observation(t), t * MS) is None
    for offset in (0, 200):
        assert hold.update(observation(source_after_gap + offset), (6500 + offset) * MS) is None
    held = hold.update(observation(source_after_gap + 400), 6900 * MS)
    assert held.samples == 3 and held.duration_s == .4
    assert held.first_row_monotonic_ns == source_after_gap * MS
