import numpy as np
import pytest

from griptile_core.equalization import EqualizationProfile
from griptile_core.grid import SHAPE
from griptile_viewer.print_capture import ContactPrintCapture


MS = 1_000_000
THRESHOLDS = np.full(SHAPE, 10.0)


def field(value):
    result = np.zeros(SHAPE, np.float32)
    result[3, 6] = value
    return result


def feed(capture, levels, *, at=0, step=50):
    for i, value in enumerate(levels):
        capture.observe((at + i * step) * MS, field(value), THRESHOLDS)


def test_manual_exposures_use_nearest_source_stamps_and_cannot_mutate():
    c = ContactPrintCapture()
    original = field(100)
    # The renderer may poll irregularly; timestamps do not get replaced by idealized times.
    for t in (0, 57, 104, 218, 291, 376, 413, 542, 609):
        c.observe(t * MS, original, THRESHOLDS)
    assert c.can_snapshot
    p = c.snapshot()
    assert p.stamps_ns == (0, 291 * MS, 609 * MS) and not p.automatic and not p.equalized
    original[:] = 99
    assert all(s[3, 6] == 100 for s in p.frames)
    with pytest.raises(ValueError):
        p.frames[0].flags.writeable = True
    c.interrupt()
    assert not c.can_snapshot and c.snapshot() is None and c.latest is p
    c.reset()
    assert c.latest is None


def test_release_is_one_observed_edge_only():
    c = ContactPrintCapture()
    feed(c, [0] * 20 + [900] * 16 + [700, 400, 0])
    assert c.latest is None
    feed(c, [0] * 10, at=1950)
    first = c.latest
    assert first.automatic and not first.equalized
    assert first.frames[0][3, 6] == first.frames[1][3, 6] == 900
    assert first.frames[2][3, 6] == 0
    feed(c, [0] * 10, at=2450)
    assert c.latest is first
    feed(c, [500] * 12 + [0] * 12, at=2950)
    assert c.latest is not first and c.latest.automatic
    assert c.latest.stamps_ns[0] > first.stamps_ns[-1]
    assert max(s[3, 6] for s in c.latest.frames) == 500


@pytest.mark.parametrize('step', [33, 100])
def test_quick_and_slow_releases_keep_real_contact_exposures(step):
    for levels in ([0] * 24 + [600] * 5 + [0] * 24,
                   [0] * 24 + [600] * 25 + list(np.linspace(600, 0, 32)) + [0] * 24):
        c = ContactPrintCapture()
        feed(c, levels, step=step)
        got = c.latest
        assert got is not None and got.automatic
        indices = [t // (step * MS) for t in got.stamps_ns]
        expected = np.asarray(levels)[indices].astype(np.float32)
        np.testing.assert_array_equal([s[3, 6] for s in got.frames], expected)
        assert max(expected[:-1]) > expected[-1]
        for a, b in zip(got.stamps_ns, got.stamps_ns[1:]):
            assert abs((b - a) / MS - 300) <= step


def test_sustained_evidence_is_required_for_automatic_capture():
    c = ContactPrintCapture()
    feed(c, [0] * 16 + [1200] + [0] * 12 + [79] * 16 + [0] * 16)
    assert c.latest is None
    t = 4000 * MS
    for i in range(20):
        c.observe(t, field(900), THRESHOLDS)
        c.observe(t - i * MS, field(900), THRESHOLDS)
    feed(c, [0] * 20, at=4020)
    assert c.latest is None


def test_a_lighter_hold_remains_contact_and_unrelated_weak_background_cannot_block_release():
    c = ContactPrintCapture()
    states = [0] * 20 + [110] * 6 + [70] * 25
    for i, level in enumerate(states + [0] * 20):
        measured = field(level)
        measured[7, 8] = 70  # Never a strong seed at this independent taxel.
        c.observe(i * 50 * MS, measured, THRESHOLDS)
        if i < len(states):
            assert c.latest is None
    assert c.latest.automatic
    assert [f[3, 6] for f in c.latest.frames] == [70, 70, 0]
    assert all(f[7, 8] == 70 for f in c.latest.frames)


def test_gap_partial_scan_and_new_context_cannot_complete_a_release():
    for barrier in ('gap', 'missing', 'partial'):
        c = ContactPrintCapture()
        feed(c, [900] * 20)
        previous = c.snapshot()
        if barrier != 'gap':
            missing = field(0)
            if barrier == 'missing':
                missing[:] = np.nan
            else:
                missing[1, 8] = np.nan
            c.observe(1000 * MS, missing, THRESHOLDS)
        feed(c, [0] * 16, at=1260 if barrier == 'gap' else 1050)
        assert c.latest is previous
    c.reset()
    feed(c, [0] * 14)
    assert not c.can_snapshot and c.contact_present is False and c.latest is None


def test_manual_preserves_missing_per_cell_without_inventing_contacts():
    c = ContactPrintCapture()
    for i in range(9):
        x = field(900)
        x[0, 0] = np.nan
        c.observe(i * 100 * MS, x, THRESHOLDS)
    assert all(np.isnan(f[0, 0]) for f in c.snapshot().frames)
    for i in range(9, 20):
        c.observe(i * 100 * MS, field(0), THRESHOLDS)
    assert not c.latest.automatic


def test_gains_affect_print_pixels_only_and_do_not_mix_in_one_triplet():
    low = np.full(SHAPE, 100.0)
    low[3, 6] = 50.0
    profile = EqualizationProfile('both', tuple(low.ravel()), tuple(2 * low.ravel()))
    c = ContactPrintCapture()
    feed(c, [90] * 16)
    raw = c.snapshot()
    assert not raw.equalized
    for n in range(16, 40):
        c.observe(n * 50 * MS, field(90), THRESHOLDS, profile)
        if n == 16:
            assert not c.can_snapshot and c.latest is None
    assert c.latest is None
    result = c.snapshot()
    assert result.equalized and max(f[3, 6] for f in result.frames) == 180
    for n in range(40, 46):
        c.observe(n * 50 * MS, field(0), THRESHOLDS, profile)
    assert c.latest is result and not c.can_snapshot


def test_manual_requires_overlapping_observations():
    c = ContactPrintCapture()
    for n in range(7):
        f = np.full(SHAPE, np.nan)
        f[n, 0] = 100 + n
        c.observe(n * 100 * MS, f, THRESHOLDS)
    assert not c.can_snapshot


def test_current_state_quiet_with_residual_and_lighter_held_contact():
    c = ContactPrintCapture()
    assert c.contact_present is None and c.contact_id is None and c.preview() is None
    # A stable offset on many taxels can survive the driver's noise gate.
    feed(c, [70] * 20)
    assert c.contact_present is False and not c.can_snapshot and c.snapshot() is None
    feed(c, [80, 25, 20], at=1000)
    token = c.contact_id
    current = c.preview()
    assert token is not None and c.contact_present is True and c.latest is None
    assert current is c.preview() and current.frames[-1][3, 6] == 20
    assert c.snapshot() is current and c.latest is current
    c.observe(1150 * MS, field(19), THRESHOLDS)
    assert c.contact_id is None and c.contact_present is False
    assert c.snapshot() is None and c.latest is current
    c.interrupt()
    assert c.contact_present is None and c.preview() is None and c.latest is current


def test_new_contact_preview_does_not_overlay_previous_contact_or_modify_release_history():
    c = ContactPrintCapture()
    feed(c, [0] * 16 + [300] * 16)
    first_id = c.contact_id
    assert c.preview() and c.latest is None
    feed(c, [0] * 8, at=1600)
    history = c.latest
    assert history and history.automatic and c.preview() is None
    second = np.zeros(SHAPE, np.float32)
    second[12, 10] = 900
    c.observe(2000 * MS, second, THRESHOLDS)
    assert c.contact_id is not None and c.contact_id is not first_id and c.contact_present is True
    assert c.preview() is None and c.latest is history
    for t in range(2050, 2251, 50):
        c.observe(t * MS, second, THRESHOLDS)
    record = c.preview()
    assert c.latest is history and record.stamps_ns[0] >= 1600 * MS
    assert all(frame[3, 6] == 0 for frame in record.frames)
    assert record.frames[-1][12, 10] == 900


def test_partial_scan_can_prove_contact_but_cannot_prove_no_contact():
    c = ContactPrintCapture()
    x = field(100)
    x[4, 4] = np.nan
    for i in range(8):
        c.observe(i * 100 * MS, x, THRESHOLDS)
    assert c.contact_present is True and c.can_snapshot
    x[3, 6] = 30
    c.observe(800 * MS, x, THRESHOLDS)
    assert c.contact_present is True
    x[3, 6] = np.nan
    c.observe(900 * MS, x, THRESHOLDS)
    assert c.contact_present is None and c.preview() is None
    feed(c, [0] * 12, at=1000)
    assert c.contact_present is False and c.latest is None
