"""The print and its proofs preserve measured locations, time order and validity."""

import numpy as np
import pytest

from griptile_core.contact_image import INVALID_RGB, SURFACE_SCALE
from griptile_core.contact_print import render_contact_print
from griptile_core.grid import SHAPE


GROUND = np.array([13, 16, 19])
PAPER = np.array([248, 245, 235])


def fixed_proof(value):
    tone = np.arcsinh(min(value, 12000) / 900) / np.arcsinh(12000 / 900)
    return np.rint(GROUND + (PAPER - GROUND) * tone).astype(np.uint8)


def test_channel_ownership_exposure_curve_and_center_registration():
    amplitudes = (340., 1300., 9400.)
    fields = np.zeros((3, *SHAPE))
    fields[:, 3:8, 11:16] = np.array(amplitudes)[:, None, None]
    original = fields.copy()
    composite, proofs = render_contact_print(fields)
    np.testing.assert_array_equal(fields, original)
    assert proofs.shape == (3, 300, 300, 3)
    assert composite.shape == (300, 300, 3)
    assert all(a.flags.c_contiguous and a.dtype == np.uint8 for a in (proofs, composite))

    y, x = 5 * SURFACE_SCALE + SURFACE_SCALE // 2, 13 * SURFACE_SCALE + SURFACE_SCALE // 2
    expected = np.stack([fixed_proof(v) for v in amplitudes])
    np.testing.assert_array_equal(proofs[:, y, x], expected)
    color = expected[[2, 1, 0], [0, 1, 2]]
    np.testing.assert_array_equal(composite[y, x], np.rint(GROUND + .72 * (color - GROUND)))
    np.testing.assert_array_equal(composite[x, y], GROUND)


def test_proofs_clip_at_white_only_and_flat_threshold_is_not_an_outline():
    fields = np.full((3, *SHAPE), 1500.)
    image, proofs = render_contact_print(fields)
    assert np.all(proofs == fixed_proof(1500))
    assert np.all(image == np.rint(GROUND + .72 * (fixed_proof(1500) - GROUND)))
    _, proofs = render_contact_print(fields * 100)
    assert np.all(proofs == PAPER)


def test_each_of_seven_supported_isolines_draws_at_its_native_data_level():
    fields = np.broadcast_to(np.arange(20) * 450., (3, *SHAPE))
    image, proofs = render_contact_print(fields)
    base = np.rint(GROUND + .72 * (proofs[-1, 130].astype(float) - GROUND))
    added = image[130].astype(float) - base
    columns = np.arange(image.shape[1])
    line_expected = np.zeros(len(columns), dtype=bool)
    for level in (500, 850, 1500, 2400, 3500, 5200, 7500):
        center = SURFACE_SCALE // 2 + level * SURFACE_SCALE / 450
        close = np.abs(columns - center) <= 1
        assert added[close].max() > 25
        line_expected |= close
    assert np.abs(added[~line_expected]).max() <= 1


def test_earliest_and_latest_contours_have_distinct_cool_and_warm_inks():
    inks = ((.09, .53, .73), (.24, .27, .23), (.67, .20, .04))
    for phase, ink in enumerate(inks):
        fields = np.zeros((3, *SHAPE))
        fields[phase] = np.arange(SHAPE[1]) * 500.
        image, proofs = render_contact_print(fields)
        y, x = 130, SURFACE_SCALE + SURFACE_SCALE // 2  # Precisely on the 500 ADC level.
        base = GROUND + .72 * (proofs[[2, 1, 0], y, x, [0, 1, 2]] - GROUND)
        expected = np.rint(base + .98 * np.asarray(ink) * (PAPER - base))
        np.testing.assert_array_equal(image[y, x], expected)


def test_hole_is_empty_and_separate_contact_stays_visible():
    fields = np.zeros((3, *SHAPE))
    fields[:, 3:10, 3:10] = 2200
    fields[:, 5:8, 5:8] = 0
    fields[:, 15, 17] = 800
    center = SURFACE_SCALE // 2
    image, _ = render_contact_print(fields)
    centers = image[center::SURFACE_SCALE, center::SURFACE_SCALE]
    np.testing.assert_array_equal(centers[6, 6], GROUND)
    assert np.all(centers[15, 17] > GROUND)
    np.testing.assert_array_equal(centers[15, 16], GROUND)


def test_missing_data_in_one_exposure_marks_whole_cell_but_not_other_proofs():
    fields = np.full((3, *SHAPE), 6400.)
    fields[1, 11, 6] = np.nan
    image, proofs = render_contact_print(fields)
    missing = np.isnan(fields[1]).repeat(SURFACE_SCALE, 0).repeat(SURFACE_SCALE, 1)
    assert np.all(proofs[1][missing] == INVALID_RGB)
    assert np.all(proofs[[0, 2]] == fixed_proof(6400))
    assert np.all(image[missing] == INVALID_RGB)
    # No new gradient or ink beside a dropout in a uniform field.
    assert np.all(image[~missing] == image[0, 0])
    image, proofs = render_contact_print(np.full((3, *SHAPE), np.nan))
    assert np.all(image == INVALID_RGB) and np.all(proofs == INVALID_RGB)


@pytest.mark.parametrize("fields", [np.zeros((2, *SHAPE)), np.zeros((3, 400)),
                                    np.full((3, *SHAPE), -1.),
                                    np.full((3, *SHAPE), np.inf),
                                    np.full((3, *SHAPE), True)])
def test_invalid_fields_are_rejected(fields):
    with pytest.raises(ValueError):
        render_contact_print(fields)
