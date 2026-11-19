"""Color, contour and time semantics of the contact print exported to an operator."""

import numpy as np
import pytest

from griptile_core.contact_image import (
    INVALID_RGB, STRONG_RGB, SURFACE_SCALE, UNLOADED_RGB, amplitude_rgb, contact_surface_rgb,
)
from griptile_core.contact_print import render_contact_print
from griptile_core.grid import SHAPE


CENTER = SURFACE_SCALE // 2


@pytest.mark.parametrize("span", [900., 3000., 9000.])
def test_three_frames_match_live_color_map_and_latest_native_centers(span):
    rows, cols = np.indices(SHAPE)
    fields = np.stack([factor * (rows + cols) * 14.0 for factor in (1, 5, 9)])
    fields[:, 0, 0] = 0
    fields[2, 11, 6] = np.nan
    original = fields.copy()
    master, proofs = render_contact_print(fields, span)
    assert master.shape == (300, 300, 3) and proofs.shape == (3, 300, 300, 3)
    assert all(a.flags.c_contiguous and a.dtype == np.uint8 for a in (proofs, master))
    for i, field in enumerate(fields):
        np.testing.assert_array_equal(proofs[i], contact_surface_rgb(field, span))
    centers = master[CENTER::SURFACE_SCALE, CENTER::SURFACE_SCALE]
    np.testing.assert_array_equal(centers, amplitude_rgb(fields[-1], span))
    np.testing.assert_array_equal(fields, original)


def test_saturation_identical_loads_and_zero_match_display_across_exposures():
    fields = np.full((3, *SHAPE), 1200.)
    fields[1, 2, 4] = 3200
    fields[2, 2, 4] = 11000
    fields[:, 0, 0] = 0
    _, proofs = render_contact_print(fields, 3000)
    samples = proofs[:, CENTER::SURFACE_SCALE, CENTER::SURFACE_SCALE]
    assert np.all(samples[:, 0, 0] == UNLOADED_RGB)
    assert np.all(samples[1:, 2, 4] == STRONG_RGB)
    assert np.all(samples[:, 16, 17] == [34, 176, 100])  # 40% green in every frame.
    assert fields[2, 2, 4] == 11000


def test_current_contours_mark_relative_isolevels_without_historical_ghosts():
    fields = np.broadcast_to(np.arange(20) * 250., (3, *SHAPE)).copy()
    fields[0] = fields[1] = fields[-1, :, ::-1].copy()
    span = 2800.
    master, proofs = render_contact_print(fields, span)
    delta = proofs[-1, 130].astype(float) - master[130]
    columns = np.arange(master.shape[1])
    marked = np.zeros(len(columns), dtype=bool)
    for fraction in (.2, .4, .6, .8, 1.):
        center = CENTER + fraction * span * SURFACE_SCALE / 250
        close = np.abs(columns - center) <= 1
        assert delta[close].max() > 15
        marked |= close
    assert not delta[~marked].any() and (delta >= 0).all()
    fields[:2] = 0.
    redrawn, _ = render_contact_print(fields, span)
    np.testing.assert_array_equal(master, redrawn)


def test_gradients_above_full_scale_stay_saturated_without_extra_contours():
    fields = np.broadcast_to(3200. + np.arange(20) * 580., (3, *SHAPE))
    master, proofs = render_contact_print(fields, 3000.)
    assert np.all(master == STRONG_RGB) and np.all(proofs == STRONG_RGB)


def test_holes_dropouts_and_complete_release_are_current_only():
    fields = np.zeros((3, *SHAPE))
    fields[:, 3:12, 4:14] = 2500
    fields[:, 6:9, 8:10] = 0
    fields[0, 10, 11] = np.nan
    fields[2, 10, 5] = np.nan
    master, proofs = render_contact_print(fields)
    centers = master[CENTER::SURFACE_SCALE, CENTER::SURFACE_SCALE]
    np.testing.assert_array_equal(centers[7, 8], UNLOADED_RGB)
    missing_first = np.isnan(fields[0]).repeat(SURFACE_SCALE, axis=0).repeat(SURFACE_SCALE, axis=1)
    assert np.all(proofs[0][missing_first] == INVALID_RGB)
    assert not np.array_equal(centers[10, 11], INVALID_RGB)
    unknown = np.isnan(fields[2]).repeat(SURFACE_SCALE, axis=0).repeat(SURFACE_SCALE, axis=1)
    assert np.all(master[unknown] == INVALID_RGB)
    unloaded = np.all(proofs[2] == UNLOADED_RGB, axis=2)
    assert np.all(master[unloaded] == UNLOADED_RGB)
    fields[2] = 0
    released, _ = render_contact_print(fields)
    assert np.all(released == UNLOADED_RGB)


@pytest.mark.parametrize("fields", [np.zeros((2, *SHAPE)), np.zeros((3, 400)),
                                    np.full((3, *SHAPE), -1.), np.full((3, *SHAPE), np.inf),
                                    np.full((3, *SHAPE), True)])
def test_invalid_fields_are_rejected(fields):
    with pytest.raises(ValueError):
        render_contact_print(fields)


@pytest.mark.parametrize("span", [0., -100., np.nan, np.inf])
def test_invalid_span_is_rejected(span):
    with pytest.raises(ValueError):
        render_contact_print(np.zeros((3, *SHAPE)), span)
