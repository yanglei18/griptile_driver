"""Color, contour and time semantics of the contact print exported to an operator."""

import numpy as np
import pytest

from griptile_core.contact_image import (
    CONTACT_OUTLINE_RGB, INVALID_RGB, STRONG_RGB, SURFACE_SCALE, UNLOADED_RGB,
    amplitude_rgb, contact_footprint, contact_surface_rgb,
)
from griptile_core.contact_print import render_contact_face, render_contact_print
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


def test_contact_face_uses_geometry_instead_of_internal_intensity_bands():
    # Below the seed threshold, but connected to one strong taxel, the left
    # column and bottom rim of this ring still have measured support.
    values = np.zeros(SHAPE)
    values[4:10, 4:11] = 40.
    values[5:9, 5:10] = 0.
    values[4, 4:11] = np.linspace(90., 4500., 7)
    values[17, 18] = 65.   # Isolated weak response stays visible in the proof only.
    thresholds = np.full(SHAPE, 10.)
    region = contact_footprint(values, thresholds)
    assert region[9, 7] and not region[17, 18]
    frames = np.stack([np.full(SHAPE, 500.), np.full(SHAPE, 900.), values])
    master, proofs = render_contact_print(frames, 3000., region)
    np.testing.assert_array_equal(master, render_contact_face(values, 3000., region))
    np.testing.assert_array_equal(proofs[-1], contact_surface_rgb(values, 3000.))
    native = master[CENTER::SURFACE_SCALE, CENTER::SURFACE_SCALE]
    np.testing.assert_array_equal(native[region], amplitude_rgb(values, 3000.)[region])
    assert np.all(native[~region] == UNLOADED_RGB)
    assert not np.array_equal(proofs[-1, 17 * SURFACE_SCALE + CENTER, 18 * SURFACE_SCALE + CENTER], UNLOADED_RGB)
    ink = np.all(master == CONTACT_OUTLINE_RGB, axis=2)
    # A vertical slice crosses only the outer and inner ring edges. The red,
    # green and yellow patches in the top rim do not create any extra contours.
    column = 7 * SURFACE_SCALE + CENTER
    assert np.flatnonzero(ink[:, column]).tolist() == [60, 74, 135, 149]
    assert not ink[4 * SURFACE_SCALE + CENTER, 5 * SURFACE_SCALE:10 * SURFACE_SCALE].any()
    values[9, 7] = 0
    opened = render_contact_face(values, 3000., contact_footprint(values, thresholds))
    assert np.all(opened[9 * SURFACE_SCALE:10 * SURFACE_SCALE, column] == UNLOADED_RGB)


def test_started_contact_can_support_neighbors_while_unrelated_or_missing_taxels_cannot_seed():
    values = np.zeros(SHAPE)
    values[6, 7:10] = 25.
    values[7, 8] = np.nan
    values[8, 8:11] = 60.
    active = np.zeros(SHAPE, dtype=bool)
    active[6, 7] = active[7, 8] = True
    thresholds = np.full(SHAPE, 10.)
    region = contact_footprint(values, thresholds, active)
    assert region[6, 7:10].all() and region.sum() == 3
    values[6, 7] = 19.
    assert not contact_footprint(values, thresholds, active).any()
    values[6, 7] = 25.
    thresholds[6, 7] = np.nan
    assert not contact_footprint(values, thresholds, active).any()


def test_footprint_at_sensor_border_or_missing_rows_has_no_invented_end_cap():
    values = np.full(SHAPE, 200.)
    values[5] = np.nan
    thresholds = np.full(SHAPE, 10.)
    region = contact_footprint(values, thresholds)
    master = render_contact_face(values, 3000., region)
    np.testing.assert_array_equal(master, contact_surface_rgb(values, 3000.))
    assert np.all(master[5 * SURFACE_SCALE:6 * SURFACE_SCALE] == INVALID_RGB)


@pytest.mark.parametrize("invalid", [np.ones((19, 20), bool), np.ones(SHAPE, dtype=int)])
def test_native_boolean_geometry_is_required_for_a_measured_shape(invalid):
    with pytest.raises(ValueError, match="boolean 20x20"):
        render_contact_face(np.ones(SHAPE), 3000, invalid)
    with pytest.raises(ValueError, match="boolean 20x20"):
        contact_footprint(np.ones(SHAPE), np.ones(SHAPE), invalid)


def test_gradients_above_full_scale_stay_saturated_without_extra_contours():
    fields = np.broadcast_to(3200. + np.arange(20) * 580., (3, *SHAPE))
    master, proofs = render_contact_print(fields, 3000., footprint=np.ones(SHAPE, dtype=bool))
    assert np.all(master == STRONG_RGB) and np.all(proofs == STRONG_RGB)


def test_holes_dropouts_and_complete_release_are_current_only():
    fields = np.zeros((3, *SHAPE))
    fields[:, 3:12, 4:14] = 2500
    fields[:, 6:9, 8:10] = 0
    fields[0, 10, 11] = np.nan
    fields[2, 10, 5] = np.nan
    master, proofs = render_contact_print(fields, footprint=np.nan_to_num(fields[-1]) > 0)
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
    released, _ = render_contact_print(fields, footprint=fields[-1] > 0)
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


def test_registered_inks_show_growth_hold_and_release_with_the_same_color_proofs():
    fields = np.stack([np.full(SHAPE, v, dtype=np.float32) for v in (200, 900, 2200)])
    region = np.ones(SHAPE, dtype=bool)
    growing, proofs = render_contact_print(fields, footprint=region, style="imprint")
    falling, inverse_proofs = render_contact_print(fields[::-1], footprint=region, style="imprint")
    assert growing[100, 100, 0] > growing[100, 100, 2] + 12
    assert falling[100, 100, 0] + 12 < falling[100, 100, 2]
    np.testing.assert_array_equal(proofs, inverse_proofs[::-1])
    for proof, field in zip(proofs, fields):
        np.testing.assert_array_equal(proof, contact_surface_rgb(field, 3000.))

    # On a stationary slope, contour bands remain visible inside the patch;
    # taking three equal exposures registers the same near-neutral ink as one.
    stationary = np.broadcast_to(np.linspace(0, 9000., 20), SHAPE)
    single = render_contact_face(stationary, 3000., region, style="imprint")
    held, _ = render_contact_print(np.stack([stationary] * 3), footprint=region, style="imprint")
    np.testing.assert_allclose(held, single, atol=1)
    x = np.where(np.max(np.diff(held[150].astype(int), axis=0), axis=1) > 35)[0]
    assert len(x) >= 6
    np.testing.assert_allclose(single, render_contact_face(2 * stationary, 6000., region, style="imprint"))


def test_imprint_ink_cannot_extend_the_latest_footprint_or_close_a_hole():
    fields = np.full((3, *SHAPE), 2200.)
    fields[0, 1:3, 13:17] = 8400.  # Former contact outside the current patch.
    region = np.zeros(SHAPE, dtype=bool)
    region[3:12, 3:13] = True
    region[5:10, 5:11] = False
    region[16, 17] = True
    master, _ = render_contact_print(fields, footprint=region, style="imprint")
    ground = np.array((13, 16, 19))
    visible = np.any(master != ground, axis=2)
    native = visible[CENTER::SURFACE_SCALE, CENTER::SURFACE_SCALE]
    np.testing.assert_array_equal(native, region)
    col = 8 * SURFACE_SCALE + CENTER
    assert np.flatnonzero(np.diff(visible[:, col])).tolist() == [44, 74, 149, 179]
    assert master[45, col, 0] > master[46, col, 0]  # Outer edge.
    assert master[74, col, 0] > master[73, col, 0]  # Hole edge.
    empty, _ = render_contact_print(fields, footprint=np.zeros_like(region), style="imprint")
    assert np.all(empty == ground)


def test_imprint_missing_history_shows_current_neutral_ink_and_latest_missing_stays_gray():
    rows, cols = np.indices(SHAPE)
    current = np.where(cols < 10, 500., 2400.) + rows * 75.
    fields = np.stack([current / 4, current / 2, current])
    fields[0, 3:7, 4:9] = np.nan
    fields[1, 10:17, 1:6] = np.nan
    fields[-1, 8, 9] = np.nan
    region = np.ones(SHAPE, dtype=bool)
    master, proofs = render_contact_print(fields, footprint=region, style="imprint")
    single = render_contact_face(fields[-1], 3000., region, style="imprint")
    for r, c in ((4, 5), (13, 3)):
        sample = np.s_[r * SURFACE_SCALE:(r + 1) * SURFACE_SCALE, c * SURFACE_SCALE:(c + 1) * SURFACE_SCALE]
        np.testing.assert_array_equal(master[sample], single[sample])
        assert not np.all(master[sample] == INVALID_RGB)
    assert np.all(master[8 * SURFACE_SCALE:9 * SURFACE_SCALE, 9 * SURFACE_SCALE:10 * SURFACE_SCALE] == INVALID_RGB)
    assert np.all(proofs[0, 4 * SURFACE_SCALE:5 * SURFACE_SCALE, 5 * SURFACE_SCALE:6 * SURFACE_SCALE] == INVALID_RGB)
    assert master[200, 200, 0] > master[200, 200, 2]


def test_style_fallback_uses_measured_response_and_rejects_unknown_styles():
    fields = np.full((3, *SHAPE), 2400.)
    master, proofs = render_contact_print(fields, style="imprint")
    np.testing.assert_array_equal(master, proofs[-1])
    np.testing.assert_array_equal(render_contact_face(fields[-1], style="imprint"), master)
    with pytest.raises(ValueError, match="style"):
        render_contact_print(fields, style="unknown")


@pytest.mark.parametrize("style", ["imprint", "response"])
@pytest.mark.parametrize("measured_shape", [False, True])
def test_face_only_matches_full_print_and_checks_every_exposure(style, measured_shape):
    fields = np.arange(3 * 400, dtype=np.float32).reshape(3, *SHAPE)
    fields[0, 8, 3] = fields[2, 13, 2] = np.nan
    footprint = fields[-1] > 940 if measured_shape else None
    expected, _ = render_contact_print(fields, footprint=footprint, style=style)
    face, proofs = render_contact_print(fields, footprint=footprint, style=style, include_proofs=False)
    assert proofs is None
    np.testing.assert_array_equal(face, expected)
    fields[0, 2, 2] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        render_contact_print(fields, footprint=footprint, style=style, include_proofs=False)
