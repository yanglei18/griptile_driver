"""One physical taxel per output pixel, including contact and release."""

import numpy as np
import pytest

from griptile_core.contact_image import (
    DEFAULT_CONTACT_SPAN, INVALID_RGB, STRONG_RGB, SURFACE_SCALE, UNLOADED_RGB, WEAK_RGB,
    amplitude_rgb, contact_amplitude, contact_rgb, contact_surface_rgb,
)
from griptile_core.display_response import ReferenceProfile, ResponseSettings, TactileResponseProcessor
from griptile_core.grid import CELL_COUNT, COLS, ROWS


IDENTITY = ("sensor", 1)
BASELINE = np.arange(CELL_COUNT) + 20_000


def calibrated(direction="both", reference_mask=None):
    if reference_mask is None:
        reference_mask = np.ones(CELL_COUNT, dtype=bool)
    profile = ReferenceProfile(BASELINE, np.full(CELL_COUNT, 10.), reference_mask)
    detector = TactileResponseProcessor(ResponseSettings(DEFAULT_CONTACT_SPAN, direction))
    detector.load_reference(profile, IDENTITY, 0)
    return detector


def image(detector, adc, *, time=1, mask=None):
    valid = np.ones(CELL_COUNT, dtype=bool) if mask is None else mask
    return contact_rgb(detector.transform(adc, valid, IDENTITY, time))


def test_empty_contact_requires_fresh_measurement_and_reference():
    assert np.all(image(TactileResponseProcessor(), BASELINE) == INVALID_RGB)
    refs = np.ones(CELL_COUNT, dtype=bool)
    refs[77] = False
    valid = np.ones(CELL_COUNT, dtype=bool)
    valid[118] = False
    rgb = image(calibrated(reference_mask=refs), BASELINE, mask=valid)
    assert rgb.shape == (ROWS, COLS, 3)
    assert rgb.dtype == np.uint8 and rgb.flags.c_contiguous
    flat = rgb.reshape(-1, 3)
    assert np.all(flat[[77, 118]] == INVALID_RGB)
    assert np.all(flat[refs & valid] == UNLOADED_RGB)


def test_contact_positions_and_hole_survive_rgb_encoding_and_release():
    # Asymmetric angle with a one-cell cavity and a separate point near the edge.
    footprint = np.zeros((ROWS, COLS), dtype=bool)
    footprint[1:4, 6:9] = True
    footprint[2, 7] = False
    footprint[3:7, 6] = True
    footprint[19, 0] = True
    loaded = BASELINE + 6000 * footprint.ravel()
    detector = calibrated()
    rgb = image(detector, loaded)
    assert np.all(rgb[footprint] == STRONG_RGB)
    assert np.all(rgb[~footprint] == UNLOADED_RGB)
    assert np.all(image(detector, BASELINE, time=2) == UNLOADED_RGB)


@pytest.mark.parametrize("direction,sign", [("increase", 1), ("decrease", -1), ("both", -1)])
def test_rgb_follows_configured_polarity_and_detector_hysteresis(direction, sign):
    detector = calibrated(direction)
    def at(delta, time):
        values = BASELINE.copy()
        values[0] += sign * delta
        return tuple(image(detector, values, time=time)[0, 0])

    assert at(10, 1) == UNLOADED_RGB  # Activation needs more than ten counts.
    assert at(11, 2) != UNLOADED_RGB
    assert at(6, 3) == WEAK_RGB      # A held touch remains until five counts.
    assert at(5, 4) == UNLOADED_RGB
    assert at(6, 5) == UNLOADED_RGB
    if direction != "both":
        assert at(-2000, 6) == UNLOADED_RGB


def test_fixed_colors_are_independent_of_other_taxels_and_clip_only_at_full_scale():
    detector = calibrated()
    values = BASELINE.copy()
    values[:3] += [20, 1505, 3005]
    first = image(detector, values)
    weak, middle, full = first[0, :3]
    assert weak[0] < middle[0] < full[0]
    assert weak[1] > middle[1] > full[1]
    assert weak[2] > middle[2] > full[2]
    assert tuple(full) == STRONG_RGB
    values[2] += 5000
    np.testing.assert_array_equal(image(detector, values, time=2), first)


def test_surface_keeps_measured_touch_states_and_invalid_taxel_footprints():
    rng = np.random.default_rng(17)
    amplitude = rng.integers(0, 6000, size=(ROWS, COLS)).astype(float)
    amplitude[5::3, 8::4] = np.nan
    amplitude[::2, ::3] = 0
    native = contact_rgb({"reference_ready": True, "values": amplitude,
                          "valid_mask": np.isfinite(amplitude), "levels": (0, DEFAULT_CONTACT_SPAN)})
    surface = contact_surface_rgb(amplitude, DEFAULT_CONTACT_SPAN)
    assert surface.shape == (300, 300, 3) and surface.dtype == np.uint8 and surface.flags.c_contiguous
    center = SURFACE_SCALE // 2
    centers = surface[center::SURFACE_SCALE, center::SURFACE_SCALE]
    np.testing.assert_array_equal(centers[amplitude == 0], native[amplitude == 0])
    # Contact opacity can soften a boundary taxel, including its center.
    loaded = centers[amplitude > 0]
    assert np.all(loaded[:, 0] > loaded[:, 1])
    missing = np.repeat(np.repeat(np.isnan(amplitude), SURFACE_SCALE, axis=0), SURFACE_SCALE, axis=1)
    assert np.all(surface[missing] == INVALID_RGB)


@pytest.mark.parametrize("strength", [1., 1500., 6000.])
def test_single_touch_feathers_without_losing_its_peak(strength):
    amplitude = np.zeros((ROWS, COLS))
    row, col = 5, 12
    amplitude[row, col] = strength
    surface = contact_surface_rgb(amplitude, DEFAULT_CONTACT_SPAN)
    y = SURFACE_SCALE * row + SURFACE_SCALE // 2
    x = SURFACE_SCALE * col + SURFACE_SCALE // 2
    np.testing.assert_array_equal(surface[y, x], amplitude_rgb(amplitude, DEFAULT_CONTACT_SPAN)[row, col])
    np.testing.assert_array_equal(surface[x, y], UNLOADED_RGB)
    # The white channel contribution increases gradually over several pixels
    # before the neighboring unloaded center. A binary hard edge fails this.
    outward = surface[y, x:x + SURFACE_SCALE + 1, 1].astype(int)
    assert outward[-1] == 255 and len(np.unique(outward)) >= 8
    assert np.all((np.diff(outward) >= 0) & (np.diff(outward) < 64))
    np.testing.assert_array_equal(outward, surface[y:y + SURFACE_SCALE + 1, x, 1])


def test_diagonally_touching_contacts_keep_a_white_gap():
    field = np.zeros((ROWS, COLS))
    field[5, 5] = field[6, 6] = DEFAULT_CONTACT_SPAN
    surface = contact_surface_rgb(field, DEFAULT_CONTACT_SPAN)
    p = 5 * SURFACE_SCALE + SURFACE_SCALE // 2
    offset = np.arange(SURFACE_SCALE + 1)
    assert np.all(surface[p + offset, p + SURFACE_SCALE - offset] == UNLOADED_RGB)


def test_surface_keeps_holes_separated_contacts_and_unknown_rows():
    amplitude = np.zeros((ROWS, COLS))
    amplitude[2:8, 7:13] = 1800
    amplitude[4:6, 9:11] = 0
    amplitude[10:12, 7:13] = 3000
    amplitude[14, :] = np.nan
    amplitude[15:, :] = 700
    surface = contact_surface_rgb(amplitude, DEFAULT_CONTACT_SPAN)
    center = SURFACE_SCALE // 2
    assert np.all(surface[4 * SURFACE_SCALE + center:5 * SURFACE_SCALE + center + 1,
                          9 * SURFACE_SCALE + center:10 * SURFACE_SCALE + center + 1] == UNLOADED_RGB)
    assert np.all(surface[8 * SURFACE_SCALE + center:9 * SURFACE_SCALE + center + 1] == UNLOADED_RGB)
    assert np.all(surface[14 * SURFACE_SCALE:15 * SURFACE_SCALE] == INVALID_RGB)
    # Unsupported neighborhoods on the known sides keep their own measurement.
    np.testing.assert_array_equal(surface[14 * SURFACE_SCALE - 1, 0], UNLOADED_RGB)
    np.testing.assert_array_equal(surface[15 * SURFACE_SCALE, 0], surface[15 * SURFACE_SCALE + center, 0])


@pytest.mark.parametrize("value,color", [(0, UNLOADED_RGB), (np.nan, INVALID_RGB), (6000, STRONG_RGB)])
def test_uniform_surfaces_fill_image_to_its_outer_edges(value, color):
    assert np.all(contact_surface_rgb(np.full((ROWS, COLS), value), DEFAULT_CONTACT_SPAN) == color)


def test_contact_amplitude_preserves_supra_range_counts_and_missing_measurements():
    assert np.isnan(contact_amplitude({"reference_ready": False})).all()
    valid = np.ones(CELL_COUNT, dtype=bool)
    valid[2] = False
    values = np.arange(CELL_COUNT, dtype=float)
    values[1] = 5000
    response = {"reference_ready": True, "values": values, "valid_mask": valid,
                "levels": (0., DEFAULT_CONTACT_SPAN)}
    field = contact_amplitude(response)
    assert field.shape == (ROWS, COLS) and np.issubdtype(field.dtype, np.floating)
    np.testing.assert_equal(field[0, :3], [0., 5000., np.nan])
    np.testing.assert_equal(amplitude_rgb(field, DEFAULT_CONTACT_SPAN), contact_rgb(response))
    assert not np.shares_memory(values, field)
