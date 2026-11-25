"""One physical taxel per output pixel, including contact and release."""

import numpy as np
import pytest

from griptile_core.contact_image import (
    DEFAULT_CONTACT_SPAN, INVALID_RGB, STRONG_RGB, SURFACE_SCALE, UNLOADED_RGB, WEAK_RGB,
    amplitude_rgb, contact_amplitude, contact_intensity_rgb, contact_map_rgb, contact_rgb, contact_surface_rgb,
    surface_contact_outline,
)
from griptile_core.display_response import ReferenceProfile, ResponseSettings, TactileResponseProcessor
from griptile_core.grid import CELL_COUNT, COLS, ROWS


IDENTITY = ("sensor", 1)
BASELINE = np.arange(CELL_COUNT) + 20_000


def test_readout_swatch_matches_native_taxel_at_any_display_span():
    field = np.zeros((ROWS, COLS))
    field[0, :6] = [np.nan, 0, 1, 380, 1300, 4500]
    native = amplitude_rgb(field, 1500)
    for col in range(6):
        assert contact_intensity_rgb(field[0, col], 1500) == tuple(native[0, col])


def test_operator_map_reveals_light_measured_edges_at_a_fixed_span():
    # Bottom and upper parts of the bottle-cap edge were still detectable at
    # about 96 ADC on press and 265 ADC during release at a 3000 ADC span.
    field = np.zeros((ROWS, COLS))
    field[8, :8] = [0., 6.5, 96., 265., 1500., 3000., 4500., np.nan]
    mid = SURFACE_SCALE // 2
    def centers():
        return contact_map_rgb(field, 3000.)[8 * SURFACE_SCALE + mid, mid::SURFACE_SCALE][:8]

    colors = centers()
    assert tuple(colors[0]) == (11, 17, 28) and tuple(colors[-1]) == INVALID_RGB
    normalized = colors[:6] / 255.
    linear = np.where(normalized <= .04045, normalized / 12.92, ((normalized + .055) / 1.055) ** 2.4)
    luminance = linear @ [0.2126, 0.7152, 0.0722]
    contrast = (luminance + .05) / (luminance[0] + .05)
    assert contrast[2] > 3. and contrast[3] > 4.
    assert np.all(np.diff(luminance) > 0)
    np.testing.assert_array_equal(colors[5], colors[6])
    field[0, 19] = 60_000.
    np.testing.assert_array_equal(centers(), colors)


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
    # The detector subtracts the five-count release threshold from active taxels.
    values[:6] += [605, 1205, 1805, 2405, 2990, 3005]
    first = image(detector, values)
    np.testing.assert_array_equal(first[0, :4], [
        (0, 176, 205), (34, 176, 100), (246, 217, 60), (247, 139, 37),
    ])
    assert tuple(first[0, 4]) != STRONG_RGB
    assert tuple(first[0, 5]) == STRONG_RGB
    values[5] += 5000
    np.testing.assert_array_equal(image(detector, values, time=2), first)


@pytest.mark.parametrize("span", [1000., 3000.])
def test_first_contact_is_adjacent_to_unloaded_at_any_span(span):
    field = np.zeros((ROWS, COLS))
    field[0, :5] = [np.nan, 0, 1.e-6 * span, span, span * 2]
    np.testing.assert_array_equal(amplitude_rgb(field, span)[0, :5], [
        INVALID_RGB, UNLOADED_RGB, WEAK_RGB, STRONG_RGB, STRONG_RGB,
    ])
    assert np.abs(np.subtract(UNLOADED_RGB, WEAK_RGB)).sum() == 1


def test_detected_contact_ramps_in_and_out_without_a_dark_onset():
    detector = calibrated()
    raw_deltas = [0, 10, 11, 35, 155, 365, 605, 365, 155, 35, 11, 6, 5]
    colors = []
    for stamp, delta in enumerate(raw_deltas, start=1):
        values = BASELINE.copy()
        values[0] += delta
        colors.append(tuple(image(detector, values, time=stamp)[0, 0]))
    # The ten-count activation threshold and five-count release threshold
    # give net responses of 6, 30, 150, 360 and 600 counts while loading.
    assert colors[0] == colors[1] == colors[-1] == UNLOADED_RGB
    first = np.abs(np.subtract(UNLOADED_RGB, colors[2]))
    assert 0 < first.max() <= 2
    assert colors[3:7] == [(219, 231, 238), (201, 221, 234), (128, 199, 224), (0, 176, 205)]
    assert colors[7:11] == list(reversed(colors[2:6]))
    assert colors[-2] == WEAK_RGB


def test_faint_contact_preserves_the_same_surface_footprint_as_strong_contact():
    field = np.zeros((ROWS, COLS))
    field[2:6, 5:10] = 1
    field[3, 7] = 0
    field[1] = np.nan
    faint = contact_surface_rgb(field, DEFAULT_CONTACT_SPAN)
    strong = contact_surface_rgb(field * DEFAULT_CONTACT_SPAN, DEFAULT_CONTACT_SPAN)
    for color in (UNLOADED_RGB, INVALID_RGB):
        np.testing.assert_array_equal(np.all(faint == color, axis=2), np.all(strong == color, axis=2))


def test_surface_preserves_all_measured_centers_and_invalid_taxel_footprints():
    rng = np.random.default_rng(17)
    amplitude = rng.integers(0, 6000, size=(ROWS, COLS)).astype(float)
    amplitude[5::3, 8::4] = np.nan
    amplitude[::2, ::3] = 0
    native = contact_rgb({"reference_ready": True, "values": amplitude,
                          "valid_mask": np.isfinite(amplitude), "levels": (0, DEFAULT_CONTACT_SPAN)})
    surface = contact_surface_rgb(amplitude, DEFAULT_CONTACT_SPAN)
    assert surface.shape == (300, 300, 3) and surface.dtype == np.uint8 and surface.flags.c_contiguous
    center = SURFACE_SCALE // 2
    np.testing.assert_array_equal(surface[center::SURFACE_SCALE, center::SURFACE_SCALE], native)
    missing = np.repeat(np.repeat(np.isnan(amplitude), SURFACE_SCALE, axis=0), SURFACE_SCALE, axis=1)
    assert np.all(surface[missing] == INVALID_RGB)


def test_surface_uses_scalar_bilinear_weights_and_separate_half_coverage_contour():
    amplitude = np.zeros((ROWS, COLS))
    row, col = 5, 12
    amplitude[row, col] = 1500
    surface = contact_surface_rgb(amplitude, DEFAULT_CONTACT_SPAN)
    y, x = SURFACE_SCALE * row + 7, SURFACE_SCALE * col + 7

    # The inside sample has 1500 * (1-3/15) * (1-4/15) = 880 ADC counts.
    # Mapping those counts to RGB gives a point between cyan and green.
    np.testing.assert_array_equal(surface[y + 3, x + 4], (16, 176, 156))
    np.testing.assert_array_equal(surface[y + 5, x + 5], UNLOADED_RGB)
    np.testing.assert_array_equal(surface[x, y], UNLOADED_RGB)
    np.testing.assert_array_equal(surface[y, x], amplitude_rgb(amplitude, DEFAULT_CONTACT_SPAN)[row, col])


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


def test_outline_traces_outer_and_inner_edges_without_recoloring_taxel_centers_or_closing_a_gap():
    thresholds = np.arange(1, CELL_COUNT + 1, dtype=float).reshape(ROWS, COLS)
    ring = np.zeros((ROWS, COLS), dtype=bool)
    ring[4:11, 5:12] = True
    ring[5:10, 6:11] = False
    response = np.where(ring, 2 * thresholds, 0)
    assert not surface_contact_outline(response, thresholds).any()
    response[7, 5] = 8 * thresholds[7, 5]
    line = surface_contact_outline(response, thresholds)
    assert line.shape == (ROWS * SURFACE_SCALE, COLS * SURFACE_SCALE) and line.dtype == bool
    # Both the outside and inside of the top edge are visible.
    center = SURFACE_SCALE // 2
    column = 8 * SURFACE_SCALE + center
    assert line[4 * SURFACE_SCALE, column] and line[5 * SURFACE_SCALE - 1, column]
    assert not line[center::SURFACE_SCALE, center::SURFACE_SCALE].any()
    # Absolute counts and the RGB color span do not define this membership.
    stronger = np.where(ring, thresholds * 16, 0)
    np.testing.assert_array_equal(line, surface_contact_outline(stronger, thresholds))
    response[4, 8] = 0
    opened = surface_contact_outline(response, thresholds)
    assert opened.any() and not opened[4 * SURFACE_SCALE:5 * SURFACE_SCALE, column].any()
    assert not surface_contact_outline(np.zeros_like(response), thresholds).any()


def test_outline_grows_through_shared_sides_only_and_retains_separate_strong_contacts():
    thresholds = np.full((ROWS, COLS), 20.)
    response = np.zeros_like(thresholds)
    response[3, 3] = 160
    response[5, 5] = 160
    separate = surface_contact_outline(response, thresholds)
    assert separate[3 * SURFACE_SCALE:4 * SURFACE_SCALE, 3 * SURFACE_SCALE:4 * SURFACE_SCALE].any()
    assert separate[5 * SURFACE_SCALE:6 * SURFACE_SCALE, 5 * SURFACE_SCALE:6 * SURFACE_SCALE].any()
    response[4, 4] = 40
    np.testing.assert_array_equal(surface_contact_outline(response, thresholds), separate)
    response[4, 3] = 40
    assert not np.array_equal(surface_contact_outline(response, thresholds), separate)
    thresholds[4, 3] = 21
    np.testing.assert_array_equal(surface_contact_outline(response, thresholds), separate)


@pytest.mark.parametrize("missing", ["response", "reference"])
def test_outline_does_not_trace_sensor_edges_or_infer_contact_against_unknown_rows(missing):
    thresholds = np.full((ROWS, COLS), 10.)
    response = np.full_like(thresholds, 100.)
    if missing == "response":
        response[10, :] = np.nan
    else:
        thresholds[10, :] = 0
    assert not surface_contact_outline(response, thresholds).any()
    # With measured unloaded samples the transition has real support on both sides.
    thresholds[10, :] = 10
    response[10, :] = 0
    line = surface_contact_outline(response, thresholds)
    assert line[10 * SURFACE_SCALE - 1, 150] and line[11 * SURFACE_SCALE, 150]
    assert not line[[0, -1], :].any() and not line[:, [0, -1]].any()


def test_outline_requires_the_reference_to_address_the_same_grid():
    with pytest.raises(ValueError, match="20x20"):
        surface_contact_outline(np.zeros((ROWS, COLS)), np.ones(CELL_COUNT))
