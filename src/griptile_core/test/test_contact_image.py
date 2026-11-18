"""One physical taxel per output pixel, including contact and release."""

import numpy as np
import pytest

from griptile_core.contact_image import (
    DEFAULT_CONTACT_SPAN, INVALID_RGB, STRONG_RGB, UNLOADED_RGB, WEAK_RGB, contact_rgb,
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
