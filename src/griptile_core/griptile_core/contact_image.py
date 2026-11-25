"""Color a tactile response at its native grid or as an interpolated display."""

import numpy as np

from .grid import COLS, ROWS


DEFAULT_CONTACT_SPAN = 3000.0  # Noise-gated ADC counts at full red.
INVALID_RGB = (128, 128, 128)
UNLOADED_RGB = (224, 233, 239)
CONTACT_OUTLINE_RGB = (58, 83, 97)
CONTACT_SEED_RATIO = 8.0       # Strong response, in multiples of each taxel's noise threshold.
CONTACT_SUPPORT_RATIO = 2.0    # Lower threshold for continuing a previously supported contact.
# The first active color is one red-channel code below unloaded. Even a
# sub-quantization response retains its contact flag in an RGB-only image.
CONTACT_COLOR_STOPS = (
    (0.0, (223, 233, 239)),
    (0.05, (201, 221, 234)),
    (0.12, (128, 199, 224)),
    (0.2, (0, 176, 205)),
    (0.4, (34, 176, 100)),
    (0.6, (246, 217, 60)),
    (0.8, (247, 139, 37)),
    (1.0, (220, 35, 38)),
)
WEAK_RGB = CONTACT_COLOR_STOPS[0][1]
STRONG_RGB = CONTACT_COLOR_STOPS[-1][1]
# Stops are fractions of the configured ADC span. Give low responses more
# blue contrast while keeping zero dark and increasing luminance at every stop.
CONTACT_MAP_STOPS = (
    (0., (11, 17, 28)), (.0025, (26, 51, 87)), (.01, (42, 90, 157)),
    (.05, (48, 110, 184)), (.15, (67, 131, 203)), (.30, (115, 140, 208)),
    (.50, (172, 132, 189)), (.75, (227, 125, 147)), (1., (255, 127, 102)),
)
SURFACE_SCALE = 15  # An odd scale puts every measured center exactly on a display pixel.


def color_scale_rgb(fractions, stops):
    """Map fractions of a fixed ADC span to RGB for measured pixels and their legend."""
    levels, colors = zip(*stops)
    channels = [np.interp(fractions, levels, channel) for channel in zip(*colors)]
    return np.rint(np.stack(channels, axis=-1)).astype(np.uint8)


def _color_response(amplitude, valid, active, span, stops=CONTACT_COLOR_STOPS, unloaded=UNLOADED_RGB):
    pixels = np.empty((*amplitude.shape, 3), dtype=np.uint8)
    pixels[:] = unloaded
    pixels[~valid] = INVALID_RGB
    pixels[active] = color_scale_rgb(amplitude[active] / span, stops)
    return pixels


def contact_amplitude(response) -> np.ndarray:
    """Return the native noise-gated ADC field, with NaN for unknown taxels.

    The field retains above-range counts. Only color rendering clips at the
    configured display span. It never aliases mutable acquisition data.
    """
    if not response["reference_ready"]:
        return np.full((ROWS, COLS), np.nan)
    valid = np.asarray(response["valid_mask"], dtype=bool).reshape(ROWS, COLS)
    values = np.asarray(response["values"], dtype=float).reshape(ROWS, COLS)
    return np.where(valid, values, np.nan)


def _check_span(span):
    if not np.isfinite(span) or span <= 0:
        raise ValueError("span must be finite and positive")


def _check_response(amplitude):
    values = np.asarray(amplitude, dtype=float)
    if np.any(np.isinf(values)) or np.any(values < 0):
        raise ValueError("amplitude must contain nonnegative counts or NaN")
    return values


def _check_field(amplitude):
    values = _check_response(amplitude)
    if values.shape != (ROWS, COLS):
        raise ValueError("amplitude must have the native 20x20 grid shape")
    return values


def contact_intensity_rgb(response_adc, span):
    """Return the native taxel color for a numeric readout or report."""
    _check_span(span)
    samples = _check_response([response_adc])
    if samples.shape != (1,):
        raise ValueError("expected one taxel response")
    rgb = _color_response(samples, np.isfinite(samples), samples > 0, span)
    return tuple(int(channel) for channel in rgb[0])


def amplitude_rgb(amplitude, span) -> np.ndarray:
    """Apply the fixed contact color scale to a native ADC response field."""
    _check_span(span)
    values = _check_field(amplitude)
    return _color_response(values, np.isfinite(values), values > 0, span)


def contact_rgb(response) -> np.ndarray:
    """Color the output of TactileResponseProcessor.transform at native resolution."""
    span = response["levels"][1] if response["reference_ready"] else DEFAULT_CONTACT_SPAN
    return amplitude_rgb(contact_amplitude(response), span)


def _sample_axis(count):
    offset = np.arange(count * SURFACE_SCALE) - SURFACE_SCALE // 2
    lower, remainder = np.divmod(np.clip(offset, 0, (count - 1) * SURFACE_SCALE), SURFACE_SCALE)
    return lower, np.minimum(lower + 1, count - 1), remainder / SURFACE_SCALE


_LEFT, _RIGHT, _DX = _sample_axis(COLS)
_TOP, _BOTTOM, _DY = _sample_axis(ROWS)


def _bilinear(field):
    """Interpolate between taxel centers, extending edge samples to the array boundary."""
    horizontal = field[:, _LEFT] * (1.0 - _DX) + field[:, _RIGHT] * _DX
    return horizontal[_TOP] * (1.0 - _DY[:, None]) + horizontal[_BOTTOM] * _DY[:, None]


def contact_surface_rgb(amplitude, span) -> np.ndarray:
    """Render a continuous view from 20x20 noise-gated ADC counts.

    NaN marks an unknown sample, zero an unloaded sample. The half-coverage
    contour of the active samples defines the footprint independently of their
    strength. A neighborhood containing unknown data uses its nearest taxel
    instead of estimating a gradient across the missing measurements.
    """
    return _surface_rgb(amplitude, span, CONTACT_COLOR_STOPS, UNLOADED_RGB)


def contact_map_rgb(contact, span) -> np.ndarray:
    """Color confirmed contact. Zero is quiet, NaN marks an unknown measurement."""
    return _surface_rgb(contact, span, CONTACT_MAP_STOPS, CONTACT_MAP_STOPS[0][1])


def _surface_rgb(amplitude, span, stops, unloaded):
    _check_span(span)
    values = _check_field(amplitude)
    valid = np.isfinite(values)
    active = values > 0
    # Handle uniform unloaded and unknown frames without interpolating an empty surface.
    if not active.any():
        native = np.where(valid[..., None], unloaded, INVALID_RGB).astype(np.uint8)
        return native.repeat(SURFACE_SCALE, axis=0).repeat(SURFACE_SCALE, axis=1)

    smooth = _bilinear(np.where(valid, values, 0))
    inside = _bilinear(active) >= 0.5
    visible = np.ones(inside.shape, dtype=bool)
    if not valid.all():
        # Fully unknown taxels stay gray across their whole footprint. On the
        # valid side of a missing edge, keep the known nearest sample.
        missing_neighbor = _bilinear(~valid) > 0
        rows, cols = np.nonzero(missing_neighbor)
        nearest = values[rows // SURFACE_SCALE, cols // SURFACE_SCALE]
        smooth[rows, cols] = nearest
        inside[rows, cols] = nearest > 0
        visible[rows, cols] = np.isfinite(nearest)
    return _color_response(smooth, visible, inside, span, stops, unloaded)


def _contact_region(amplitude, thresholds, active_taxels):
    values = _check_field(amplitude)
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.shape != values.shape:
        raise ValueError("contact thresholds must have the native 20x20 grid shape")
    valid = np.isfinite(values) & np.isfinite(thresholds) & (thresholds > 0)
    weak = valid & (values >= CONTACT_SUPPORT_RATIO * thresholds)
    region = valid & (values >= CONTACT_SEED_RATIO * thresholds)
    if active_taxels is not None:
        seeds = np.asarray(active_taxels)
        if seeds.shape != values.shape or seeds.dtype != np.bool_:
            raise ValueError("active taxels must be a boolean 20x20 grid")
        region |= seeds & weak

    while region.any():
        expanded = region.copy()
        expanded[1:] |= region[:-1]
        expanded[:-1] |= region[1:]
        expanded[:, 1:] |= region[:, :-1]
        expanded[:, :-1] |= region[:, 1:]
        expanded &= weak
        if np.array_equal(expanded, region):
            break
        region = expanded
    return region, valid


def contact_footprint(amplitude, thresholds, active_taxels=None):
    """Measured taxels connected by shared sides to a strong or still-active taxel.

    Supply the acquisition response and its own reference thresholds before any
    display equalization. Previously activated taxels must still meet the lower
    support threshold in this frame. Missing taxels never connect two regions.
    """
    region, _ = _contact_region(amplitude, thresholds, active_taxels)
    return region


def _all_four_neighbors(mask):
    """True where all four side neighbors of an interior pixel are true."""
    return mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]


def footprint_surface(region, valid):
    """Rasterize a native footprint and its measured boundary on the display grid.

    Half-coverage between neighboring taxel centers defines the displayed area.
    Along unknown cells, keep the nearest measurement and omit the adjoining
    boundary. The array edge clips the view without implying an object edge.
    """
    region = np.asarray(region)
    valid = np.asarray(valid)
    if any(mask.dtype != np.bool_ or mask.shape != (ROWS, COLS) for mask in (region, valid)):
        raise ValueError("footprint and validity must be boolean 20x20 grids")
    region = region & valid
    if not region.any() or region.all():
        size = ROWS * SURFACE_SCALE, COLS * SURFACE_SCALE
        return np.full(size, region[0, 0], dtype=bool), np.zeros(size, dtype=bool)
    inside = _bilinear(region) >= 0.5
    outline = np.zeros_like(inside)
    supported = None
    if not valid.all():
        supported = _bilinear(~valid) == 0
        missing_rows, missing_cols = np.nonzero(~supported)
        inside[missing_rows, missing_cols] = region[missing_rows // SURFACE_SCALE, missing_cols // SURFACE_SCALE]
    boundary = inside[1:-1, 1:-1] & ~_all_four_neighbors(inside)
    if supported is not None:
        boundary &= supported[1:-1, 1:-1] & _all_four_neighbors(supported)
    outline[1:-1, 1:-1] = boundary
    return inside, outline


def surface_contact_outline(amplitude, contour_thresholds) -> np.ndarray:
    """Return a one-pixel boundary on the same grid as the continuous RGB view.

    Both inputs are native 20×20 arrays. The reference supplies the activation
    threshold in ADC counts for each taxel; NaN or a nonpositive threshold
    marks an unsupported reference. The caller must pair it with the measured
    response from the same acquisition. No previous frames or color span enter.
    """
    region, valid = _contact_region(amplitude, contour_thresholds, None)
    _, outline = footprint_surface(region, valid)
    return outline
