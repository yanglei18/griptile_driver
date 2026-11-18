"""Color a tactile response at its native grid or as an interpolated display."""

import numpy as np

from .grid import COLS, ROWS


DEFAULT_CONTACT_SPAN = 3000.0  # Noise-gated ADC counts at full red.
INVALID_RGB = (128, 128, 128)
UNLOADED_RGB = (255, 255, 255)
# Fractions of the configured display span, for positive responses only.
# Zero response and missing measurements have their own colors above.
CONTACT_COLOR_STOPS = (
    (0.0, (36, 64, 164)),
    (0.2, (0, 176, 205)),
    (0.4, (34, 176, 100)),
    (0.6, (246, 217, 60)),
    (0.8, (247, 139, 37)),
    (1.0, (220, 35, 38)),
)
WEAK_RGB = CONTACT_COLOR_STOPS[0][1]
STRONG_RGB = CONTACT_COLOR_STOPS[-1][1]
_STOP_LEVELS, _STOP_COLORS = zip(*CONTACT_COLOR_STOPS)
_CHANNELS = tuple(zip(*_STOP_COLORS))
SURFACE_SCALE = 15  # An odd scale puts every measured center exactly on a display pixel.


def _color_response(amplitude, valid, active, span):
    pixels = np.empty((*amplitude.shape, 3), dtype=np.uint8)
    pixels[:] = UNLOADED_RGB
    pixels[~valid] = INVALID_RGB
    strength = amplitude[active] / span
    channels = [np.interp(strength, _STOP_LEVELS, channel) for channel in _CHANNELS]
    pixels[active] = np.rint(np.column_stack(channels))
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


def _check_response(amplitude, span):
    values = np.asarray(amplitude, dtype=float)
    if not np.isfinite(span) or span <= 0:
        raise ValueError("span must be finite and positive")
    if np.any(np.isinf(values)) or np.any(values < 0):
        raise ValueError("amplitude must contain nonnegative counts or NaN")
    return values


def _check_field(amplitude, span):
    values = _check_response(amplitude, span)
    if values.shape != (ROWS, COLS):
        raise ValueError("amplitude must have the native 20x20 grid shape")
    return values


def contact_intensity_rgb(response_adc, span):
    """Return the native taxel color for a numeric readout or report."""
    samples = _check_response([response_adc], span)
    if samples.shape != (1,):
        raise ValueError("expected one taxel response")
    rgb = _color_response(samples, np.isfinite(samples), samples > 0, span)
    return tuple(int(channel) for channel in rgb[0])


def amplitude_rgb(amplitude, span) -> np.ndarray:
    """Apply the fixed contact color scale to a native ADC response field."""
    values = _check_field(amplitude, span)
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
    values = _check_field(amplitude, span)
    valid = np.isfinite(values)
    active = values > 0
    # Handle uniform unloaded and unknown frames without interpolating an empty surface.
    if not active.any():
        native = np.where(valid[..., None], UNLOADED_RGB, INVALID_RGB).astype(np.uint8)
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
    return _color_response(smooth, visible, inside, span)
