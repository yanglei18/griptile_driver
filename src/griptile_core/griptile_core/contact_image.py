"""Color a tactile response at its native grid or as an interpolated display."""

import numpy as np

from .grid import COLS, ROWS


DEFAULT_CONTACT_SPAN = 3000.0  # Noise-gated ADC counts at full red.
INVALID_RGB = (128, 128, 128)
UNLOADED_RGB = (255, 255, 255)
WEAK_RGB = (76, 42, 35)
STRONG_RGB = (255, 0, 0)
SURFACE_SCALE = 15  # An odd scale puts every measured center exactly on a display pixel.


def _color_response(amplitude, valid, active, span):
    pixels = np.empty((*amplitude.shape, 3), dtype=np.uint8)
    pixels[:] = UNLOADED_RGB
    pixels[~valid] = INVALID_RGB
    strength = np.minimum(amplitude[active] / span, 1.0)
    color_range = np.subtract(STRONG_RGB, WEAK_RGB)
    pixels[active] = np.rint(WEAK_RGB + strength[:, None] * color_range)
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


def _check_field(amplitude, span):
    values = np.asarray(amplitude, dtype=float)
    if values.shape != (ROWS, COLS):
        raise ValueError("amplitude must have the native 20x20 grid shape")
    if not np.isfinite(span) or span <= 0:
        raise ValueError("span must be finite and positive")
    if np.any(np.isinf(values)) or np.any(values < 0):
        raise ValueError("amplitude must contain nonnegative counts or NaN")
    return values


def amplitude_rgb(amplitude, span) -> np.ndarray:
    """Apply the fixed contact color scale to a native ADC response field."""
    values = _check_field(amplitude, span)
    return _color_response(values, np.isfinite(values), values > 0, span)


def contact_rgb(response) -> np.ndarray:
    """Color the output of TactileResponseProcessor.transform at native resolution."""
    span = response["levels"][1] if response["reference_ready"] else DEFAULT_CONTACT_SPAN
    return amplitude_rgb(contact_amplitude(response), span)


def _sample_axis(count):
    at = np.clip((np.arange(count * SURFACE_SCALE) - SURFACE_SCALE // 2) / SURFACE_SCALE,
                 0, count - 1)
    left = at.astype(int)
    t = at - left
    indexes = np.clip(left[:, None] + np.arange(-1, 3), 0, count - 1)
    # Cubic B-spline weights for coverage and interpolating Catmull-Rom weights
    # for measured intensity. Edges repeat the outermost taxel.
    spline = np.array([(1-t)**3, 4-6*t*t+3*t**3, 1+3*t+3*t*t-3*t**3, t**3]) / 6
    cardinal = np.array([-t*(1-t)**2, 2-5*t*t+3*t**3, t+4*t*t-3*t**3, -t*t*(1-t)]) / 2
    return indexes.T, spline, cardinal, t


_X, _X_SPLINE, _X_CARDINAL, _DX = _sample_axis(COLS)
_Y, _Y_SPLINE, _Y_CARDINAL, _DY = _sample_axis(ROWS)


def _resample(field, x_weights, y_weights):
    horizontal = sum(field[:, index] * weight for index, weight in zip(_X, x_weights))
    return sum(horizontal[index] * weight[:, None] for index, weight in zip(_Y, y_weights))


def _smoothstep(t):
    t = np.clip(t, 0, 1)
    return t*t*(3-2*t)


def _contact_opacity(active):
    # Compensate the coverage attenuation of isolated taxels at their centers.
    # A cubic B-spline sampled at an integer has weights [1, 4, 1] / 6.
    padded = np.pad(active.astype(float), 1, mode="edge")
    across = (padded[:, :-2] + 4*padded[:, 1:-1] + padded[:, 2:]) / 6
    center = (across[:-2] + 4*across[1:-1] + across[2:]) / 6
    coefficients = np.where(active, 1.0 / np.maximum(center, 4/9), 0.0)
    coverage = _resample(coefficients, _X_SPLINE, _Y_SPLINE)
    opacity = _smoothstep((coverage - .30) / .64)

    # Keep measured gaps open. At a checkerboard saddle, connect the two
    # unloaded corners. The two diagonally touching positives stay distinct.
    tl, tr = (active[_Y[1, :, None], x] for x in _X[1:3])
    bl, br = (active[_Y[2, :, None], x] for x in _X[1:3])
    dx, dy = _DX, _DY[:, None]
    support = ((1-dy) * ((1-dx)*tl + dx*tr) + dy * ((1-dx)*bl + dx*br))
    saddle = (tl == br) & (bl == tr) & (tl != tr)
    support = np.where(saddle, np.where(tl, abs(1-dx-dy), abs(dx-dy)), support)
    return np.minimum(opacity, _smoothstep(support / .65))


def contact_surface_rgb(amplitude, span) -> np.ndarray:
    """Render a feathered contact impression of the native 20x20 ADC response.

    Coverage and intensity are reconstructed separately before compositing on
    white. NaN taxels stay gray. Where a cubic footprint crosses missing data,
    display the nearest measurement instead of estimating across that gap.
    """
    values = _check_field(amplitude, span)
    valid = np.isfinite(values)
    active = values > 0
    if not active.any():
        native = np.where(valid[..., None], UNLOADED_RGB, INVALID_RGB).astype(np.uint8)
        return native.repeat(SURFACE_SCALE, axis=0).repeat(SURFACE_SCALE, axis=1)

    source = np.where(valid, values, 0.0)
    smooth = _resample(source, _X_CARDINAL, _Y_CARDINAL)
    # Bound cubic overshoot by the four bracketing measured amplitudes.
    corners = [source[y[:, None], x] for y in _Y[1:3] for x in _X[1:3]]
    smooth = np.clip(smooth, np.minimum.reduce(corners), np.maximum.reduce(corners))
    opacity = _contact_opacity(active)
    strength = np.minimum(smooth/span, 1.0)
    pixels = np.empty((*strength.shape, 3), dtype=np.uint8)
    for channel, (weak, strong) in enumerate(zip(WEAK_RGB, STRONG_RGB)):
        ink = weak + (strong-weak)*strength
        # A translucent red edge warms towards white; the native palette is
        # recovered exactly at full opacity.
        amount = opacity**2 if channel == 0 else opacity
        pixels[..., channel] = np.rint(255 + amount*(ink-255))
    if not valid.all():
        unsupported = _resample(~valid, _X_SPLINE, _Y_SPLINE) > 0
        native = amplitude_rgb(values, span)
        y, x = np.nonzero(unsupported)
        pixels[y, x] = native[y // SURFACE_SCALE, x // SURFACE_SCALE]
    return pixels
