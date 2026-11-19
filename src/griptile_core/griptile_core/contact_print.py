"""Registered, three-exposure contact prints from measured response amplitudes."""

import numpy as np

from .contact_image import INVALID_RGB, SURFACE_SCALE, _bilinear
from .grid import SHAPE


_GROUND = np.array((13, 16, 19), dtype=np.float32)
_PAPER = np.array((248, 245, 235), dtype=np.float32)
_WHITE_POINT_ADC = 12000.
_TONE_KNEE_ADC = 900.
_TONE_DIVISOR = np.arcsinh(_WHITE_POINT_ADC / _TONE_KNEE_ADC)
_LEVELS_ADC = np.array((500, 850, 1500, 2400, 3500, 5200, 7500))
# Old, middle, newest. Coincident equal-density strokes add to neutral paper.
_INKS = np.array(((.09, .53, .73), (.24, .27, .23), (.67, .20, .04)), dtype=np.float32)


def _exposure(field):
    """Shade a proof on the common pixel grid and identify supported gradients."""
    valid = np.isfinite(field)
    surface = _bilinear(np.where(valid, field, 0.))
    supported = None
    if not valid.all():
        supported = _bilinear(~valid) == 0
        # Use the local measured pixel beside missing data, including its NaN.
        nearest = field.repeat(SURFACE_SCALE, axis=0).repeat(SURFACE_SCALE, axis=1)
        surface[~supported] = nearest[~supported]
    tone = np.arcsinh(np.minimum(np.nan_to_num(surface, nan=0.), _WHITE_POINT_ADC)
                      / _TONE_KNEE_ADC) / _TONE_DIVISOR
    proof = np.rint(_GROUND + tone[..., None] * (_PAPER - _GROUND)).astype(np.uint8)
    proof[~np.isfinite(surface)] = INVALID_RGB
    return surface, supported, proof


def _contour_ink(surface, supported):
    """Rasterize isolevels crossing adjacent output-pixel centers.

    Each linear crossing deposits ink in proportion to its distance from the
    two centers. When several levels cross one pixel pair, only the crossing
    nearest each center matters for that center's maximum ink coverage.
    """
    ink = np.zeros(surface.shape, dtype=np.float32)
    band = np.searchsorted(_LEVELS_ADC, surface, side="right")
    for axis in (0, 1):
        values = np.moveaxis(surface, axis, 0)
        bands = np.moveaxis(band, axis, 0)
        strokes = np.moveaxis(ink, axis, 0)
        crossed = bands[:-1] != bands[1:]
        if supported is not None:
            known = np.moveaxis(supported, axis, 0)
            crossed &= known[:-1] & known[1:]
        if not crossed.any():
            continue
        i, j = np.nonzero(crossed)
        start_band, end_band = bands[i, j], bands[i + 1, j]
        low = _LEVELS_ADC[np.minimum(start_band, end_band)]
        high = _LEVELS_ADC[np.maximum(start_band, end_band) - 1]
        start, end = values[i, j], values[i + 1, j]
        rising = end > start
        step = np.abs(end - start)
        start_coverage = 1 - np.abs(np.where(rising, low, high) - start) / step
        end_coverage = 1 - np.abs(np.where(rising, high, low) - end) / step
        strokes[i, j] = np.maximum(strokes[i, j], start_coverage)
        strokes[i + 1, j] = np.maximum(strokes[i + 1, j], end_coverage)
    return ink


def render_contact_print(frames):
    """Return a composite RGB print and its three monochrome RGB proofs.

    Input is three noise-gated 20×20 ADC response fields in chronological
    order. NaN denotes unavailable data; a composite pixel requires all three
    exposures. Outputs are uint8 arrays of shape (300, 300, 3) and
    (3, 300, 300, 3). Row zero is at the top, column zero at the left.
    """
    fields = np.asarray(frames)
    if fields.shape != (3, *SHAPE) or fields.dtype.kind not in "iuf":
        raise ValueError("a contact print needs three numeric 20×20 response fields")
    if np.any(np.isinf(fields)) or np.any(fields < 0):
        raise ValueError("response must contain nonnegative ADC counts or NaN")
    size = tuple(n * SURFACE_SCALE for n in SHAPE)
    proofs = np.empty((3, *size, 3), dtype=np.uint8)
    film = np.empty((*size, 3), dtype=np.float32)
    density = np.zeros_like(film)
    visible = np.ones(size, dtype=bool)
    for index, field in enumerate(fields):
        surface, supported, proof = _exposure(field)
        proofs[index] = proof
        channel = 2 - index  # The most recent exposure owns red, the oldest blue.
        film[..., channel] = proof[..., channel]
        visible &= np.isfinite(surface)
        density += _contour_ink(surface, supported)[..., None] * _INKS[index]
    base = _GROUND + .72 * (film - _GROUND)
    composite = np.rint(base + .98 * density * (_PAPER - base)).astype(np.uint8)
    composite[~visible] = INVALID_RGB
    return composite, proofs
