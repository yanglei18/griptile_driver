"""Three registered, fixed-scale RGB response images and a contoured view of the latest."""

import numpy as np

from .contact_image import DEFAULT_CONTACT_SPAN, SURFACE_SCALE, _bilinear, contact_surface_rgb
from .grid import SHAPE


CONTOUR_FRACTIONS = (.2, .4, .6, .8, 1.)


def _contour_ink(surface, supported, levels):
    """Deposit neutral ink at supported isolevel crossings on the display grid."""
    ink = np.zeros(surface.shape, dtype=np.float32)
    band = np.searchsorted(levels, surface, side="right")
    for axis in (0, 1):
        values = np.moveaxis(surface, axis, 0)
        bands = np.moveaxis(band, axis, 0)
        strokes = np.moveaxis(ink, axis, 0)
        known = np.moveaxis(supported, axis, 0)
        crossed = (bands[:-1] != bands[1:]) & known[:-1] & known[1:]
        if not crossed.any():
            continue
        i, j = np.nonzero(crossed)
        start_band, end_band = bands[i, j], bands[i + 1, j]
        low = levels[np.minimum(start_band, end_band)]
        high = levels[np.maximum(start_band, end_band) - 1]
        start, end = values[i, j], values[i + 1, j]
        rising = end > start
        step = np.abs(end - start)
        # For multiple crossings between the two pixels use the nearest to each center.
        start_coverage = 1 - np.abs(np.where(rising, low, high) - start) / step
        end_coverage = 1 - np.abs(np.where(rising, high, low) - end) / step
        strokes[i, j] = np.maximum(strokes[i, j], start_coverage)
        strokes[i + 1, j] = np.maximum(strokes[i + 1, j], end_coverage)
    return ink


def render_contact_print(frames, span_adc=DEFAULT_CONTACT_SPAN):
    """Return the latest response with contours and three chronological, unmarked exposures.

    Each 20×20 input is a noise-gated response in ADC units. All four color
    maps use the same supplied display span. NaN means unavailable in that
    particular exposure. Outputs are row-major uint8 RGB at 300×300 pixels.
    """
    fields = np.asarray(frames)
    if fields.shape != (3, *SHAPE) or fields.dtype.kind not in "iuf":
        raise ValueError("a contact print needs three numeric 20×20 response fields")

    proofs = np.stack([contact_surface_rgb(field, span_adc) for field in fields])
    current = fields[-1]
    levels = np.asarray(CONTOUR_FRACTIONS) * span_adc
    if not np.any(current >= levels[0]):
        return proofs[-1].copy(), proofs

    valid = np.isfinite(current)
    surface = _bilinear(np.where(valid, current, 0.))
    supported = _bilinear(current > 0) >= .5
    if not valid.all():
        supported &= _bilinear(~valid) == 0
    ink = _contour_ink(surface, supported, levels)
    # At a measured taxel center, retain exactly the native response color.
    center = SURFACE_SCALE // 2
    ink[center::SURFACE_SCALE, center::SURFACE_SCALE] = 0.
    master = np.rint(proofs[-1] * (1. - .36 * ink[..., None])).astype(np.uint8)
    return master, proofs
