"""Registered contact imprints and fixed-scale RGB response images."""

import numpy as np

from .contact_image import (
    CONTACT_OUTLINE_RGB, DEFAULT_CONTACT_SPAN, INVALID_RGB, SURFACE_SCALE, UNLOADED_RGB,
    _bilinear, contact_surface_rgb, footprint_surface,
)
from .grid import SHAPE


_GROUND = np.array((13, 16, 19), dtype=np.float32)
_PAPER = np.array((248, 245, 235), dtype=np.float32)
# These levels reproduce the original plate at its nominal span and scale together.
_CONTOUR_LEVELS = np.array((500, 850, 1500, 2400, 3500, 5200, 7500)) / DEFAULT_CONTACT_SPAN
# Old, middle, latest. The three plates register to neutral ivory where they agree.
_INKS = np.array(((.09, .53, .73), (.24, .27, .23), (.67, .20, .04)), dtype=np.float32)


def _contour_ink(surface, supported, levels):
    """Rasterize isolevels crossing adjacent output-pixel centers.

    Each linear crossing deposits ink in proportion to its distance from the
    two centers. When several levels cross one pixel pair, only the crossing
    nearest each center matters for that center's maximum ink coverage.
    """
    ink = np.zeros(surface.shape, dtype=np.float32)
    band = np.searchsorted(levels, surface, side="right")
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
        low = levels[np.minimum(start_band, end_band)]
        high = levels[np.maximum(start_band, end_band) - 1]
        start, end = values[i, j], values[i + 1, j]
        rising = end > start
        step = np.abs(end - start)
        start_coverage = 1 - np.abs(np.where(rising, low, high) - start) / step
        end_coverage = 1 - np.abs(np.where(rising, high, low) - end) / step
        strokes[i, j] = np.maximum(strokes[i, j], start_coverage)
        strokes[i + 1, j] = np.maximum(strokes[i + 1, j], end_coverage)
    return ink


def _exposure(field, span_adc, levels):
    """Continuous response, ink coverage and support for a single plate."""
    valid = np.isfinite(field)
    surface = _bilinear(np.where(valid, field, 0.))
    supported = None
    if not valid.all():
        supported = _bilinear(~valid) == 0
        rows, cols = np.nonzero(~supported)
        surface[rows, cols] = np.nan_to_num(field)[rows // SURFACE_SCALE, cols // SURFACE_SCALE]
    # Common tone curve for all three exposures, including values above full color scale.
    tone = np.arcsinh(np.minimum(surface / span_adc, 4.) / .3) / np.arcsinh(4. / .3)
    return tone, _contour_ink(surface, supported, levels), supported


def _imprint(fields, span_adc, inside, outline):
    size = (*inside.shape, 3)
    face = np.broadcast_to(_GROUND, size).astype(np.uint8)
    if inside.any():
        tones = np.empty(size, dtype=np.float32)
        density = np.zeros(size, dtype=np.float32)
        comparable = np.ones(inside.shape, dtype=bool)
        levels = _CONTOUR_LEVELS * span_adc
        for i, field in enumerate(fields):
            tone, ink, supported = _exposure(field, span_adc, levels)
            if len(fields) == 1:
                tones[:] = tone[..., None]
                density[:] = ink[..., None]
                break
            tones[..., 2 - i] = tone
            density += ink[..., None] * _INKS[i]
            if supported is not None:
                comparable &= supported
        if not comparable.all():
            # A missing past exposure cannot imply a rise or fall. Use the
            # current exposure in neutral ink at those locations.
            tones[~comparable] = tone[~comparable, None]
            density[~comparable] = ink[~comparable, None]
        base = _GROUND + (.045 + .67 * tones) * (_PAPER - _GROUND)
        result = base + .98 * density * (_PAPER - base)
        # Give the independently measured external and hole edges a fine keyline,
        # even along weak contact too faint to cross the first engraved level.
        result[outline] += .24 * (_PAPER - result[outline])
        face[inside] = np.rint(result[inside]).astype(np.uint8)
    valid = np.isfinite(fields[-1])
    if not valid.all():
        face[~valid.repeat(SURFACE_SCALE, axis=0).repeat(SURFACE_SCALE, axis=1)] = INVALID_RGB
    return face


def _contact_face(fields, rgb, span_adc, footprint, style):
    if style not in ("imprint", "response"):
        raise ValueError("contact print style must be imprint or response")
    if footprint is None:
        return rgb.copy()
    valid = np.isfinite(fields[-1])
    inside, outline = footprint_surface(footprint, valid)
    if style == "imprint":
        return _imprint(fields, span_adc, inside, outline)
    face = rgb.copy()
    outside = ~inside
    if not valid.all():
        outside &= valid.repeat(SURFACE_SCALE, axis=0).repeat(SURFACE_SCALE, axis=1)
    face[outside] = UNLOADED_RGB
    face[outline] = CONTACT_OUTLINE_RGB
    return face


def render_contact_face(field, span_adc=DEFAULT_CONTACT_SPAN, footprint=None, *, style="response"):
    """Render a single measured contact face in response colors or neutral engraved ink."""
    rgb = contact_surface_rgb(field, span_adc)
    return _contact_face((field,), rgb, span_adc, footprint, style)


def render_contact_print(frames, span_adc=DEFAULT_CONTACT_SPAN, footprint=None, *, style="response"):
    """Show a contact face and three unmodified, chronological RGB response maps.

    In imprint style the three exposures also color the engraved contours inside
    the latest measured footprint. In response style only the latest frame
    colors the face. Without a footprint both styles fall back to the latest
    RGB response. Each input is a 20×20 noise-gated ADC field. NaN is unknown.
    All output images are row-major 300×300 RGB.
    """
    fields = np.asarray(frames)
    if fields.shape != (3, *SHAPE) or fields.dtype.kind not in "iuf":
        raise ValueError("a contact print needs three numeric 20x20 response fields")

    proofs = np.stack([contact_surface_rgb(field, span_adc) for field in fields])
    return _contact_face(fields, proofs[-1], span_adc, footprint, style), proofs
