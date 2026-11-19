"""A measured contact footprint and three registered, fixed-scale RGB response images."""

import numpy as np

from .contact_image import (
    CONTACT_OUTLINE_RGB, DEFAULT_CONTACT_SPAN, SURFACE_SCALE, UNLOADED_RGB,
    contact_surface_rgb, footprint_surface,
)
from .grid import SHAPE


def _contact_face(field, rgb, footprint):
    if footprint is None:
        return rgb.copy()
    valid = np.isfinite(field)
    inside, outline = footprint_surface(footprint, valid)
    face = rgb.copy()
    outside = ~inside
    if not valid.all():
        outside &= valid.repeat(SURFACE_SCALE, axis=0).repeat(SURFACE_SCALE, axis=1)
    face[outside] = UNLOADED_RGB
    face[outline] = CONTACT_OUTLINE_RGB
    return face


def render_contact_face(field, span_adc=DEFAULT_CONTACT_SPAN, footprint=None):
    """Show the supported shape of one source frame, with response colors inside."""
    return _contact_face(field, contact_surface_rgb(field, span_adc), footprint)


def render_contact_print(frames, span_adc=DEFAULT_CONTACT_SPAN, footprint=None):
    """Show the latest contact area and the full response of each of the three exposures.

    Each 20×20 input is a noise-gated ADC response. The optional native boolean
    footprint is measured before display equalization at the final frame. If
    absent, the latest response is shown without claiming a contact boundary.
    A NaN is unknown in that exposure. All outputs are row-major 300×300 RGB.
    """
    fields = np.asarray(frames)
    if fields.shape != (3, *SHAPE) or fields.dtype.kind not in "iuf":
        raise ValueError("a contact print needs three numeric 20×20 response fields")

    proofs = np.stack([contact_surface_rgb(field, span_adc) for field in fields])
    return _contact_face(fields[-1], proofs[-1], footprint), proofs
