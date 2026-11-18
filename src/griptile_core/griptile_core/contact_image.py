"""Native row-major RGB image of a noise-gated tactile response."""

import numpy as np

from .grid import COLS, ROWS


DEFAULT_CONTACT_SPAN = 3000.0  # Noise-gated ADC counts at full red.
INVALID_RGB = (128, 128, 128)
UNLOADED_RGB = (255, 255, 255)
WEAK_RGB = (76, 42, 35)
STRONG_RGB = (255, 0, 0)


def contact_rgb(response) -> np.ndarray:
    """Color the output of TactileResponseProcessor.transform at native resolution.

    Every valid inactive taxel is white. Any active response starts at dark
    brown and increases to saturated red at the configured upper ADC limit.
    Missing references and invalid measurements share a neutral gray.
    """
    pixels = np.full((ROWS, COLS, 3), INVALID_RGB, dtype=np.uint8)
    if not response["reference_ready"]:
        return pixels

    valid = np.asarray(response["valid_mask"], dtype=bool).reshape(ROWS, COLS)
    amplitude = np.asarray(response["values"], dtype=float).reshape(ROWS, COLS)
    pixels[valid] = UNLOADED_RGB
    active = valid & (amplitude > 0)
    strength = np.minimum(amplitude[active] / response["levels"][1], 1.0)
    color_range = np.subtract(STRONG_RGB, WEAK_RGB)
    pixels[active] = np.rint(WEAK_RGB + strength[:, None] * color_range)
    return pixels
