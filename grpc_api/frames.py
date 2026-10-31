"""Render capture for the Minigrid gRPC API.

Turns the environment's rgb_array render into a PNG, with the same
orientation the human window shows in manual_control.py. The env must have
been created with render_mode="rgb_array"; the server never touches
pygame.display, so it is safe to run headless.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def ndarray_to_png(img: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 RGB array as PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(img, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def capture_frame(env) -> tuple[bytes, int, int]:
    """Render the env to PNG bytes, oriented like manual_control's window.

    minigrid's `render()` in rgb_array mode skips the transpose that the
    human branch applies (minigrid_env.py: `np.transpose(img, (1, 0, 2))` is
    only in the human path), so we apply it here — otherwise frames appear
    rotated 90 degrees relative to manual_control.py.

    Works for both the full top-down render and the agent-POV render, since
    `render()` dispatches on `agent_pov` set at env construction time.
    """
    img = np.transpose(env.render(), axes=(1, 0, 2))
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    height, width = img.shape[:2]
    return ndarray_to_png(img), width, height
