"""
core/explainability/overlay.py

Utilities to overlay Grad-CAM heatmaps on original images, producing the
visual explanations shown in the clinical UI.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def heatmap_to_colormap(
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Convert a float32 heatmap (H, W) in [0,1] to a BGR colormap image.

    Args:
        heatmap:  float32 ndarray (H, W) with values in [0, 1].
        colormap: OpenCV colormap constant (default: cv2.COLORMAP_JET).

    Returns:
        Coloured heatmap as uint8 BGR (H, W, 3).
    """
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    return cv2.applyColorMap(heatmap_uint8, colormap)


def overlay_heatmap_on_image(
    orig_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap over the original image.

    Args:
        orig_bgr:  Original image as uint8 BGR (H, W, 3).
        heatmap:   Float32 (H, W) Grad-CAM map in [0, 1].
        alpha:     Heatmap transparency (0 = invisible, 1 = fully opaque).
        colormap:  OpenCV colormap for false-color rendering.

    Returns:
        Blended overlay as uint8 BGR (H, W, 3).
    """
    # Resize heatmap to match original image
    h, w = orig_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    colored = heatmap_to_colormap(heatmap_resized, colormap)

    # Blend
    overlay = cv2.addWeighted(orig_bgr, 1 - alpha, colored, alpha, 0)
    return overlay


def create_side_by_side(
    orig_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    titles: Tuple[str, str] = ("Original", "Grad-CAM Overlay"),
    font_scale: float = 0.65,
) -> np.ndarray:
    """
    Stack the original and overlay side-by-side with labels.

    Args:
        orig_bgr:    Original image BGR uint8.
        overlay_bgr: Overlay image BGR uint8.
        titles:      Text labels for each panel.
        font_scale:  OpenCV font size.

    Returns:
        Combined image (H, W*2, 3) uint8.
    """
    h, w = orig_bgr.shape[:2]
    combined = np.zeros((h + 30, w * 2 + 10, 3), dtype=np.uint8)

    # Place images
    combined[30:30 + h, :w] = orig_bgr
    combined[30:30 + h, w + 10:] = overlay_bgr

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, titles[0], (10, 22), font, font_scale, (200, 200, 200), 1)
    cv2.putText(combined, titles[1], (w + 20, 22), font, font_scale, (200, 200, 200), 1)

    return combined


def bgr_to_rgb_array(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 to RGB uint8 (for matplotlib / PIL display)."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert BGR uint8 ndarray to a PIL RGB Image."""
    return Image.fromarray(bgr_to_rgb_array(bgr))


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL RGB Image to a BGR uint8 ndarray."""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
