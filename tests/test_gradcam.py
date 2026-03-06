"""
tests/test_gradcam.py

Unit tests for Grad-CAM heatmap generation and overlay utilities.
"""

import sys
from pathlib import Path
import numpy as np  # type: ignore
import pytest  # type: ignore
import torch  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from architectures.mini_med_net import MiniMedNet  # type: ignore
from explainability.gradcam import GradCAM, generate_gradcam  # type: ignore
from explainability.overlay import (  # type: ignore
    heatmap_to_colormap,
    overlay_heatmap_on_image,
    bgr_to_pil,
)


@pytest.fixture
def model_and_input():
    model  = MiniMedNet()
    model.eval()
    tensor = torch.randn(1, 3, 224, 224)
    return model, tensor


class TestGradCAM:
    """Tests for the GradCAM class and generate_gradcam wrapper."""

    def test_heatmap_shape(self, model_and_input):
        """Heatmap must match input spatial dimensions (224, 224)."""
        model, tensor = model_and_input
        heatmap = generate_gradcam(model, tensor)
        assert heatmap.shape == (224, 224), \
            f"Expected (224,224), got {heatmap.shape}"

    def test_heatmap_range(self, model_and_input):
        """Heatmap values must be in [0, 1]."""
        model, tensor = model_and_input
        heatmap = generate_gradcam(model, tensor)
        assert heatmap.min() >= 0.0, f"Min below 0: {heatmap.min()}"
        assert heatmap.max() <= 1.0, f"Max above 1: {heatmap.max()}"

    def test_heatmap_dtype(self, model_and_input):
        """Heatmap should be float32."""
        model, tensor = model_and_input
        heatmap = generate_gradcam(model, tensor)
        assert heatmap.dtype == np.float32, \
            f"Expected float32, got {heatmap.dtype}"

    def test_class_idx_0(self, model_and_input):
        """Heatmap generation should work for class_idx=0 (Normal)."""
        model, tensor = model_and_input
        heatmap = generate_gradcam(model, tensor, class_idx=0)
        assert heatmap.shape == (224, 224)

    def test_class_idx_1(self, model_and_input):
        """Heatmap generation should work for class_idx=1 (Pneumonia)."""
        model, tensor = model_and_input
        heatmap = generate_gradcam(model, tensor, class_idx=1)
        assert heatmap.shape == (224, 224)

    def test_hooks_removed_after_generate(self, model_and_input):
        """generate_gradcam should clean up its hooks."""
        model, tensor = model_and_input
        target = model.get_gradcam_target_layer()
        n_hooks_before = len(target._forward_hooks)
        generate_gradcam(model, tensor)
        n_hooks_after = len(target._forward_hooks)
        # Hooks must not increase permanently
        assert n_hooks_after <= n_hooks_before


class TestOverlay:
    """Tests for heatmap overlay utilities."""

    @pytest.fixture
    def sample_bgr(self):
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_heatmap(self):
        return np.random.rand(224, 224).astype(np.float32)

    def test_heatmap_to_colormap_shape(self, sample_heatmap):
        import cv2  # type: ignore
        colored = heatmap_to_colormap(sample_heatmap, cv2.COLORMAP_JET)
        assert colored.shape == (224, 224, 3)
        assert colored.dtype == np.uint8

    def test_overlay_output_shape(self, sample_bgr, sample_heatmap):
        overlay = overlay_heatmap_on_image(sample_bgr, sample_heatmap)
        assert overlay.shape == sample_bgr.shape

    def test_overlay_output_dtype(self, sample_bgr, sample_heatmap):
        overlay = overlay_heatmap_on_image(sample_bgr, sample_heatmap)
        assert overlay.dtype == np.uint8

    def test_bgr_to_pil_conversion(self, sample_bgr):
        pil = bgr_to_pil(sample_bgr)
        from PIL import Image  # type: ignore
        assert isinstance(pil, Image.Image)
        assert pil.mode == "RGB"
        assert pil.size == (224, 224)
