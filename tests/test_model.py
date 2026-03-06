"""
tests/test_model.py

Unit tests for MiniMedNet architecture.
"""

import sys
from pathlib import Path
import pytest  # type: ignore
import torch  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from architectures.mini_med_net import MiniMedNet, build_model  # type: ignore


class TestMiniMedNet:
    """Tests for the MiniMedNet CNN."""
    model: MiniMedNet
    dummy: torch.Tensor

    def setup_method(self):
        self.model = MiniMedNet()
        self.model.eval()
        self.dummy = torch.randn(2, 3, 224, 224)

    def test_output_shape(self):
        """Forward pass should produce (B, 1) logits."""
        logits = self.model(self.dummy)
        assert logits.shape == (2, 1), f"Expected (2,1), got {logits.shape}"

    def test_output_range_after_sigmoid(self):
        """Sigmoid-activated output should be in [0, 1]."""
        logits = self.model(self.dummy)
        probs  = torch.sigmoid(logits)
        assert probs.min().item() >= 0.0, "Probability below 0"
        assert probs.max().item() <= 1.0, "Probability above 1"

    def test_single_image_inference(self):
        """Single image (B=1) inference should work."""
        x = torch.randn(1, 3, 224, 224)
        out = self.model(x)
        assert out.shape == (1, 1)

    def test_parameter_count(self):
        """Model should have a reasonable number of parameters (<= 15M)."""
        n_params = self.model.num_parameters
        assert n_params < 15_000_000, f"Too many params: {n_params:,}"
        assert n_params > 100_000,   f"Too few params: {n_params:,}"
        print(f"\n  MiniMedNet params: {n_params:,}")

    def test_gradcam_target_layer(self):
        """get_gradcam_target_layer() should return an nn.Sequential (layer4 in ResNet)."""
        import torch.nn as nn  # type: ignore
        layer = self.model.get_gradcam_target_layer()
        assert isinstance(layer, nn.Sequential), \
            f"Expected Sequential, got {type(layer)}"

    def test_build_model_no_weights(self):
        """build_model() without weights returns an untrained model."""
        m = build_model(device="cpu")
        assert m is not None
        x = torch.randn(1, 3, 224, 224)
        out = m(x)
        assert out.shape == (1, 1)

    def test_model_train_eval_modes(self):
        """Confirm we can switch between train and eval modes."""
        self.model.train()
        assert self.model.training
        self.model.eval()
        assert not self.model.training

    def test_gradient_flow(self):
        """Gradients should flow through all parameters in train mode."""
        model = MiniMedNet()
        model.train()
        x = torch.randn(2, 3, 224, 224)
        loss = model(x).sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
