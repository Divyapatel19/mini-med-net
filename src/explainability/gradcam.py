"""
core/explainability/gradcam.py

Grad-CAM (Gradient-weighted Class Activation Mapping) for MiniMedNet.

Grad-CAM Algorithm:
    1. Forward pass → obtain class score (logit)
    2. Backward pass w.r.t. the target class → gradients at conv target layer
    3. Global-Average-Pool the gradients → neuron importance weights (α)
    4. Weighted combination of feature maps + ReLU → saliency map
    5. Resize saliency map to input resolution

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization," ICCV 2017.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM generator for a PyTorch model.

    Args:
        model:        The nn.Module to explain (e.g., MiniMedNet).
        target_layer: The conv layer to hook for activations and gradients.
                      For MiniMedNet pass `model.conv4.block[0]`.

    Example:
        gradcam  = GradCAM(model, model.conv4.block[0])
        heatmap  = gradcam.generate(input_tensor)  # (H, W) float32 in [0,1]
        gradcam.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model        = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    # ── Hooks ──────────────────────────────────────────────────────────────

    def _save_activations(self, module, input, output) -> None:  # noqa: ARG002
        """Store forward activations."""
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:  # noqa: ARG002
        """Store gradients flowing back through the target layer."""
        self._gradients = grad_output[0].detach()

    # ── Core ───────────────────────────────────────────────────────────────

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed (1, C, H, W) image tensor. Requires grad.
            class_idx: Target class index. If None, uses the predicted class.

        Returns:
            heatmap: float32 ndarray of shape (H, W) with values in [0, 1].
                     Resize as needed before overlaying on the original image.
        """
        self.model.eval()
        input_tensor  = input_tensor.clone().requires_grad_(True)
        device        = next(self.model.parameters()).device
        input_tensor  = input_tensor.to(device)

        # Forward
        logits = self.model(input_tensor)             # (1, 1)
        prob   = torch.sigmoid(logits)

        # Determine target class
        if class_idx is None:
            class_idx = int((prob.item() >= 0.5))     # 1 → Pneumonia, 0 → Normal

        # Score to back-prop (binary: use the logit itself)
        score = logits[0, 0]

        # Backward
        self.model.zero_grad()
        score.backward()

        # ── Grad-CAM computation ─────────────────────────────────────────
        # activations: (1, C, h, w)
        # gradients:   (1, C, h, w)
        activations = self._activations          # (1, C, h, w)
        gradients   = self._gradients            # (1, C, h, w)

        # α_c = mean of gradients over spatial dims → (1, C, 1, 1)
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of feature maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)

        # ReLU — keep only positive influence
        cam = F.relu(cam)

        # Resize to input resolution
        _, _, H, W = input_tensor.shape
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()        # (H, W)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    def remove_hooks(self) -> None:
        """Remove forward/backward hooks (call when done to avoid memory leaks)."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __del__(self) -> None:
        try:
            self.remove_hooks()
        except Exception:
            pass


# ── Convenience wrapper ───────────────────────────────────────────────────────

def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: Optional[int] = None,
) -> np.ndarray:
    """
    One-shot Grad-CAM generation (handles hook lifecycle internally).

    Args:
        model:        MiniMedNet instance (or any model with get_gradcam_target_layer()).
        input_tensor: (1, C, H, W) preprocessed tensor.
        class_idx:    Target class (None = predicted class).

    Returns:
        heatmap: float32 ndarray (H, W) normalised to [0, 1].
    """
    target_layer = model.get_gradcam_target_layer()
    gcam = GradCAM(model, target_layer)
    try:
        heatmap = gcam.generate(input_tensor, class_idx=class_idx)
    finally:
        gcam.remove_hooks()
    return heatmap
