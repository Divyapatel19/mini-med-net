"""
core/models/mini_med_net.py

MiniMedNet — A lightweight CNN (~1.5M parameters) for binary medical image
classification (Pneumonia vs. Normal from chest X-rays).

Architecture:
    Conv1 (32) → BN → ReLU → MaxPool 2×2
    Conv2 (64) → BN → ReLU → MaxPool 2×2
    Conv3 (128) → BN → ReLU → MaxPool 2×2
    Conv4 (256) → BN → ReLU → GlobalAvgPool
    Dropout → FC(256→1) → Sigmoid

The last conv block (conv4) is named so Grad-CAM can hook onto it easily.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class MiniMedNet(nn.Module):
    """
    Robust medical image classifier using a ResNet18 backbone.
    Improved accuracy via ImageNet pre-trained feature extraction.

    Input:  (B, 3, 224, 224)
    Output: (B, 1)  — raw logit (apply sigmoid for probability)
    """

    def __init__(self, dropout_rate: float = 0.5) -> None:
        super().__init__()
        
        # Load pre-trained ResNet18
        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Capture the number of input features for the final FC layer
        num_ftrs = self.backbone.fc.in_features
        
        # Replace the original FC layer with a medical-specific head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_target_layer(self) -> nn.Module:
        """Returns the last convolutional layer block (layer4)."""
        # In ResNet, layer4 is the last group of conv blocks
        return self.backbone.layer4

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(dropout_rate: float = 0.5,
                weights_path: str | None = None,
                device: str = "cpu") -> MiniMedNet:
    """
    Instantiate MiniMedNet, optionally loading saved weights.

    Args:
        dropout_rate: Dropout probability.
        weights_path: Path to a .pth file with state_dict. If None, random init.
        device: 'cpu', 'cuda', or 'auto'.

    Returns:
        MiniMedNet model in eval mode (if weights loaded).
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MiniMedNet(dropout_rate=dropout_rate).to(device)

    if weights_path is not None:
        state = torch.load(weights_path, map_location=device)
        # Support both raw state_dict and checkpoint dicts
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval()
        print(f"[MiniMedNet] Loaded weights from: {weights_path}")
    else:
        print(f"[MiniMedNet] Initialized with random weights. Params: {model.num_parameters:,}")

    return model
