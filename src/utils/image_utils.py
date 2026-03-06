"""
core/utils/image_utils.py

Image preprocessing and postprocessing utilities for Transparent Mini-Med.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ── ImageNet-normalisation constants ────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Default transform pipelines ─────────────────────────────────────────────

def get_inference_transform(input_size: int = 224) -> transforms.Compose:
    """Standard resize + centre-crop + normalise pipeline for inference."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_train_transform(input_size: int = 224) -> transforms.Compose:
    """Augmented transform pipeline for training."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_val_transform(input_size: int = 224) -> transforms.Compose:
    """Deterministic transform for validation / test sets."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ── Loading helpers ──────────────────────────────────────────────────────────

def load_image_pil(path: str | Path) -> Image.Image:
    """Load an image from disk as an RGB PIL Image."""
    return Image.open(str(path)).convert("RGB")


def load_image_cv2(path: str | Path) -> np.ndarray:
    """Load an image from disk as a BGR NumPy array (OpenCV format)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def pil_to_tensor(image: Image.Image,
                  transform: transforms.Compose | None = None) -> torch.Tensor:
    """
    Convert a PIL Image to a normalised (1, 3, H, W) tensor.

    Args:
        image: PIL Image (RGB).
        transform: torchvision transform pipeline. Uses inference default if None.

    Returns:
        Tensor of shape (1, 3, 224, 224).
    """
    if transform is None:
        transform = get_inference_transform()
    return transform(image).unsqueeze(0)


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a (C, H, W) or (1, C, H, W) normalised tensor back to a
    uint8 NumPy array in HWC BGR format (suitable for OpenCV display).
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # De-normalise
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1)

    # Convert to uint8 HWC BGR
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_bgr


def resize_to_original(heatmap: np.ndarray,
                        target_h: int,
                        target_w: int) -> np.ndarray:
    """Resize a heatmap (H, W) to (target_h, target_w)."""
    return cv2.resize(heatmap, (target_w, target_h))


def load_and_preprocess(path: str | Path,
                        input_size: int = 224
                        ) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
    """
    Load an image, apply inference transforms, and return:
        - tensor:       (1, 3, 224, 224) normalised torch.Tensor
        - orig_bgr:     original image as uint8 BGR NumPy array
        - orig_size:    (height, width) of the original image

    Args:
        path: Path to the image file.
        input_size: CNN input resolution.

    Returns:
        (tensor, orig_bgr, orig_size)
    """
    pil_img   = load_image_pil(path)
    orig_size = (pil_img.height, pil_img.width)
    orig_bgr  = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    transform = get_inference_transform(input_size)
    tensor    = pil_to_tensor(pil_img, transform)

    return tensor, orig_bgr, orig_size
