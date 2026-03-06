"""
core/training/dataset.py

ChestXRayDataset — PyTorch Dataset for the Kaggle Chest X-Ray (Pneumonia) dataset.
Expects the standard folder structure:

    root/
    ├── NORMAL/
    │   ├── img1.jpeg
    │   └── ...
    └── PNEUMONIA/
        ├── img1.jpeg
        └── ...

Labels:   NORMAL = 0,  PNEUMONIA = 1
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from utils.image_utils import get_train_transform, get_val_transform


class ChestXRayDataset(Dataset):
    """
    Dataset for binary classification of chest X-rays.

    Args:
        root_dir:  Path to the split directory (train / val / test).
        transform: Optional torchvision transform. Falls back to a sensible default.
        augment:   If True and transform is None, applies training augmentation.

    Returns (per item):
        image  (torch.Tensor): normalised (3, 224, 224) tensor
        label  (torch.Tensor): scalar float — 0.0 (Normal) or 1.0 (Pneumonia)
        path   (str):          path of the source file (for debugging)
    """

    CLASSES = {"NORMAL": 0, "PNEUMONIA": 1}

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
        augment: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self._validate_structure()

        # Build (path, label) list
        self.samples: list[Tuple[Path, int]] = []
        for class_name, label in self.CLASSES.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for ext in ("*.jpeg", "*.jpg", "*.png"):
                for img_path in class_dir.glob(ext):
                    self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.root_dir}. "
                               "Check NORMAL/ and PNEUMONIA/ subdirectories.")

        # Assign transform
        if transform is not None:
            self.transform = transform
        elif augment:
            self.transform = get_train_transform()
        else:
            self.transform = get_val_transform()

    def _validate_structure(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32), str(img_path)

    def class_counts(self) -> dict[str, int]:
        """Return {class_name: count} for the loaded split."""
        counts: dict[str, int] = {"NORMAL": 0, "PNEUMONIA": 0}
        inv = {v: k for k, v in self.CLASSES.items()}
        for _, label in self.samples:
            counts[inv[label]] += 1
        return counts

    def compute_weights(self) -> list[float]:
        """
        Per-sample weights for WeightedRandomSampler (handles class imbalance).
        Minority class gets a higher weight.
        """
        counts = self.class_counts()
        total  = sum(counts.values())
        class_w = {cls: total / count for cls, count in counts.items()}
        inv = {v: k for k, v in self.CLASSES.items()}
        return [class_w[inv[lbl]] for _, lbl in self.samples]


# ── DataLoader factory ───────────────────────────────────────────────────────

def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train / val / test DataLoaders from the dataset directory.

    Args:
        data_dir: Root directory containing train/, val/, test/ sub-folders.
        batch_size: Mini-batch size.
        num_workers: DataLoader worker processes.
        pin_memory: Pin tensors to pinned memory (faster GPU transfer).
        use_weighted_sampler: Balance training mini-batches via WeightedRandomSampler.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)

    train_ds = ChestXRayDataset(data_dir / "train", augment=True)
    val_ds   = ChestXRayDataset(data_dir / "val",   augment=False)
    test_ds  = ChestXRayDataset(data_dir / "test",  augment=False)

    # Weighted sampler for training
    if use_weighted_sampler:
        weights = train_ds.compute_weights()
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    _print_split_info(train_ds, val_ds, test_ds)
    return train_loader, val_loader, test_loader


def _print_split_info(*datasets: ChestXRayDataset) -> None:
    names = ["Train", "Val", "Test"]
    for name, ds in zip(names, datasets):
        counts = ds.class_counts()
        print(f"  {name:5s}: {len(ds):5d} images  "
              f"(Normal: {counts['NORMAL']}, Pneumonia: {counts['PNEUMONIA']})")
