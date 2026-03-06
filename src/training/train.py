"""
core/training/train.py

Training loop for MiniMedNet.
Supports:
    - Binary Cross-Entropy loss
    - Adam optimizer + ReduceLROnPlateau scheduler
    - Early stopping
    - Best-model checkpoint saving
    - Per-epoch metrics logging (loss, accuracy, F1)

Usage:
    python core/training/train.py --config config/training_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# ── path setup so we can import siblings ────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))

from models.mini_med_net import build_model
from training.dataset import build_dataloaders
from training.evaluate import compute_metrics


# ────────────────────────────────────────────────────────────────────────────
# Training utilities
# ────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 mode: str = "min") -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best_val   = float("inf") if mode == "min" else float("-inf")
        self.counter    = 0
        self.should_stop = False

    def __call__(self, val_metric: float) -> bool:
        improved = (
            (self.mode == "min" and val_metric < self.best_val - self.min_delta) or
            (self.mode == "max" and val_metric > self.best_val + self.min_delta)
        )
        if improved:
            self.best_val = val_metric
            self.counter  = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def _run_epoch(model: nn.Module,
               loader,
               criterion: nn.Module,
               device: torch.device,
               optimizer: torch.optim.Optimizer | None = None,
               phase: str = "train") -> Dict[str, float]:
    """
    One full pass over a DataLoader.

    Returns:
        dict with keys: loss, accuracy, precision, recall, f1
    """
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=f"  {phase}", leave=False, ncols=80):
            imgs, labels, _ = batch
            imgs   = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)   # (B, 1)

            if is_train:
                optimizer.zero_grad()

            logits = model(imgs)                       # (B, 1)
            loss   = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * imgs.size(0)

            probs  = torch.sigmoid(logits).detach().cpu().squeeze(1)
            preds  = (probs >= 0.5).int().tolist()
            lbls   = labels.detach().cpu().squeeze(1).int().tolist()
            all_preds.extend(preds)
            all_labels.extend(lbls)

    avg_loss = running_loss / len(loader.dataset)
    metrics  = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    return metrics


# ────────────────────────────────────────────────────────────────────────────
# Main training function
# ────────────────────────────────────────────────────────────────────────────

def train(config: Dict[str, Any]) -> None:
    """Full training run from a parsed config dict."""

    tc    = config["training"]
    model_cfg = config.get("model", {})

    # ── Device ──────────────────────────────────────────────────────────────
    device_str = model_cfg.get("inference", {}).get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"\n[Train] Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    print("[Train] Loading datasets...")
    train_loader, val_loader, _ = build_dataloaders(
        data_dir=tc["data_dir"],
        batch_size=tc["batch_size"],
        num_workers=tc.get("num_workers", 0),
        pin_memory=tc.get("pin_memory", False) and device.type == "cuda",
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_model(device=str(device))

    # ── Loss ────────────────────────────────────────────────────────────────
    # Optional class weighting for imbalance
    pos_weight = None
    if tc.get("use_class_weights", False):
        pw = tc.get("normal_weight", 1.5) / tc.get("pneumonia_weight", 1.0)
        pos_weight = torch.tensor([pw], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimizer ───────────────────────────────────────────────────────────
    optimizer = Adam(
        model.parameters(),
        lr=tc.get("learning_rate", 1e-4),
        weight_decay=tc.get("weight_decay", 1e-4),
    )

    # ── Scheduler ───────────────────────────────────────────────────────────
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=tc.get("scheduler_patience", 5),
        factor=tc.get("scheduler_factor", 0.5),
        min_lr=tc.get("min_lr", 1e-6),
    )

    # ── Early stopping ──────────────────────────────────────────────────────
    early_stopper = EarlyStopping(
        patience=tc.get("patience", 10),
        mode="min",
    ) if tc.get("early_stopping", True) else None

    # ── Output dirs ─────────────────────────────────────────────────────────
    ckpt_dir = Path(tc.get("checkpoint_dir", "models"))
    log_dir  = Path(tc.get("log_dir", "logs"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = ckpt_dir / "mini_med_net_best.pth"
    history   = []
    best_f1   = -1.0

    print(f"[Train] Starting for {tc['epochs']} epochs...\n")

    for epoch in range(1, tc["epochs"] + 1):
        t0 = time.time()

        train_m = _run_epoch(model, train_loader, criterion, device, optimizer, "train")
        val_m   = _run_epoch(model, val_loader,   criterion, device, phase="val")

        scheduler.step(val_m["loss"])
        lr_now = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0

        # ── Print epoch summary ──────────────────────────────────────────
        print(
            f"Epoch [{epoch:03d}/{tc['epochs']}] "
            f"Train Loss: {train_m['loss']:.4f}  Acc: {train_m['accuracy']:.3f}  F1: {train_m['f1']:.3f} | "
            f"Val   Loss: {val_m['loss']:.4f}  Acc: {val_m['accuracy']:.3f}  F1: {val_m['f1']:.3f} | "
            f"LR: {lr_now:.2e}  [{elapsed:.1f}s]"
        )

        # ── Save best checkpoint ─────────────────────────────────────────
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_f1,
                "val_loss": val_m["loss"],
            }, best_ckpt)
            print(f"  ✅ New best model saved (val_f1={best_f1:.4f})")

        # ── History ──────────────────────────────────────────────────────
        history.append({"epoch": epoch, "train": train_m, "val": val_m, "lr": lr_now})

        # ── Early stopping ───────────────────────────────────────────────
        if early_stopper and early_stopper(val_m["loss"]):
            print(f"\n[Train] Early stopping at epoch {epoch}.")
            break

    # ── Save training history ────────────────────────────────────────────────
    history_path = log_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Train] History saved → {history_path}")
    print(f"[Train] Best checkpoint → {best_ckpt}  (val_f1={best_f1:.4f})")


# ────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train MiniMedNet")
    parser.add_argument("--config", default="config/training_config.yaml",
                        help="Path to training_config.yaml")
    parser.add_argument("--model-config", default="config/model_config.yaml",
                        help="Path to model_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if Path(args.model_config).exists():
        with open(args.model_config, "r") as f:
            config["model"] = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
