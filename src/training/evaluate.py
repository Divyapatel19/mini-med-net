"""
core/training/evaluate.py

Evaluation utilities for MiniMedNet:
    - compute_metrics:       accuracy, precision, recall, F1, ROC-AUC
    - evaluate_on_dataset:   full-pass over a DataLoader
    - plot_confusion_matrix: save/display confusion matrix figure
    - plot_roc_curve:        save/display ROC-AUC figure

Usage (CLI):
    python src/training/evaluate.py \
        --weights models/mini_med_net_best.pth \
        --data-dir data/chest_xrays/test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import seaborn as sns  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm  # type: ignore

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from architectures.mini_med_net import build_model  # type: ignore
from training.dataset import ChestXRayDataset  # type: ignore
from torch.utils.data import DataLoader  # type: ignore


# ────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: List[float] | None = None,
) -> Dict[str, float]:
    """
    Compute classification metrics from integer lists.

    Args:
        y_true: Ground-truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1, threshold=0.5).
        y_prob: Predicted probabilities [0, 1]. Required for ROC-AUC.

    Returns:
        dict: accuracy, precision, recall, f1, (roc_auc if y_prob provided)
    """
    metrics: Dict[str, float] = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true,    y_pred, zero_division=0),
        "f1":        f1_score(y_true,        y_pred, zero_division=0),
    }
    if y_prob is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    return metrics


# ────────────────────────────────────────────────────────────────────────────
# Full evaluation pass
# ────────────────────────────────────────────────────────────────────────────

def evaluate_on_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Run model on all batches in loader and compute metrics.

    Returns:
        dict with keys: accuracy, precision, recall, f1, roc_auc,
                        y_true, y_pred, y_prob
    """
    model.eval()
    all_probs:  List[float] = []
    all_preds:  List[int]   = []
    all_labels: List[int]   = []

    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc="  Evaluating", ncols=80, leave=False):
            imgs   = imgs.to(device)
            logits = model(imgs)
            probs  = torch.sigmoid(logits).cpu().squeeze(1).tolist()
            preds  = [1 if p >= threshold else 0 for p in probs]
            lbls   = labels.int().tolist()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(lbls)

    base_metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics: Dict[str, Any] = {}
    metrics.update(base_metrics)
    metrics["y_true"] = all_labels
    metrics["y_pred"] = all_preds
    metrics["y_prob"] = all_probs
    return metrics


# ────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    save_path: str | Path | None = "outputs/confusion_matrix.png",
) -> None:
    """Plot and optionally save the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Normal", "Pneumonia"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title("Confusion Matrix — MiniMedNet", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  Confusion matrix saved → {save_path}")
    plt.show()


def plot_roc_curve(
    y_true: List[int],
    y_prob: List[float],
    save_path: str | Path | None = "outputs/roc_curve.png",
) -> None:
    """Plot and optionally save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2,
            label=f"ROC Curve (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#d62728", lw=1.5, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — MiniMedNet", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  ROC curve saved → {save_path}")
    plt.show()


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MiniMedNet on a test set")
    parser.add_argument("--weights", required=True, help="Path to .pth weights file")
    parser.add_argument("--data-dir", required=True, help="Path to test data directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving plots")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Evaluate] Device: {device}")

    # Load model
    model = build_model(weights_path=args.weights, device=str(device))

    # DataLoader
    ds = ChestXRayDataset(args.data_dir, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Evaluate
    results = evaluate_on_dataset(model, loader, device, threshold=args.threshold)

    # Print metrics
    print("\n── Evaluation Results ──────────────────────────────────")
    for k, v in results.items():
        if k not in ("y_true", "y_pred", "y_prob"):
            print(f"  {k:12s}: {v:.4f}")
    print("────────────────────────────────────────────────────────")

    # Plots
    if not args.no_plots:
        out = Path(args.output_dir)
        plot_confusion_matrix(results["y_true"], results["y_pred"],
                              save_path=out / "confusion_matrix.png")
        plot_roc_curve(results["y_true"], results["y_prob"],
                       save_path=out / "roc_curve.png")


if __name__ == "__main__":
    main()
