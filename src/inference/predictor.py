"""
core/inference/predictor.py

End-to-end Predictor for Transparent Mini-Med.

Loads the trained MiniMedNet, runs inference, computes Grad-CAM, and
returns a rich diagnostic result dict ready for the Streamlit UI.

Usage (CLI):
    python src/inference/predictor.py --image path/to/xray.jpg \
        --weights models/mini_med_net_demo.pth
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.architectures.mini_med_net import build_model
from src.explainability.gradcam import generate_gradcam
from src.explainability.overlay import overlay_heatmap_on_image, bgr_to_pil
from src.utils.image_utils import load_and_preprocess


# ────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class DiagnosticResult:
    """Prediction output for one chest X-ray image."""

    # Prediction
    label:       str    # "Pneumonia" or "Normal"
    probability: float  # Probability of Pneumonia in [0, 1]
    confidence:  str    # "High" / "Moderate" / "Low"

    # Heatmap
    heatmap:     np.ndarray   # float32 (H, W) in [0, 1]
    overlay_bgr: np.ndarray   # uint8 BGR (H, W, 3) — heatmap overlaid on image
    original_bgr: np.ndarray  # uint8 BGR (H, W, 3) — original image

    # Meta
    image_path:  str
    device:      str

    # Convenience
    @property
    def is_pneumonia(self) -> bool:
        return self.label == "Pneumonia"

    @property
    def probability_pct(self) -> str:
        return f"{self.probability * 100:.1f}%"

    @property
    def normal_probability(self) -> float:
        return 1.0 - self.probability

    def __repr__(self) -> str:
        return (f"DiagnosticResult(label={self.label!r}, "
                f"prob={self.probability:.4f}, confidence={self.confidence!r})")


def _confidence_label(prob: float) -> str:
    """Convert probability to a human-readable confidence tier."""
    diff = abs(prob - 0.5)
    if diff >= 0.35:
        return "High"
    elif diff >= 0.20:
        return "Moderate"
    else:
        return "Low"


# ────────────────────────────────────────────────────────────────────────────
# Predictor class
# ────────────────────────────────────────────────────────────────────────────

class MiniMedPredictor:
    """
    Stateful predictor: loads model once, supports repeated predict() calls.

    Args:
        weights_path: Path to .pth model weights.
        device:       'auto', 'cpu', or 'cuda'.
        threshold:    Probability cut-off for Pneumonia label (default 0.5).
        heatmap_alpha: Opacity of Grad-CAM overlay (0–1).

    Example:
        predictor = MiniMedPredictor("models/mini_med_net_demo.pth")
        result    = predictor.predict("xray.jpg")
        print(result)
    """

    def __init__(
        self,
        weights_path: str | Path,
        device: str = "auto",
        threshold: float = 0.5,
        heatmap_alpha: float = 0.4,
    ) -> None:
        self.weights_path  = str(weights_path)
        self.threshold     = threshold
        self.heatmap_alpha = heatmap_alpha

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = build_model(
            weights_path=self.weights_path,
            device=self.device,
        )
        self.model.eval()

    def predict(self, image_path: str | Path) -> DiagnosticResult:
        """
        Run full diagnostic pipeline on one image.

        Args:
            image_path: Path to a JPEG/PNG chest X-ray.

        Returns:
            DiagnosticResult with label, probability, heatmap, and overlay.
        """
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # ── Preprocess ────────────────────────────────────────────────────
        tensor, orig_bgr, _ = load_and_preprocess(img_path)
        tensor = tensor.to(self.device)

        # ── Forward pass ──────────────────────────────────────────────────
        with torch.no_grad():
            logit = self.model(tensor)
            prob  = torch.sigmoid(logit).item()

        label      = "Pneumonia" if prob >= self.threshold else "Normal"
        confidence = _confidence_label(prob)

        # ── Grad-CAM ──────────────────────────────────────────────────────
        heatmap = generate_gradcam(self.model, tensor)

        # Resize heatmap to match original image
        h, w = orig_bgr.shape[:2]
        heatmap_full = cv2.resize(heatmap, (w, h))

        overlay = overlay_heatmap_on_image(
            orig_bgr, heatmap_full, alpha=self.heatmap_alpha
        )

        return DiagnosticResult(
            label=label,
            probability=float(prob),
            confidence=confidence,
            heatmap=heatmap_full,
            overlay_bgr=overlay,
            original_bgr=orig_bgr,
            image_path=str(img_path),
            device=self.device,
        )

    def predict_from_pil(self, pil_image) -> DiagnosticResult:
        """
        Run prediction directly on a PIL Image (useful for Streamlit uploads).

        Saves the image to a temp file internally.
        """
        import tempfile, os
        from PIL import Image as PILImage

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            pil_image.convert("RGB").save(tmp_path, "JPEG")
            result = self.predict(tmp_path)
        finally:
            os.unlink(tmp_path)
            
        return result


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def cli() -> None:
    parser = argparse.ArgumentParser(description="Run MiniMedNet inference on one image")
    parser.add_argument("--image",   required=True, help="Path to chest X-ray image")
    parser.add_argument("--weights", default="models/mini_med_net_demo.pth",
                        help="Path to model weights (.pth)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-overlay", default=None,
                        help="Optional path to save heatmap overlay image")
    args = parser.parse_args()

    predictor = MiniMedPredictor(args.weights, threshold=args.threshold)
    result    = predictor.predict(args.image)

    print("\n── Diagnostic Result ─────────────────────────────────")
    print(f"  Label:       {result.label}")
    print(f"  Probability: {result.probability_pct} (Pneumonia)")
    print(f"  Confidence:  {result.confidence}")
    print(f"  Device:      {result.device}")
    print("───────────────────────────────────────────────────────")

    if args.save_overlay:
        cv2.imwrite(args.save_overlay, result.overlay_bgr)
        print(f"  Overlay saved → {args.save_overlay}")


if __name__ == "__main__":
    cli()
