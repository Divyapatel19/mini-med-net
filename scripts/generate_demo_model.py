"""
scripts/generate_demo_model.py

Generate random model weights for MiniMedNet without any training data.
Allows the Streamlit UI to run immediately for demonstration purposes.

Usage:
    python scripts/generate_demo_model.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # type: ignore
from architectures.mini_med_net import MiniMedNet  # type: ignore


def generate_demo_model(
    output_path: str | Path | None = None,
    seed: int = 42,
) -> None:
    """Create and save random-weight MiniMedNet for UI demonstration."""
    if output_path is None:
        output_path = ROOT / "models" / "mini_med_net_demo.pth"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    model = MiniMedNet(dropout_rate=0.5)

    checkpoint = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "val_f1": 0.0,
        "val_loss": 0.0,
        "note": (
            "DEMO model — random weights only. "
            "Predictions are meaningless. "
            "Train on real data for clinical use."
        ),
    }

    torch.save(checkpoint, output_path)

    print("=" * 56)
    print("  Demo model weights generated!")
    print(f"  Path:   {output_path}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print("")
    print("  [Warning] These are RANDOM weights — for UI demo only.")
    print("  Train on real data: python core/training/train.py")
    print("=" * 56)


if __name__ == "__main__":
    generate_demo_model()
