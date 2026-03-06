"""
scripts/predict.py

CLI implementation for Transparent Mini-Med image diagnosis.
Provides a simple terminal interface to run classification and generate Grad-CAM heatmaps.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

# Add root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.predictor import MiniMedPredictor
from src.explainability.overlay import create_side_by_side


def process_image(predictor, image_path, output_dir):
    """Run prediction on a single image and save the result."""
    try:
        result = predictor.predict(image_path)
        
        # Print summary
        print(f"\n[Result] {image_path.name}")
        print(f"  Diagnosis:  {result.label}")
        print(f"  Confidence: {result.confidence} ({result.probability_pct})")
        
        # Save visualization
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create side-by-side visualization
            side_by_side = create_side_by_side(
                result.original_bgr, 
                result.overlay_bgr,
                titles=[f"Original", f"Grad-CAM ({result.label})"]
            )
            
            # Add text overlay to the final image
            color = (0, 0, 255) if result.is_pneumonia else (0, 255, 0)
            text = f"{result.label} ({result.probability_pct})"
            cv2.putText(side_by_side, text, (15, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            out_path = output_dir / f"result_{image_path.stem}.jpg"
            cv2.imwrite(str(out_path), side_by_side)
            print(f"  Heatmap:    {out_path}")
            
        return result
    except Exception as e:
        print(f"  [ERROR] Failed to process {image_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Transparent Mini-Med CLI Predictor")
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single X-ray image")
    group.add_argument("--dir", type=str, help="Directory containing X-ray images")
    
    # Config options
    parser.add_argument("--weights", type=str, default="models/mini_med_net_demo.pth",
                        help="Path to model weights (.pth)")
    parser.add_argument("--output", type=str, default="outputs/predictions",
                        help="Directory to save visual results")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (cuda, cpu, or auto)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold (default: 0.5)")
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="Grad-CAM overlay transparency (0-1)")

    args = parser.parse_args()

    # Verify weights
    weights_path = Path(args.weights)
    if not weights_path.exists():
        # Fallback to best model if demo doesn't exist
        fallback = Path("models/mini_med_net_best.pth")
        if fallback.exists():
            weights_path = fallback
        else:
            print(f"ERROR: Weights not found at {args.weights} or {fallback}")
            print("Please run scripts/generate_demo_model.py or train the model first.")
            sys.exit(1)

    # Initialize Predictor
    print(f"[Init] Loading model from {weights_path}...")
    predictor = MiniMedPredictor(
        weights_path=weights_path,
        device=args.device,
        threshold=args.threshold,
        heatmap_alpha=args.alpha
    )
    print(f"[Init] Running on {predictor.device}")

    output_dir = Path(args.output)
    
    # Process single image
    if args.image:
        process_image(predictor, Path(args.image), output_dir)
        
    # Process directory
    elif args.dir:
        input_dir = Path(args.dir)
        if not input_dir.is_dir():
            print(f"ERROR: Directory not found: {args.dir}")
            sys.exit(1)
            
        extensions = (".jpg", ".jpeg", ".png")
        image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]
        
        if not image_files:
            print(f"No images found in {args.dir}")
            sys.exit(0)
            
        print(f"[Batch] Processing {len(image_files)} images...")
        for img_path in tqdm(image_files, desc="Diagnosing"):
            process_image(predictor, img_path, output_dir)
            
    print(f"\n[Done] All results saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
