"""
scripts/download_dataset.py

Helper to download the Kaggle Chest X-Ray (Pneumonia) dataset.

Prerequisites:
    1. Install Kaggle CLI:  pip install kaggle
    2. Get your API token: https://www.kaggle.com/settings  → "Create New Token"
    3. Place kaggle.json in ~/.kaggle/kaggle.json  (chmod 600 on Linux/Mac)

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --output-dir data/chest_xrays
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "chest_xrays"


def download_via_kaggle_api(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Download] Downloading '{KAGGLE_DATASET}'...")
    # Use sys.executable -m kaggle instead of just 'kaggle' to avoid PATH issues
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "-p", str(output_dir), "--unzip"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("[Download] kaggle CLI failed. Trying Python API...")
        _download_via_python_api(output_dir)
    else:
        print(f"[Download] Dataset extracted to: {output_dir}")


def _download_via_python_api(output_dir: Path) -> None:
    try:
        import kaggle  # type: ignore
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET, path=str(output_dir), unzip=True
        )
        print(f"[Download] Dataset extracted to: {output_dir}")
    except ImportError:
        print("ERROR: kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        print(
            "\nManual download steps:\n"
            f"  1. Go to: https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
            "  2. Download the zip file\n"
            f"  3. Extract to: {output_dir}\n"
            "  4. Ensure structure:\n"
            f"       {output_dir}/train/NORMAL/\n"
            f"       {output_dir}/train/PNEUMONIA/\n"
            f"       {output_dir}/val/NORMAL/\n"
            f"       {output_dir}/val/PNEUMONIA/\n"
            f"       {output_dir}/test/NORMAL/\n"
            f"       {output_dir}/test/PNEUMONIA/\n"
        )
        sys.exit(1)


def verify_structure(output_dir: Path) -> bool:
    expected = [
        "train/NORMAL", "train/PNEUMONIA",
        "val/NORMAL",   "val/PNEUMONIA",
        "test/NORMAL",  "test/PNEUMONIA",
    ]
    all_good = True
    for subdir in expected:
        full = output_dir / subdir
        if full.exists():
            count = len(list(full.glob("*.jpeg")) + list(full.glob("*.jpg")))
            print(f"  ✅  {subdir:<30} {count} images")
        else:
            print(f"  ❌  {subdir:<30} NOT FOUND")
            all_good = False
    return all_good


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out = Path(args.output_dir)
    download_via_kaggle_api(out)

    print("\n[Verify] Checking dataset structure...")
    ok = verify_structure(out)
    if ok:
        print("\n✅ Dataset ready for training!")
    else:
        print("\n⚠️  Some directories missing — check manual download steps above.")
