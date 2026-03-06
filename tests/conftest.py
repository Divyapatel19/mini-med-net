"""
tests/conftest.py — Shared pytest fixtures.
"""
import sys
from pathlib import Path

# Ensure src/ and the project root are on the Python path for all tests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
