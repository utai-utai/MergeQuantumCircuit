"""
Geometric QML — classical->quantum subspace transfer map, merging, and the five
paper experiments.  Core code lives in this package; `reproduce.ipynb` at the
repository root is the single entry point for viewing or re-running everything.

Shared paths and constants are exported here so every module agrees on where
data and artefacts live.
"""
import os

# Repository layout (src/ sits one level below the project root).
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT_DIR, "result")
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(RESULT_DIR, exist_ok=True)

# Global defaults shared across experiments.
SEED = 0
SHOTS = 4096
BACKEND_NAME = "ibm_kobe"

__all__ = ["ROOT_DIR", "RESULT_DIR", "DATA_DIR", "SEED", "SHOTS", "BACKEND_NAME"]
