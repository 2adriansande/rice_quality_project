"""Shared settings and helpers for the rice grain quality project (imported by all four scripts)."""
import argparse
import sys
from pathlib import Path

SEED = 42                     # makes every random step repeatable
TEST_SIZE = 0.2               # 80:20 split
N_FOLDS = 5                   # k for stratified k-fold cross-validation
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = {"train", "training", "test", "testing", "val", "valid", "validation"}
# Spanish folder names in the Peruvian rice dataset -> English labels used in charts and the report
LABEL_MAP = {"entero": "whole", "mancha": "stain", "quebrado": "broken", "tiza": "chalk"}

BLUE, ORANGE = "#2a78d6", "#eb6834"

FEATURES_CSV = "features.csv"
TRAIN_CSV = "train_features.csv"
TEST_CSV = "test_features.csv"
MODEL_FILE = "best_model.joblib"


def make_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--out", default="rice_quality_output", help="output folder shared by all four scripts")
    p.add_argument("--seed", type=int, default=SEED)
    return p


def need(path: Path, hint: str):
    """Stop with a clear message if the previous step has not been run."""
    if not Path(path).exists():
        sys.exit(f"Missing {path}. {hint}")


def clean_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
