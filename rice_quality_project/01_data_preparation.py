"""STEP 2 of the assignment - DATA PREPARATION.

Wrangling : find every image, use the folder name as the class label
Cleaning  : drop unreadable files and exact duplicates, log every removal
Features  : turn each image into numbers (colour, texture, shape)

Usage:  python 01_data_preparation.py DATASET_DIR [--out rice_quality_output] [--max-per-class 1500] [--img-size 128]
Writes: cleaning_log.csv, features.csv, fig1_class_distribution.png, prep_settings.json
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color, feature, filters, measure

from common import *
from grain_utils import chalk_features


# ------------------------------------------------------------------ wrangling
def build_inventory(root: Path) -> pd.DataFrame:
    """One row per image: path + label (the innermost class folder; train/val/test folder names are ignored)."""
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            parts = [x for x in p.relative_to(root).parts[:-1] if x.lower() not in SPLIT_NAMES]
            if parts:
                rows.append(dict(path=str(p), label=parts[-1].strip().lower()))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ cleaning
def clean_inventory(inv: pd.DataFrame, log: list) -> pd.DataFrame:
    log.append(dict(step="images found", rows=len(inv), removed=0))
    ok = []
    for p in inv["path"]:
        try:
            with Image.open(p) as im:
                im.verify()
            ok.append(True)
        except Exception:
            ok.append(False)
    n = len(inv)
    inv = inv[ok].copy()
    log.append(dict(step="removed unreadable/corrupt files", rows=len(inv), removed=n - len(inv)))

    inv["md5"] = [hashlib.md5(Path(p).read_bytes()).hexdigest() for p in inv["path"]]
    n = len(inv)
    inv = inv.drop_duplicates("md5")
    log.append(dict(step="removed exact duplicate files", rows=len(inv), removed=n - len(inv)))
    conflicts = int((inv.groupby("md5")["label"].nunique() > 1).sum())
    log.append(dict(step="same image under two classes (label conflicts)", rows=len(inv), removed=conflicts))
    return inv.drop(columns="md5").reset_index(drop=True)


# ------------------------------------------------------------------ feature extraction
def extract_features(path: str, size: int) -> dict:
    img = Image.open(path).convert("RGB").resize((size, size))
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    hsv = color.rgb2hsv(rgb)
    gray = color.rgb2gray(rgb)
    f = {}

    # colour: mean and spread per channel (RGB and HSV)
    for i, c in enumerate("rgb"):
        f[f"mean_{c}"], f[f"std_{c}"] = rgb[..., i].mean(), rgb[..., i].std()
    for i, c in enumerate("hsv"):
        f[f"mean_{c}"], f[f"std_{c}"] = hsv[..., i].mean(), hsv[..., i].std()
    # colour histograms (8 bins for each of hue, saturation, value)
    for i, c in enumerate("hsv"):
        h, _ = np.histogram(hsv[..., i], bins=8, range=(0, 1))
        for k, v in enumerate(h / h.sum()):
            f[f"hist_{c}{k}"] = v

    # defect-oriented colour fractions: chalk = bright and unsaturated; stain = dark or yellow-brown
    f["frac_bright_unsat"] = float(((hsv[..., 2] > 0.8) & (hsv[..., 1] < 0.25)).mean())
    f["frac_dark"] = float((hsv[..., 2] < 0.35).mean())
    f["frac_yellow_brown"] = float(((hsv[..., 0] > 0.05) & (hsv[..., 0] < 0.17) & (hsv[..., 1] > 0.35)).mean())

    # texture: local binary patterns (uniform, P=8, R=1 -> 10 bins) and edge strength
    lbp = feature.local_binary_pattern((gray * 255).astype(np.uint8), 8, 1, "uniform")
    h, _ = np.histogram(lbp, bins=10, range=(0, 10))
    for k, v in enumerate(h / h.sum()):
        f[f"lbp{k}"] = v
    f["edge_mean"] = float(filters.sobel(gray).mean())
    f["gray_std"] = float(gray.std())

    # shape of the largest object after Otsu thresholding (grain vs background)
    try:
        mask = gray > filters.threshold_otsu(gray)
        if mask.mean() > 0.5:                      # assume the grain is the smaller region
            mask = ~mask
        lab = measure.label(mask)
        if lab.max() == 0:
            raise ValueError
        r = max(measure.regionprops(lab), key=lambda x: x.area)
        minr, minc, maxr, maxc = r.bbox
        f["shape_area_ratio"] = r.area / mask.size
        f["shape_solidity"] = r.solidity
        f["shape_eccentricity"] = r.eccentricity
        f["shape_aspect"] = (maxc - minc) / max(1, (maxr - minr))
    except Exception:
        f.update(shape_area_ratio=0.0, shape_solidity=0.0, shape_eccentricity=0.0, shape_aspect=1.0)

    # chalk-oriented features measured inside the grain only (all start with "chalk_"; see grain_utils.py)
    f.update(chalk_features(hsv, gray))
    return f


def main():
    ap = make_parser(__doc__)
    ap.add_argument("dataset_dir")
    ap.add_argument("--max-per-class", type=int, default=1500, help="cap images per class to keep run time short")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--keep-spanish", action="store_true", help="do not translate Spanish class names")
    a = ap.parse_args()

    root, out = Path(a.dataset_dir), Path(a.out)
    if not root.is_dir():
        sys.exit(f"Dataset folder not found: {root}")
    out.mkdir(parents=True, exist_ok=True)

    inv = build_inventory(root)
    if inv.empty:
        sys.exit("No images found. Expected one sub-folder per class containing image files.")
    if not a.keep_spanish:
        inv["label"] = inv["label"].replace(LABEL_MAP)
    log = []
    inv = clean_inventory(inv, log)
    inv = (inv.sample(frac=1, random_state=a.seed).groupby("label").head(a.max_per_class)
              .sort_values("path").reset_index(drop=True))
    log.append(dict(step=f"cap at {a.max_per_class} images per class", rows=len(inv), removed=0))
    pd.DataFrame(log).to_csv(out / "cleaning_log.csv", index=False)

    counts = inv["label"].value_counts()
    print("Class counts after cleaning:\n" + counts.to_string())
    if counts.size < 2:
        sys.exit("Need at least two classes.")
    if counts.min() < 10:
        print("WARNING: a class has fewer than 10 images; later steps may fail or be unreliable.")

    print(f"Extracting features from {len(inv)} images ...")
    rows = []
    for i, (p, lab) in enumerate(zip(inv["path"], inv["label"]), 1):
        rows.append(dict(path=p, label=lab, **extract_features(p, a.img_size)))
        if i % 500 == 0:
            print(f"  {i}/{len(inv)}")
    feats = pd.DataFrame(rows)
    feats.to_csv(out / FEATURES_CSV, index=False)
    (out / "prep_settings.json").write_text(json.dumps(dict(img_size=a.img_size, max_per_class=a.max_per_class,
                                                            n_images=len(feats), n_features=feats.shape[1] - 2)))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(counts.index, counts.values, color=BLUE)
    ax.set_ylabel("images"); ax.set_title("Class distribution (after cleaning)")
    clean_axes(ax); fig.tight_layout(); fig.savefig(out / "fig1_class_distribution.png", dpi=160); plt.close(fig)
    print(f"\nSaved {out / FEATURES_CSV}  ({len(feats)} images x {feats.shape[1] - 2} features)")
    print("Next: python 02_data_splitting.py --out", out)


if __name__ == "__main__":
    main()
