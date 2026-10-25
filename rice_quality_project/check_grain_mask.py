"""Visual check of the grain mask used by the chalk features (run this BEFORE trusting them).

Saves a picture with, for each class, a few original images (top row) and the detected grain outline (bottom row).
If the outline does not follow the grain (for example it picks the background), the shape and chalk features
are unreliable for those images and the thresholds in grain_utils.py need adjusting.

Usage:  python check_grain_mask.py [--out rice_quality_output] [--per-class 4]
Reads:  features.csv (for image paths and labels)     Writes: fig_grain_mask_check.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color, segmentation

from common import *
from grain_utils import grain_mask


def main():
    ap = make_parser(__doc__)
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=128)
    a = ap.parse_args()
    out = Path(a.out)
    need(out / FEATURES_CSV, "Run 01_data_preparation.py first.")
    df = pd.read_csv(out / FEATURES_CSV, usecols=["path", "label"])
    classes = sorted(df["label"].unique())
    n = a.per_class
    fig, axes = plt.subplots(2 * len(classes), n, figsize=(2.2 * n, 4.4 * len(classes)))
    axes = np.atleast_2d(axes)
    for ci, c in enumerate(classes):
        sub = df[df["label"] == c].sample(min(n, (df["label"] == c).sum()), random_state=a.seed)
        for j in range(n):
            ax_img, ax_msk = axes[2 * ci, j], axes[2 * ci + 1, j]
            ax_img.axis("off"); ax_msk.axis("off")
            if j >= len(sub):
                continue
            img = Image.open(sub["path"].iloc[j]).convert("RGB").resize((a.img_size, a.img_size))
            rgb = np.asarray(img, dtype=np.float32) / 255.0
            m = grain_mask(color.rgb2gray(rgb))
            ax_img.imshow(rgb)
            if j == 0:
                ax_img.set_title(c, loc="left", fontsize=10, fontweight="bold")
            if m is None:
                ax_msk.imshow(rgb); ax_msk.set_title("NO GRAIN FOUND", fontsize=8, color="red")
            else:
                ax_msk.imshow(segmentation.mark_boundaries(rgb, m.astype(int), color=(1, 0, 0)))
                ax_msk.set_title(f"mask {m.mean():.0%} of image", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_grain_mask_check.png", dpi=130)
    print("Saved", out / "fig_grain_mask_check.png", "- open it and check the red outlines follow the grains.")


if __name__ == "__main__":
    main()
