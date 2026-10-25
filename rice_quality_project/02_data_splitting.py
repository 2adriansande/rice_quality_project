"""STEP 3 of the assignment - DATA SPLITTING.

Holds out 20% of the images as a TEST set (stratified, so each class keeps its share).
The test set is locked away: scripts 03 (training) never read it; only 04 (evaluation) does.
The k-fold cross-validation happens later, inside the 80% training part (script 03).

Usage:  python 02_data_splitting.py [--out rice_quality_output] [--test-size 0.2]
Reads:  features.csv        Writes: train_features.csv, test_features.csv, split_summary.csv, fig_split.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from common import *


def main():
    ap = make_parser(__doc__)
    ap.add_argument("--test-size", type=float, default=TEST_SIZE, help="share held out for testing (0.2 = 80:20, 0.3 = 70:30)")
    a = ap.parse_args()
    out = Path(a.out)
    need(out / FEATURES_CSV, "Run 01_data_preparation.py first.")

    df = pd.read_csv(out / FEATURES_CSV)
    train, test = train_test_split(df, test_size=a.test_size, stratify=df["label"], random_state=a.seed)
    train, test = train.sort_index(), test.sort_index()
    assert set(train["path"]).isdisjoint(test["path"]), "an image is in both train and test"

    train.to_csv(out / TRAIN_CSV, index=False)
    test.to_csv(out / TEST_CSV, index=False)

    summary = pd.DataFrame({"train": train["label"].value_counts(), "test": test["label"].value_counts()}).fillna(0).astype(int)
    summary.loc["TOTAL"] = summary.sum()
    summary.to_csv(out / "split_summary.csv")
    print(f"Split {1 - a.test_size:.0%}:{a.test_size:.0%}  ->  train {len(train)}  test {len(test)}\n")
    print(summary.to_string())

    body = summary.drop(index="TOTAL")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    body.plot.bar(ax=ax, color=[BLUE, ORANGE], width=0.75, rot=0)
    ax.set_ylabel("images"); ax.set_title(f"Train / test split per class ({1 - a.test_size:.0%}:{a.test_size:.0%})")
    clean_axes(ax); fig.tight_layout(); fig.savefig(out / "fig_split.png", dpi=160); plt.close(fig)
    print("\nNext: python 03_model_training.py --out", out)


if __name__ == "__main__":
    main()
