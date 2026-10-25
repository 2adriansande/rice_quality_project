"""Compare several runs (different output folders) in one table and one chart.

Each run is given as  LABEL=FOLDER  or  LABEL=FOLDER:metrics_file  (default metrics file is metrics.json;
use tl_metrics.json for the transfer-learning run). It also checks that all runs were tested on the same images.

Example:
  python compare_results.py "Baseline (55 features)=rice_quality_output" ^
      "+ chalk features=out_chalk" "+ chalk + tuned=out_tuned" "Transfer learning=out_chalk:tl_metrics.json"

Writes: model_comparison.csv and fig_model_comparison.png into --out (default: model_comparison)
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BLUE, ORANGE, clean_axes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="LABEL=FOLDER or LABEL=FOLDER:metrics_file")
    ap.add_argument("--out", default="model_comparison")
    a = ap.parse_args()

    rows, test_sets = [], {}
    for spec in a.runs:
        if "=" not in spec:
            sys.exit(f"Bad run '{spec}'. Use LABEL=FOLDER")
        label, rest = spec.split("=", 1)
        folder, _, mfile = rest.partition(":")
        folder, mfile = Path(folder), (mfile or "metrics.json")
        mpath = folder / mfile
        if not mpath.exists():
            sys.exit(f"Not found: {mpath}")
        m = json.loads(mpath.read_text())
        rows.append(dict(run=label, model=m["model"], n_test=m["n_test"], accuracy=m["accuracy"],
                         precision_macro=m["precision_macro"], recall_macro=m["recall_macro"], kappa=m["kappa"]))
        tf = folder / "test_features.csv"
        if tf.exists():
            test_sets[label] = set(pd.read_csv(tf, usecols=["path"])["path"])

    if len({frozenset(s) for s in test_sets.values()}) > 1:
        print("WARNING: these runs were NOT tested on identical image sets, so the comparison is not like-for-like.")
    elif test_sets:
        print(f"Check passed: all runs share the same {len(next(iter(test_sets.values())))} test images.")

    T = pd.DataFrame(rows)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / "model_comparison.csv", index=False)
    print("\n" + T.round(4).to_string(index=False))

    x = np.arange(len(T)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(T)), 4.2))
    b1 = ax.bar(x - w / 2, T["accuracy"], w, color=BLUE, label="Accuracy")
    b2 = ax.bar(x + w / 2, T["kappa"], w, color=ORANGE, label="Kappa")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01, f"{r.get_height():.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([t.replace(" (", "\n(") for t in T["run"]], fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_title("Test-set results by run (same test images)")
    ax.legend(frameon=False); clean_axes(ax)
    fig.tight_layout(); fig.savefig(out / "fig_model_comparison.png", dpi=160); plt.close(fig)
    print("\nSaved", out / "model_comparison.csv", "and", out / "fig_model_comparison.png")


if __name__ == "__main__":
    main()
