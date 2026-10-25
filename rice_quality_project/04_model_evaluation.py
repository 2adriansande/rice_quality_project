"""STEP 5 of the assignment - MODEL EVALUATION on the held-out test set.

Tests the model saved by script 03 ONCE on test_features.csv and reports:
accuracy, precision, recall, Cohen's kappa, per-class report, confusion matrices.

Usage:  python 04_model_evaluation.py [--out rice_quality_output]
Reads:  best_model.joblib, test_features.csv
Writes: metrics.json, classification_report.txt, fig3_confusion_counts.png, fig3_confusion_normalized.png,
        fig4_feature_importance.png (Random Forest only), misclassified_test_images.csv, test_predictions.csv
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, precision_recall_fscore_support, precision_score, recall_score)

from common import *


def main():
    ap = make_parser(__doc__)
    a = ap.parse_args()
    out = Path(a.out)
    need(out / MODEL_FILE, "Run 03_model_training.py first.")
    need(out / TEST_CSV, "Run 02_data_splitting.py first.")

    art = joblib.load(out / MODEL_FILE)          # file written by script 03 of this project
    model, name, classes = art["model"], art["name"], art["classes"]
    test = pd.read_csv(out / TEST_CSV)
    X, y = test[art["features"]], test["label"]
    pred = model.predict(X)

    metrics = dict(model=name, n_test=len(test), classes=classes,
                   accuracy=accuracy_score(y, pred),
                   precision_macro=precision_score(y, pred, average="macro", zero_division=0),
                   recall_macro=recall_score(y, pred, average="macro", zero_division=0),
                   kappa=cohen_kappa_score(y, pred))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    report = classification_report(y, pred, labels=classes, zero_division=0)
    (out / "classification_report.txt").write_text(report)
    print(f"TEST RESULTS - {name} ({len(test)} unseen images)")
    for k_ in ("accuracy", "precision_macro", "recall_macro", "kappa"):
        print(f"  {k_:16s} {metrics[k_]:.4f}")
    print("\n" + report)

    # per-class table for the report
    p, r, f1, s = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    pd.DataFrame(dict(class_=classes, precision=p, recall=r, f1=f1, support=s)).to_csv(out / "per_class_metrics.csv", index=False)

    for norm, tag in ((None, "counts"), ("true", "normalized")):
        cm = confusion_matrix(y, pred, labels=classes, normalize=norm)
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False,
                                                                 values_format=".2f" if norm else "d")
        ax.set_title(f"Test confusion matrix ({tag})\n{name}", fontsize=10)
        fig.tight_layout(); fig.savefig(out / f"fig3_confusion_{tag}.png", dpi=160); plt.close(fig)

    if isinstance(model, RandomForestClassifier):
        imp = pd.Series(model.feature_importances_, index=art["features"]).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.barh(imp.index, imp.values, color=ORANGE)
        ax.set_title("Top 15 feature importances (Random Forest)")
        clean_axes(ax); fig.tight_layout(); fig.savefig(out / "fig4_feature_importance.png", dpi=160); plt.close(fig)

    res = pd.DataFrame(dict(path=test["path"], true=y, predicted=pred))
    res.to_csv(out / "test_predictions.csv", index=False)
    res[res["true"] != res["predicted"]].to_csv(out / "misclassified_test_images.csv", index=False)
    print(f"{int((res['true'] != res['predicted']).sum())} of {len(res)} test images misclassified "
          f"(listed in misclassified_test_images.csv)")


if __name__ == "__main__":
    main()
