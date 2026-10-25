"""OPTIONAL STEP - TRANSFER LEARNING comparison on the SAME train/test split.

A neural network (MobileNetV2) pretrained on ImageNet is used as a fixed feature extractor: every image becomes a
1280-number "embedding" learned by the network instead of the hand-designed features of script 01. Simple classifiers
(logistic regression, SVM) are then trained on those embeddings with the same stratified k-fold cross-validation, the
best one is chosen by CV kappa, and it is tested once on the same held-out images as script 04.
(The network itself is NOT fine-tuned; that would be the next step if you want a full CNN comparison.)

Needs:  pip install torch torchvision   (first run downloads the pretrained weights, about 14 MB)
Usage:  python 05_transfer_learning.py [--out rice_quality_output] [--img-size 224] [--weights imagenet]
Reads:  train_features.csv, test_features.csv (only the image paths and labels; the same split as scripts 03/04)
Writes: tl_cv_results.csv, tl_metrics.json, tl_classification_report.txt, tl_confusion_*.png,
        tl_test_predictions.csv, tl_embeddings_*.npy (cache)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, make_scorer, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import *


def embed_images(paths, img_size, weights, batch=64):
    """Return an (n_images x 1280) array of MobileNetV2 embeddings. Torch is imported here so the rest of the
    project works without it."""
    try:
        import torch
        import torchvision
        from torchvision import transforms
    except ImportError:
        raise SystemExit("PyTorch is not installed. Run:  pip install torch torchvision")
    torch.manual_seed(SEED)
    w = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1 if weights == "imagenet" else None
    net = torchvision.models.mobilenet_v2(weights=w).eval()
    tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    outs = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch]])
            f = net.features(x).mean(dim=(2, 3))          # global average pooling -> 1280 numbers per image
            outs.append(f.numpy())
            print(f"  embedded {min(i + batch, len(paths))}/{len(paths)}", end="\r")
    print()
    return np.vstack(outs)


def get_embeddings(df, tag, out, img_size, weights):
    cache = out / f"tl_embeddings_{tag}.npy"
    if cache.exists():
        E = np.load(cache)
        if E.shape[0] == len(df):
            print(f"Using cached embeddings: {cache.name}")
            return E
    print(f"Computing {tag} embeddings ({len(df)} images, weights={weights}) ...")
    E = embed_images(df["path"].tolist(), img_size, weights)
    np.save(cache, E)
    return E


def main():
    ap = make_parser(__doc__)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--weights", choices=["imagenet", "none"], default="imagenet",
                    help="'none' = random weights (only for testing the code; results are meaningless)")
    a = ap.parse_args()
    out = Path(a.out)
    need(out / TRAIN_CSV, "Run 02_data_splitting.py first.")
    train = pd.read_csv(out / TRAIN_CSV, usecols=["path", "label"])
    test = pd.read_csv(out / TEST_CSV, usecols=["path", "label"])
    Xtr = get_embeddings(train, "train", out, a.img_size, a.weights)
    Xte = get_embeddings(test, "test", out, a.img_size, a.weights)
    ytr, yte = train["label"], test["label"]
    classes = sorted(ytr.unique())

    models = {
        "Logistic Regression (on embeddings)": make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=a.seed)),
        "SVM RBF (on embeddings)": make_pipeline(StandardScaler(), SVC(class_weight="balanced", random_state=a.seed)),
    }
    k = min(N_FOLDS, int(ytr.value_counts().min()))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=a.seed)
    scoring = dict(accuracy="accuracy",
                   precision=make_scorer(precision_score, average="macro", zero_division=0),
                   recall=make_scorer(recall_score, average="macro", zero_division=0),
                   kappa=make_scorer(cohen_kappa_score))
    rows = []
    print(f"\n{k}-fold stratified CV on {len(train)} training images (same folds logic as script 03)\n")
    for name, m in models.items():
        r = cross_validate(m, Xtr, ytr, cv=cv, scoring=scoring, n_jobs=1)
        row = dict(model=name, folds=k)
        for met in scoring:
            row[f"cv_{met}_mean"], row[f"cv_{met}_std"] = r[f"test_{met}"].mean(), r[f"test_{met}"].std()
        rows.append(row)
        print(f"{name:38s} accuracy {row['cv_accuracy_mean']:.3f} +/- {row['cv_accuracy_std']:.3f}   kappa {row['cv_kappa_mean']:.3f}")
    CV = pd.DataFrame(rows)
    CV.to_csv(out / "tl_cv_results.csv", index=False)

    best_name = CV.loc[CV["cv_kappa_mean"].idxmax(), "model"]
    best = models[best_name].fit(Xtr, ytr)
    pred = best.predict(Xte)
    metrics = dict(model=best_name + f" [MobileNetV2 {a.weights}]", n_test=len(test), classes=classes,
                   accuracy=accuracy_score(yte, pred),
                   precision_macro=precision_score(yte, pred, average="macro", zero_division=0),
                   recall_macro=recall_score(yte, pred, average="macro", zero_division=0),
                   kappa=cohen_kappa_score(yte, pred))
    (out / "tl_metrics.json").write_text(json.dumps(metrics, indent=2))
    rep = classification_report(yte, pred, labels=classes, zero_division=0)
    (out / "tl_classification_report.txt").write_text(rep)
    print(f"\nTEST RESULTS - {metrics['model']}")
    for k_ in ("accuracy", "precision_macro", "recall_macro", "kappa"):
        print(f"  {k_:16s} {metrics[k_]:.4f}")
    print("\n" + rep)

    for norm, tag in ((None, "counts"), ("true", "normalized")):
        cm = confusion_matrix(yte, pred, labels=classes, normalize=norm)
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False,
                                                                 values_format=".2f" if norm else "d")
        ax.set_title(f"Test confusion matrix ({tag})\n{best_name}", fontsize=9)
        fig.tight_layout(); fig.savefig(out / f"tl_confusion_{tag}.png", dpi=160); plt.close(fig)
    pd.DataFrame(dict(path=test["path"], true=yte, predicted=pred)).to_csv(out / "tl_test_predictions.csv", index=False)
    print("Next: python compare_results.py  (compares all your runs in one table and chart)")


if __name__ == "__main__":
    main()
