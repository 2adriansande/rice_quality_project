"""STEP 4 of the assignment - MODEL TRAINING (with stratified k-fold cross-validation).

Uses ONLY train_features.csv. The test set is never opened here.
1. Compares several models with stratified k-fold cross-validation on the training data.
2. Picks the best one by cross-validated kappa.
3. Fits that model on the whole training set and saves it for script 04.

Usage:  python 03_model_training.py [--out rice_quality_output] [--folds 5]
Reads:  train_features.csv
Writes: cv_results.csv, fig2_cv_accuracy.png, selected_model.json, best_model.joblib
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import *


def build_models(seed, tune=False, inner_folds=3):
    # Scaling is inside the pipeline, so inside each fold it is learned from that fold's training part only (no leakage).
    models = {
        "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(class_weight="balanced", random_state=seed)),
        "k-NN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }
    if tune:
        # Hyper-parameter tuning INSIDE cross-validation (nested CV): the outer folds below give an honest score,
        # because each outer fold re-runs its own inner search using only that fold's training images.
        kappa = make_scorer(cohen_kappa_score)
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
        rf_grid = dict(max_depth=[None, 20], min_samples_leaf=[1, 3], max_features=["sqrt", 0.3])
        svm_grid = dict(svc__C=[1, 5, 20], svc__gamma=["scale", 0.01, 0.03])
        models["Random Forest (tuned)"] = GridSearchCV(
            RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
            rf_grid, cv=inner, scoring=kappa, n_jobs=1)
        models["SVM (RBF, tuned)"] = GridSearchCV(
            make_pipeline(StandardScaler(), SVC(class_weight="balanced", random_state=seed)),
            svm_grid, cv=inner, scoring=kappa, n_jobs=-1)
    return models


def main():
    ap = make_parser(__doc__)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--tune", action="store_true", help="also tune Random Forest and SVM settings with nested CV (slower)")
    ap.add_argument("--drop-prefix", default=None, help="leave out features whose name starts with this, e.g. chalk_ "
                    "(used to measure what the chalk features add)")
    a = ap.parse_args()
    out = Path(a.out)
    need(out / TRAIN_CSV, "Run 02_data_splitting.py first.")

    train = pd.read_csv(out / TRAIN_CSV)
    X, y = train.drop(columns=["path", "label"]), train["label"]
    if a.drop_prefix:
        dropped = [c for c in X.columns if c.startswith(a.drop_prefix)]
        X = X.drop(columns=dropped)
        print(f"Left out {len(dropped)} features starting with '{a.drop_prefix}'")
    print(f"Using {X.shape[1]} features")
    k = min(a.folds, int(y.value_counts().min()))
    if k < 2:
        raise SystemExit("A class has fewer than 2 training images; cannot cross-validate.")
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=a.seed)
    scoring = dict(accuracy="accuracy",
                   precision=make_scorer(precision_score, average="macro", zero_division=0),
                   recall=make_scorer(recall_score, average="macro", zero_division=0),
                   kappa=make_scorer(cohen_kappa_score))

    models = build_models(a.seed, tune=a.tune)
    rows = []
    print(f"{k}-fold stratified cross-validation on {len(train)} training images\n")
    for name, m in models.items():
        r = cross_validate(m, X, y, cv=cv, scoring=scoring, n_jobs=1)
        row = dict(model=name, folds=k)
        for met in scoring:
            row[f"cv_{met}_mean"], row[f"cv_{met}_std"] = r[f"test_{met}"].mean(), r[f"test_{met}"].std()
        rows.append(row)
        print(f"{name:26s} accuracy {row['cv_accuracy_mean']:.3f} +/- {row['cv_accuracy_std']:.3f}   "
              f"precision {row['cv_precision_mean']:.3f}   recall {row['cv_recall_mean']:.3f}   kappa {row['cv_kappa_mean']:.3f}")
    CV = pd.DataFrame(rows)
    CV.to_csv(out / "cv_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(CV["model"][::-1], CV["cv_accuracy_mean"][::-1], xerr=CV["cv_accuracy_std"][::-1], color=BLUE, capsize=3)
    ax.set_xlim(0, 1); ax.set_xlabel(f"{k}-fold CV accuracy on the training set (mean +/- std)")
    clean_axes(ax); fig.tight_layout(); fig.savefig(out / "fig2_cv_accuracy.png", dpi=160); plt.close(fig)

    best_name = CV.loc[CV["cv_kappa_mean"].idxmax(), "model"]
    best = models[best_name].fit(X, y)
    best_params = None
    if isinstance(best, GridSearchCV):             # tuned model: keep the refitted best estimator and its settings
        best_params = {k_: (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k_, v in best.best_params_.items()}
        print(f"Tuned settings chosen by inner CV: {best_params}")
        best = best.best_estimator_
    joblib.dump(dict(model=best, name=best_name, features=list(X.columns), classes=sorted(y.unique())), out / MODEL_FILE)
    (out / "selected_model.json").write_text(json.dumps(dict(best_model=best_name, chosen_by="highest mean CV kappa",
                                                             folds=k, n_train=len(train), n_features=X.shape[1],
                                                             tuned=a.tune, best_params=best_params,
                                                             dropped_prefix=a.drop_prefix), indent=2))
    print(f"\nBest by CV kappa: {best_name}  (fitted on all {len(train)} training images, saved to {out / MODEL_FILE})")
    print("Next: python 04_model_evaluation.py --out", out)


if __name__ == "__main__":
    main()
