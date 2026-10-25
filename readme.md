# Rice Grain Quality Classification

Classifies images of rice grains into four quality classes (**whole, chalk, stain, broken**) using classic machine
learning on hand-made image features (Random Forest, SVM, k-NN), with an optional transfer-learning comparison.

The project follows the assignment steps: data preparation, data splitting (80:20 plus k-fold cross-validation),
model training, and evaluation (accuracy, precision, recall, Cohen's kappa).

## Dataset

Peruvian rice dataset ("Dataset de arroz peruano") on Kaggle, uploaded by Cristhian Sempertegui:
<https://www.kaggle.com/datasets/cristhiansempertegui/dataset-de-arroz-peruano>

- One folder per class, with Spanish names: `entero` (whole), `mancha` (stain), `quebrado` (broken), `tiza` (chalk).
  The scripts translate these to English automatically (use `--keep-spanish` to turn that off).
- The dataset is **not included** in this project. Download it yourself and note the license on its Kaggle page.
- Capture conditions, image size and license: *https://www.kaggle.com/datasets/cristhiansempertegui/dataset-de-arroz-peruano*

## Project structure

```
rice_quality_project/
  common.py                  shared settings (seed 42, 80:20 split, 5 folds, class-name translation)
  grain_utils.py             grain mask + chalk-oriented features
  01_data_preparation.py     STEP 2  find images, clean, extract features
  02_data_splitting.py       STEP 3  stratified 80:20 train/test split
  03_model_training.py       STEP 4  k-fold cross-validation, model selection, training
  04_model_evaluation.py     STEP 5  one test on the held-out images, metrics and confusion matrices
  05_transfer_learning.py    optional  pretrained MobileNetV2 features on the same split
  check_grain_mask.py        optional  picture to check the grain outlines before trusting the chalk features
  compare_results.py         optional  compare several runs in one table and chart
  run_all.bat                runs steps 01 to 04 on Windows
  requirements.txt           core packages
  requirements-transfer.txt  extra packages for step 05 (torch, torchvision)
```

## Setup

Python 3.10 or newer (the code was tested on Python 3.13). From the project folder:

```
pip install -r requirements.txt
```

Only for the transfer-learning step:

```
pip install -r requirements-transfer.txt
```

## Run the main pipeline

PowerShell, from the project folder (put your own dataset path in the quotes):

```
python 01_data_preparation.py "C:\path\to\dataset"
python 02_data_splitting.py
python 03_model_training.py
python 04_model_evaluation.py
```

Or run all four with `run_all.bat "C:\path\to\dataset"`.

Every script reads the previous step's files from the output folder (default `rice_quality_output`). Use
`--out some_folder` on **all** scripts to change it, and keep `--seed` the same across scripts. If a step is skipped,
the next script says which one to run first.

Useful options:

| Option | Script | Meaning |
|---|---|---|
| `--max-per-class 1500` | 01 | cap images per class to keep run time short |
| `--img-size 128` | 01 | image size used for feature extraction |
| `--test-size 0.3` | 02 | 70:30 split instead of 80:20 |
| `--folds 10` | 03 | number of cross-validation folds |
| `--tune` | 03 | also tune Random Forest and SVM with nested cross-validation (slower) |
| `--drop-prefix chalk_` | 03 | leave out the chalk features (to measure what they add) |

## What the pipeline does

1. **Preparation (01).** Finds every image, labels it by its folder, removes unreadable files and exact duplicates
   (logged in `cleaning_log.csv`), and turns each image into 63 numbers:
   - colour statistics and histograms (36)
   - defect colour fractions for chalk and stain (3)
   - texture: local binary patterns, edge strength (12)
   - shape of the grain: area, solidity, eccentricity, aspect ratio (4)
   - chalk features measured inside the grain only, all named `chalk_...` (8)
2. **Splitting (02).** Holds out 20% of the images as a test set, stratified by class, with a fixed random seed. The
   test set is only opened in step 04 (and in the optional step 05 and comparison).
3. **Training (03).** Compares a majority-class baseline, Random Forest, SVM (RBF) and k-NN (k = 5) with 5-fold
   stratified cross-validation on the training part only. Scaling happens inside each fold, so nothing leaks from the
   validation part. The best model by mean cross-validated kappa is fitted on the whole training set and saved.
4. **Evaluation (04).** Tests the saved model once on the held-out images and reports accuracy, macro precision,
   macro recall, kappa, per-class results and confusion matrices.

## Output files (in the output folder)

| File | Contents |
|---|---|
| `cleaning_log.csv`, `fig1_class_distribution.png` | cleaning steps and class counts |
| `features.csv` | one row per image with all features |
| `split_summary.csv`, `fig_split.png`, `train_features.csv`, `test_features.csv` | the split |
| `cv_results.csv`, `fig2_cv_accuracy.png`, `selected_model.json`, `best_model.joblib` | cross-validation and the chosen model |
| `metrics.json`, `classification_report.txt`, `per_class_metrics.csv` | test-set results |
| `fig3_confusion_counts.png`, `fig3_confusion_normalized.png` | confusion matrices |
| `fig4_feature_importance.png` | top features (Random Forest only) |
| `test_predictions.csv`, `misclassified_test_images.csv` | every test prediction, and the wrong ones |

`best_model.joblib` is a pickle file. Only load model files you created yourself.

## Optional experiments

Use a separate `--out` folder for each run so nothing is overwritten.

```
python 01_data_preparation.py "C:\path\to\dataset" --out out_chalk
python check_grain_mask.py --out out_chalk
python 02_data_splitting.py --out out_chalk
python 03_model_training.py --out out_chalk
python 04_model_evaluation.py --out out_chalk

xcopy out_chalk out_base /E /I
python 03_model_training.py --out out_base --drop-prefix chalk_
python 04_model_evaluation.py --out out_base

xcopy out_chalk out_tuned /E /I
python 03_model_training.py --out out_tuned --tune
python 04_model_evaluation.py --out out_tuned

python 05_transfer_learning.py --out out_chalk

python compare_results.py "Baseline (55 features)=out_base" "+ chalk features=out_chalk" "+ chalk + tuned=out_tuned" "Transfer learning=out_chalk:tl_metrics.json"
```

- **Open `fig_grain_mask_check.png` before trusting the chalk or shape features.** The grain mask assumes the grain is
  the smaller region after Otsu thresholding, and the chalk thresholds are starting guesses.
- `compare_results.py` also checks that every run was tested on the same images.
- Step 05 uses the network only as a fixed feature extractor (no fine-tuning). The first run downloads about 14 MB of
  pretrained weights.

## Results (test set: 905 unseen images)

| Run | Model | Accuracy | Precision (macro) | Recall (macro) | Kappa |
|---|---|---|---|---|---|
| Baseline (55 features) | Random Forest | 0.8751 | 0.8826 | 0.8908 | 0.8298 |
| + chalk features (63) | Random Forest | 0.8762 | 0.8822 | 0.8908 | 0.8313 |
| + chalk + tuned | Random Forest (tuned) | 0.8785 | 0.8858 | 0.8951 | 0.8344 |
| Transfer learning (frozen MobileNetV2) | SVM on embeddings | 0.8221 | 0.8347 | 0.8402 | 0.7576 |

Majority-class baseline: 0.331 accuracy and kappa 0 in cross-validation. The Random Forest's cross-validation accuracy
was 0.856 +/- 0.011.

How to read this:
- The chalk features and tuning changed the result by 1 and 3 images out of 905, which is within the noise (about
  1.1 percentage points). They did not clearly help.
- Transfer learning with a frozen network was clearly worse. Fine-tuning was not tried.
- The main error is whole grains predicted as chalk (50 of 300 in the baseline), and the reverse (22 of 276).

## Limitations

- One dataset, probably photographed under one setup. Performance on other cameras or lighting is unknown.
- Features are hand-designed. A model that learns its own features may separate chalk from whole better.
- One train/test split and one random seed. Near-duplicate photos could sit in both parts; only exact duplicate
  files are removed.
- The chalk thresholds and the grain mask were not validated on the real images by the author of the code.

