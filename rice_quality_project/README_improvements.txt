Improvement experiments (run in this order; every run uses its own --out folder so nothing is overwritten)

1. python 01_data_preparation.py "DATASET_PATH" --out out_chalk      (now 63 features: the 55 old + 8 chalk_ features)
2. python check_grain_mask.py --out out_chalk                         (OPEN fig_grain_mask_check.png and check the red outlines)
3. python 02_data_splitting.py --out out_chalk
4. python 03_model_training.py --out out_chalk
   python 04_model_evaluation.py --out out_chalk                      -> "+ chalk features"
5. Ablation: copy the folder, then train WITHOUT the chalk features (this should reproduce your earlier 55-feature result)
   xcopy out_chalk out_base /E /I
   python 03_model_training.py --out out_base --drop-prefix chalk_
   python 04_model_evaluation.py --out out_base                       -> "baseline"
6. Tuning: xcopy out_chalk out_tuned /E /I
   python 03_model_training.py --out out_tuned --tune
   python 04_model_evaluation.py --out out_tuned                      -> "+ chalk + tuned"
7. Transfer learning (needs: pip install -r requirements-transfer.txt)
   python 05_transfer_learning.py --out out_chalk                     -> writes tl_* files
8. python compare_results.py "Baseline=out_base" "+ chalk=out_chalk" "+ chalk + tuned=out_tuned" "Transfer learning=out_chalk:tl_metrics.json"
