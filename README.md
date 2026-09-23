# ARI 510 Lab 1 - Heart Disease Classification

University of Michigan-Flint, Fall 2026. Graduate (510) track.

Comparison of six classifier families on the UCI heart disease dataset (920 patients, binary disease presence/absence), with two baselines, repeated cross-validation, a training-size ablation, and error analysis.

**Report:** `report/report.md`

## Results summary

Six classifier families were compared across 88 configurations on a held-out dev set, then confirmed with repeated stratified cross-validation on the training split. No pair of models is statistically separable; all six span 0.0121 F1. The SVM (RBF, C=1, class_weight=balanced) ranked first under both procedures and was evaluated once on the test set: **F1 0.796, accuracy 0.777**, against a majority-class baseline of F1 0.713.

The dominant finding is in the data rather than the models: 172 cholesterol values are sentinel zeros encoding "not measured," and their pattern of absence identifies the source hospital, which correlates strongly with the label. See `report/report.md` Section 5.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ with numpy, pandas, matplotlib, and scikit-learn.

## Running

Run from the repository root, in this order:

```bash
python -m src.experiments   # dev-set comparison, 88 configs -> results/dev_results.csv
python -m src.confirm       # repeated CV on train        -> results/cv_results.csv
python -m src.final         # test eval, ablations, errors -> results/test_results.csv, ablation.csv,
                            #                                 ablation.png, site_ablation.csv, errors.csv
```

`src/final.py` is the only file that touches the test set.

## Layout
data/ heart_disease_uci.csv
src/ data.py, evaluate.py, experiments.py, confirm.py, final.py
results/ generated CSVs and the ablation figure
report/ report.md and the exported PDF
ai_logs/ generative AI use documentation


## Module notes

- `data.py` - loading, sentinel-zero correction, and the leak-free `ColumnTransformer` pipeline. All preprocessing is fit on the training split only.
- `evaluate.py` - metric helpers and the two required baselines. The random baseline is computed analytically over 10,000 simulated prediction sets and was validated against scikit-learn on 200 draws.
- `experiments.py` - dev-set hyperparameter search across six model families.
- `confirm.py` - repeated stratified cross-validation (5 folds x 6 repeats) with paired per-fold differences.
- `final.py` - single test-set evaluation, training-size ablation, site ablation, and error analysis.