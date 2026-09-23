# ARI 510 Lab 1: Heart Disease Classification

**Roshkrishna K Ranjith**
University of Michigan–Flint, Fall 2026
Track: ARI 510 (Graduate)
Code: https://github.com/LUCIFER5671/pulse-check

## 1. Introduction

This report compares machine learning classifiers on a heart disease diagnosis task: predicting from clinical measurements whether a patient has heart disease. I followed the ARI 510 (graduate) track, designing the comparison protocol rather than running a fixed set of configurations.

The data is the UCI heart disease dataset, comprising 920 patient records from four hospitals (Cleveland, Hungary, Switzerland, VA Long Beach) with 15 features mixing continuous clinical measurements with categorical ones. The raw target records severity on a 0 to 4 scale, which I collapsed to a binary presence/absence label; 55.3% of patients have disease. The dataset is not clean, and the cleaning decisions turned out to matter more than the choice of classifier (Section 3.1).

I compared six classifier families: logistic regression, support vector machines, k-nearest neighbours, random forests, histogram-based gradient boosting, and a stochastic gradient descent classifier. Two baselines were included throughout: a majority-class classifier and a random classifier predicting at the training-set class rate. I report accuracy, precision, recall, F1, and F2, treating F1 as decisive for reasons set out in Section 3.4.

## 2. Hypothesis

I hypothesized that Random Forest would achieve the highest F1. The feature space mixes categorical variables (chest pain type, `thal`, `slope`) with continuous ones (age, cholesterol, maximum heart rate), and I expected clinically plausible nonlinear interactions: the significance of a given maximum heart rate should depend on the patient's age. Tree ensembles capture such interactions without manual feature engineering, need no feature scaling, and tolerate the heavy missingness in this data (66% for `ca`, 53% for `thal`). I expected logistic regression to trail, since it fits only a linear boundary in the encoded space. Given n=920, I expected any margin to be modest; ensembles generally need more data to separate from simpler models.

This hypothesis was formed after reviewing the lecture material and an initial inspection of the data, before any models were run.

## 3. Approach

### 3.1 Cleaning and encoding

The dataset contains 920 patients and 15 features. The raw target `num` records severity on a 0 to 4 scale; following the binary framing of the task, I collapsed it to disease present (`num > 0`) versus absent, giving 509 positive and 411 negative cases (55.3% positive).

Initial inspection revealed that missing values are not confined to the columns pandas reports as null. 172 rows record a serum cholesterol of 0 and one records a resting blood pressure of 0, both physiologically impossible, and both sentinel encodings for "not measured." Left untreated, these are read as genuine measurements and distort the feature. I converted both to NaN before any imputation, which raised the missing rate for `chol` from 3.3% to 22.0% (202 of 920 rows).

The columns `fbs` and `exang` load as booleans with missing entries. I mapped these to 1/0 while preserving NaN, so that "not recorded" remains distinct from "false" rather than collapsing into the negative class.

Missingness in this dataset is substantial and highly structured: `ca` is 66.4% missing, `thal` 52.8%, and `slope` 33.6%. Crucially, these rates vary by source hospital; `ca` is 2% missing at Cleveland but 99% missing at Hungary and VA Long Beach. Since the hospitals also differ sharply in disease prevalence (36.2% at Hungary against 93.5% at Switzerland), the pattern of missingness carries information about the label. I therefore retained missingness indicators rather than discarding this signal, and return to its consequences in the error analysis (Section 5).

I dropped the `dataset` column, which records the source hospital, from the default feature set. It is a confounder rather than a clinical variable: a model can improve its score by inferring which hospital a patient came from instead of assessing the patient. An ablation measuring what this costs is reported in Section 4.

### 3.2 Avoiding leakage

All preprocessing, meaning imputation, scaling, and one-hot encoding, is implemented as a `ColumnTransformer` inside a scikit-learn `Pipeline`, so that every transformer is fit on training data only and applied unchanged to dev and test.

This departs from the provided starter notebook, which calls `X.fillna(X.median())` and `pd.get_dummies(X)` on the complete frame before splitting. Doing so computes imputation values from all 920 patients, including those later assigned to dev and test, so information from the held-out sets influences the training data. The effect on a dataset this clean is likely small, but the practice invalidates the held-out sets as estimates of performance on unseen data, which is the entire reason for holding them out.

### 3.3 Data splitting

I used a stratified 60/20/20 split into train (552), dev (184), and test (184), with stratification preserving the 55.3% positive rate in all three sets. Model selection and hyperparameter search were conducted entirely on the dev set. The test set was evaluated once, after the final configurations were fixed; no result from it was used to revise any choice.

### 3.4 Choice of decisive metric

I report accuracy, precision, recall, and F1 throughout, and initially intended F2 as the decisive metric. The reasoning was clinical: in a screening task, a false negative (a patient with disease sent home) carries a higher cost than a false positive (an unnecessary follow-up test), and F2 weights recall above precision accordingly.

Testing this choice against the baselines showed it to be unworkable. The majority-class baseline, which predicts "disease" for every patient, achieves an F2 of 0.8615 on the dev set, while a tuned logistic regression achieves 0.8743. A gap of 0.013 separates a real model from a classifier that examines no features at all. The cause is structural: this dataset has a positive majority, so the constant positive classifier attains perfect recall by construction, and F2's emphasis on recall rewards precisely that degeneracy.

I therefore adopted F1 as the decisive metric. On the same comparison, F1 gives 0.7133 for the majority baseline against 0.8768 for logistic regression, a gap of 0.164 that discriminates. The clinical asymmetry is retained as a secondary criterion: among configurations whose F1 differences fall within noise, I prefer the higher recall. All five metrics are reported for every configuration so the trade-off remains visible.

I note that this decision was made before any test-set evaluation, on the basis of baseline behaviour alone. The fact that the majority baseline still outperforms the final model on F2 at test time is taken up in Section 6; it is a consequence of the metric's structure, not a reason to revisit the choice after the fact.

### 3.5 Model comparison

I compared six classifier families: logistic regression, support vector machines, k-nearest neighbours, random forests, histogram-based gradient boosting, and a stochastic gradient descent classifier. The search covered 88 configurations in total, with grids given in Table 1. Every configuration was trained on the train split and scored on dev.

Two baselines were included, as required: a majority-class classifier, and a random classifier that predicts at the training-set class rate. The latter was computed analytically over 10,000 simulated prediction sets and validated against scikit-learn's per-draw metrics on 200 draws, matching to four decimal places.

## 4. Results

## 5. Error Analysis

## 6. Discussion

## 7. Generative AI Use Statement