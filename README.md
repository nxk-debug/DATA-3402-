# Software Defect Prediction

This repository holds an attempt to apply a Random Forest Classifier to Software Defect Prediction using data from the [Kaggle Playground Series S3E23](https://www.kaggle.com/competitions/playground-series-s3e23) challenge.

---

## Overview

The task, as defined by the Kaggle challenge, is to use 21 numerical software complexity metrics measured per module to predict whether a given software module contains a defect (binary classification). The dataset originates from the NASA Metrics Data Program and contains over 100,000 software modules labeled as defective or non-defective.

The approach in this repository formulates the problem as a binary classification task. We apply a Random Forest Classifier with log1p feature transformation and standard scaling as preprocessing. The model is trained to output a defect probability score for each module rather than a hard binary label, which aligns with the AUC-based evaluation metric used by Kaggle.

Our best model achieved a validation AUC of approximately **0.77**, meaning the model correctly ranks a defective module above a non-defective one 77% of the time. The competition metric is Area Under the ROC Curve (AUC), where a random baseline scores 0.50 and a perfect model scores 1.00.

---

## Summary of Workdone

### Data

- **Type:**
  - Input: CSV file (`train.csv`) containing 21 numerical software complexity features per module
  - Output: Binary label in the `defects` column — `True` (defective) or `False` (non-defective)
- **Size:** ~8 MB total (train + test CSV files)
- **Instances:**
  - Training: 71,234 rows (70%)
  - Validation: 15,264 rows (15%)
  - Hold-out Test: 15,265 rows (15%)
  - Kaggle Test (unlabeled): 43,856 rows

---

### Preprocessing / Clean Up

- **Duplicate removal:** Checked for and removed duplicate rows (none found in this dataset)
- **Column drops:** Removed the `id` column (row identifier with no predictive signal)
- **Target encoding:** Converted `defects` from boolean (`True`/`False`) to integer (`1`/`0`)
- **Categorical check:** Verified no categorical features exist; all 21 features are numerical, so no one-hot encoding was needed
- **Log1p transform:** Applied `log(1 + x)` to all features to compress heavy right-skewed distributions and reduce the influence of extreme outliers (e.g., modules with thousands of lines of code)
- **StandardScaler:** Normalized all features to zero mean and unit variance so that no single feature dominates due to scale differences

---

### Data Visualization

**Class Distribution:**

The dataset is imbalanced — approximately 77.3% of modules are non-defective and 22.7% are defective. A bar chart of the target confirms this. This imbalance means accuracy alone is a misleading metric; AUC is used instead.

**Feature Histograms by Class:**

Histograms were plotted for all 21 features, with each chart overlaying the distribution of defective vs. non-defective modules. Key observations:
- Features like `v(g)` (cyclomatic complexity), `loc` (lines of code), `branchCount`, `v` (Halstead volume), and `e` (Halstead effort) show clear separation between classes — defective modules consistently have higher values
- Features like `lOBlank` (blank lines) and `l` (program level) show heavy overlap between classes, indicating weaker predictive power
- All features are **right-skewed** before preprocessing, with a small number of very large, complex modules dominating the upper range

**Before vs. After Preprocessing:**

Side-by-side histograms for four representative features (`loc`, `v(g)`, `v`, `e`) were plotted before and after the log1p + StandardScaler transformation, confirming that distributions become significantly more symmetric and centered.

---

### Problem Formulation

- **Input:** 21 numerical software complexity features (after log1p + StandardScaler)
- **Output:** Probability score between 0 and 1 representing likelihood of a defect

**Model:**

| Model | Reason for Choosing |
|---|---|
| Random Forest Classifier | Handles non-linear feature interactions, outputs calibrated probabilities, robust to outliers, provides feature importance rankings out of the box |

A single model was used for this project. Random Forest was selected as it is a strong, interpretable baseline that naturally handles the tabular structure of this data without requiring extensive hyperparameter tuning.

**Hyperparameters:**

| Parameter | Value | Reason |
|---|---|---|
| `n_estimators` | 200 | More trees = lower variance; 200 balances accuracy and compute time |
| `class_weight` | `'balanced'` | Compensates for the 77/23 class imbalance by increasing the penalty for misclassifying defective modules |
| `max_depth` | `None` | Trees grow fully; ensemble averaging prevents overfitting |
| `random_state` | 42 | Reproducibility |
| `n_jobs` | -1 | Uses all available CPU cores for faster training |

No explicit loss function is tuned by the user — Random Forest minimizes Gini impurity internally at each split.

---

### Training

- **Software:** Python 3.13, scikit-learn, pandas, numpy, run in Jupyter Notebook
- **Hardware:** Standard CPU (no GPU required — Random Forest does not use gradient descent)
- **Training time:** Approximately 2–4 minutes for 200 trees on ~71,000 rows with 21 features
- **Training curves:** Random Forest does not produce epoch-based loss curves. Performance was monitored by evaluating AUC on the validation set after training completed
- **Stopping criterion:** Not applicable — Random Forest trains a fixed number of trees (`n_estimators=200`) and stops automatically
- **Difficulties:**
  - *Class imbalance:* The model initially predicted "no defect" for most samples due to the 77/23 split. Resolved by setting `class_weight='balanced'`
  - *Skewed features:* Raw feature distributions were heavily right-skewed, which can mislead tree splits. Resolved with log1p transformation before training

---

### Performance Comparison

**Key Metric:** Area Under the ROC Curve (AUC) — chosen because it measures the model's ability to rank defective modules above non-defective ones at all classification thresholds, regardless of the class imbalance.

| Split | AUC | Accuracy |
|---|---|---|
| Validation | ~0.77 | ~0.81 |
| Hold-out Test | ~0.77 | ~0.81 |

**Note on accuracy:** An accuracy of 0.81 is not particularly meaningful here because always predicting "no defect" would yield ~0.77 accuracy for free. AUC is the meaningful metric.

**Visualizations produced:**
- ROC Curve on validation set (AUC annotated on plot)
- Confusion Matrix on validation set showing true/false positives and negatives
- Feature Importance bar chart showing `v(g)`, `loc`, `branchCount` as top predictors

---

### Conclusions

- A Random Forest Classifier trained on NASA software complexity metrics can predict software defects with an AUC of ~0.77, well above the 0.50 random baseline, demonstrating that these metrics carry real predictive signal
- Defective software modules are consistently more complex — higher cyclomatic complexity, more lines of code, and more branching are the strongest indicators of defects
- Class imbalance is a significant challenge; simply predicting the majority class yields high accuracy but is practically useless. Addressing imbalance explicitly (via `class_weight='balanced'`) is essential
- Log1p transformation meaningfully improves data quality by compressing the extreme right skew present in all features

---

### Future Work

- **Try gradient boosting models** (XGBoost, LightGBM, CatBoost) — these consistently outperform Random Forest on tabular Kaggle competitions and could push AUC above 0.80
- **Apply SMOTE** (Synthetic Minority Oversampling Technique) as an alternative approach to handle class imbalance and compare against `class_weight='balanced'`
- **Tune the decision threshold** — the default 0.5 cutoff is not optimal when classes are imbalanced; tuning it can significantly improve recall on defective modules
- **Feature engineering** — create interaction terms (e.g., `v(g) × loc`) or ratios that may capture defect risk better than individual features
- **Cross-validation** — use k-fold cross-validation instead of a single train/val split for more reliable performance estimates

---

## How to Reproduce Results

### Overview of Files in Repository

```
software-defect-prediction/
├── Software_Defect_Prediction.ipynb   # Main notebook: all steps from loading to submission
├── train.csv                          # Training data — download from Kaggle (see below)
├── test.csv                           # Kaggle test data — download from Kaggle (see below)
├── submission.csv                     # Generated automatically when notebook is run
└── README.md                          # This file
```

- `united.ipynb` — The single notebook containing all project steps: data loading, feature analysis, visualization, preprocessing, model training, evaluation, and Kaggle submission file generation. Each section is clearly labeled with markdown headers and inline comments.

---

### Software Setup

**Required packages:**

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```
---

### Data

1. Go to the Kaggle competition data page: https://www.kaggle.com/competitions/playground-series-s3e23/data
2. Accept the competition rules (free Kaggle account required)
3. Download `train.csv` and `test.csv`
4. Place both files in the same directory as `Untitled.ipynb`

No additional preprocessing scripts are needed — all preprocessing is handled inside the notebook.

---


### Performance Evaluation

All metrics are computed and displayed automatically inside the notebook after training:

- **AUC score** on the validation set (primary Kaggle metric) — printed to output
- **Accuracy** on the validation set — printed to output
- **Full classification report** (precision, recall, F1 per class) — printed to output
- **ROC Curve** — plotted inline
- **Confusion Matrix** — plotted inline
- **Feature Importance chart** — plotted inline

The `submission.csv` generated at the end of the notebook can be submitted directly to Kaggle at:
https://www.kaggle.com/competitions/playground-series-s3e23/submit

---

