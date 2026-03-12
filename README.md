# Credit Default Prediction — End-to-End Acquisition Scorecard

**Dataset:** Lending Club Public Loan Data (2007–2018) | 1,348,099 Resolved Loans  
**Stack:** Python · Logistic Regression · XGBoost · LightGBM · WoE/IV · SHAP  

---

## Research Question

Can a logistic regression scorecard trained on historical loan data generalise to future originations — and does gradient boosting offer meaningful improvement over a well-specified WoE-encoded scorecard?

---

## Methodology Overview

Most published results on the Lending Club dataset report AUROC values of 0.85–0.97. These are almost universally the product of random train/test splitting, retention of post-origination leakage features, or SMOTE oversampling applied before splitting. This project enforces the methodological controls that production credit models require:

- **Temporal train/test split** — 2007–2015 train, 2016–2018 test. No future data informs past predictions.
- **Leakage removal** — 36 post-origination columns removed (payment history, recovery amounts, updated FICO scores). These would not exist at the point of a lending decision.
- **WoE encoding fitted on training data only** — WoE bin parameters are never refit on the test set.

---

## Project Structure

```
credit-default-prediction/
│
├── data/
│   ├── raw/                    # Raw Lending Club loan.csv (not tracked)
│   └── processed/              # Cleaned and encoded datasets (not tracked)
│
├── notebooks/
│   ├── 01_raw_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_EDA_clean.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_logistic_regression.ipynb
│   └── 06_gradient_boosting.ipynb
│
├── outputs/                    # All charts and plots
├── requirements.txt
└── README.md
```

---

## Notebooks

| Notebook | Description | Key Output |
|----------|-------------|------------|
| 01 | Raw data exploration — target distribution, missing values, temporal analysis | Temporal split defined: 2007–2015 train / 2016–2018 test |
| 02 | Data cleaning — leakage removal, imputation, outlier correction | 68 features retained, 0 missing values |
| 03 | EDA on clean data + reject inference | Selection bias confirmed: rejected borrowers avg FICO 637 vs accepted 690 |
| 04 | WoE encoding + IV-based feature selection | 14 features selected from 24 candidates |
| 05 | Logistic regression PD model + scorecard + ECL | Test Gini 0.408, KS 0.295, ECL $932M |
| 06 | Gradient boosting challengers + SHAP + model comparison | XGBoost Test Gini 0.418 (+0.010 vs LR) |

---

## Results

### Discrimination

| Model | Train Gini | Test Gini | Test KS | Gini Drop |
|-------|-----------|-----------|---------|-----------|
| Logistic Regression | 0.4286 | 0.4084 | 0.2946 | 0.0202 |
| LightGBM | 0.4678 | 0.4174 | 0.3033 | 0.0504 |
| XGBoost | 0.4580 | 0.4184 | 0.3042 | 0.0396 |

Test Gini of 0.408 sits within the standard 0.30–0.50 range for production retail credit scorecards and is directly at par with published results using equivalent temporal methodology on the same dataset.

Gradient boosting improves Test Gini by only +0.010 over logistic regression — the ROC curves of all three models are visually indistinguishable. This confirms that WoE encoding captures most of the available non-linear signal at application time.

### Calibration

Train set calibration is near-perfect. The test set shows systematic underprediction of 5–8pp per decile, attributable to a population default rate shift from 18.46% (train) to 22.42% (test) — a known vintage drift effect correctable via intercept recalibration in production.

### Scorecard

| Parameter | Value |
|-----------|-------|
| Score range | 377 – 529 |
| Base score | 444 |
| PDO | 20 |
| Target odds | 50:1 at 600 points |
| Dominant feature | sub_grade (63-point range: A1 = +37, G5 = −26) |

### ECL

| Model | Total ECL | ECL / EAD | Coverage of Actual Loss |
|-------|-----------|-----------|------------------------|
| Logistic Regression | $932M | 12.43% | 79.03% |
| LightGBM | $976M | 13.01% | 82.74% |
| XGBoost | $980M | 13.06% | 83.06% |
| Actual observed loss | $1,179M | 15.73% | 100% |

ECL computed as PD × LGD × EAD using empirical LGD of 64.75% derived from 153,065 training-period defaults.

---

## Feature Importance (SHAP — XGBoost)

| Rank | Feature | Mean |SHAP| | Direction |
|------|---------|-------------|-----------|
| 1 | sub_grade | 0.486 | Higher grade → higher default risk |
| 2 | term | 0.234 | 60-month → higher risk vs 36-month |
| 3 | fico_range_low | 0.154 | Higher FICO → lower default risk |
| 4 | dti | 0.137 | Higher DTI → higher default risk |
| 5 | log_annual_inc | 0.126 | Higher income → lower default risk |

SHAP ranking is consistent with IV ranking from feature selection — confirming internal consistency across the pipeline.

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Data

Download the Lending Club loan data from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) and place `loan.csv` in `data/raw/`.

### Run Order

Run notebooks in sequence: 01 → 02 → 03 → 04 → 05 → 06.  
Each notebook saves intermediate outputs to `data/processed/` for the next stage.

---

## Key Design Decisions

**Why temporal splitting?**  
Random splitting of a loan dataset allows future loans to inform past predictions, inflating AUROC. A temporal split mirrors real deployment conditions where a model trained on historical data is applied to future originations.

**Why WoE encoding for logistic regression?**  
Logistic regression assumes a linear relationship between each feature and the log-odds of default. WoE transforms each feature into log-odds space directly, satisfying this assumption by construction and enabling a fully interpretable points-based scorecard.

**Why logistic regression as champion?**  
Gradient boosting improved Test Gini by only +0.010. The marginal discrimination gain does not justify the loss of interpretability in a regulated context where model coefficients must be explainable to risk teams and regulators.

---

## Limitations

- **Reject inference** — model trained on accepted loans only (~10% of all applications). Predicted PDs will underestimate default risk for borrower profiles Lending Club historically rejected.
- **Fixed LGD** — a single empirical LGD of 64.75% applied to all loans. In practice LGD varies by loan purpose and macroeconomic cycle.
- **Calibration shift** — requires intercept recalibration before deployment in any period with a materially different population default rate from the training period.

---

## Author

**Sourodeep Roy**  
MSc Data Science & Analytics, University of Leeds (Distinction)  
[linkedin.com/in/sourodeeproy](https://linkedin.com/in/sourodeeproy) · [github.com/SourodeepRoy30](https://github.com/SourodeepRoy30)
