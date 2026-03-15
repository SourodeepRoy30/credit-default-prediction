# Credit Risk Intelligence Suite

> A production-grade credit risk system combining a PD model, LLM-powered explainability, and a multi-modal Early Warning System — built on 1.3 million Lending Club loans.

---

## The Finding

Structured financial features (WoE Logistic Regression, AUROC 0.767) outperform LLM-extracted narrative signals (AUROC 0.517) for probability of default prediction on the description-having loan subset — consistent with credit risk literature prioritising hard financial data over borrower narratives. Adding a FRED macro stream to a FinBERT news signal improves Early Warning System recall from 0.82 to 1.00 across five loan sectors over 2016–2018.

---

## What This Project Builds

Most credit risk projects stop at a model with a good AUROC. This one goes three layers deeper.

**Layer 1 — Production PD Model.** Standard but rigorous: Weight of Evidence encoding across 14 features, Logistic Regression and XGBoost trained on 829k loans, evaluated with AUROC, Gini, and KS. AUROC 0.767.

**Layer 2 — LLM Signal Extraction.** 120,000 borrower descriptions cleaned and analysed. A local LLM (Ollama/llama3.2:3b) extracts five structured signals per description — financial distress flag, purpose clarity, income stability, repayment confidence, overall sentiment — at 99.9% parse rate across 5,000 loans. Four model variants compared. Null result reported honestly.

**Layer 3 — Explainable Credit Memos.** SHAP-grounded credit memos generated for 200 loans using a local LLM. Hallucination evaluation across four dimensions: SHAP consistency 100%, feature accuracy 100%, non-invention rate 98%, grounding rate 94%. Virtual Expert query layer lets analysts interrogate individual loan decisions in natural language.

**Layer 4 — Multi-Modal Early Warning System.** FinBERT sentiment scores combined with FRED macro signals (unemployment, credit card delinquency, fed funds rate, consumer sentiment) into a sector-level EWS. Backtested against actual 2016–2018 default rates. Recall 1.00 — the system never missed a year where defaults rose. Multi-modal vs news-only recall lift: +0.18.

---

## Results

| Component | Metric | Value |
|---|---|---|
| WoE Logistic Regression | AUROC | 0.767 |
| WoE Logistic Regression | Gini | 0.534 |
| WoE Logistic Regression | KS | 0.380 |
| TF-IDF Baseline (text only) | AUROC | 0.593 |
| LLM Signals Only | AUROC | 0.517 |
| WoE + LLM XGBoost | AUROC | 0.736 |
| LLM Extraction | Parse Rate | 99.9% (4,993/5,000) |
| Credit Memo | SHAP Consistency | 100% |
| Credit Memo | Non-Invention Rate | 98.0% |
| Virtual Expert | Grounding Rate | 94.0% |
| Virtual Expert | Accuracy Rate | 100.0% |
| EWS (Multi-Modal) | Recall | 1.00 |
| EWS (Multi-Modal) | Precision | 0.73 |
| EWS vs News-Only | Recall Lift | +0.18 |

---

## Repository Structure

```
credit-default-prediction/
│
├── notebooks/
│   ├── 01_Raw_EDA.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_EDA_Clean.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Logistic_Regression.ipynb
│   ├── 06_Gradient_Boosting.ipynb
│   ├── 07_Text_EDA.ipynb
│   ├── 08_LLM_Extraction.ipynb
│   ├── 09_Combined_PD_Model.ipynb
│   ├── 10_Credit_Memo_Generator.ipynb
│   ├── 11_GDELT_Data_Collection.ipynb
│   ├── 12_FRED_Macro_Signals.ipynb
│   ├── 13_FinBERT_Sentiment.ipynb
│   ├── 14_EWS_Ensemble_Construction.ipynb
│   └── 15_EWS_Historical_Backtest.ipynb
│
├── models/
│   ├── lr_model.pkl
│   ├── xgb_model.pkl
│   └── shap_explainer.pkl
│
├── data/
│   ├── processed/
│   │   ├── df_train.pkl
│   │   ├── df_test.pkl
│   │   ├── woe_mappings.pkl
│   │   ├── llm_signals_20k.pkl
│   │   ├── memos_200.pkl
│   │   ├── finbert_sentiment_monthly.pkl
│   │   ├── fred_macro_score.pkl
│   │   ├── ews_scores.pkl
│   │   └── actual_default_rates_by_sector.csv
│
├── prompts/
│   ├── extraction_prompt.txt
│   ├── credit_memo_prompt.txt
│   └── virtual_expert_prompt.txt
│
├── outputs/
│   ├── model_comparison.csv
│   ├── memo_evaluation.csv
│   ├── virtual_expert_evaluation.csv
│   ├── ews_backtest_comparison.csv
│   └── *.png
│
├── Setup_Check.ipynb
├── RESPONSIBLE_AI.md
├── requirements.txt
└── README.md
```

---

## Methodology

### PD Model (NB01–06)

Raw Lending Club data (1,348,099 loans) cleaned, deduplicated, and split on a temporal boundary (training: loans issued before cutoff, test: after). 14 features encoded using Weight of Evidence binning. Logistic Regression trained as the interpretable baseline; XGBoost as the performance model. SHAP TreeExplainer used for post-hoc attribution.

### LLM Signal Extraction (NB07–09)

The `desc` field in Lending Club data was discontinued partway through the dataset — only 9.3% of loans have descriptions, concentrated in the training window. A custom cleaning pipeline strips Lending Club boilerplate and HTML artefacts. A structured extraction prompt extracts five signals per description using Ollama (llama3.2:3b), validated at 99.9% parse rate. All four model variants evaluated on an internal 80/20 stratified split of the description-having subset.

### Credit Memo Generator (NB10)

For each loan, SHAP values identify the top three risk drivers. A four-section memo is generated via Ollama, grounded strictly in SHAP context and loan features. Hallucination evaluation uses four automated checks: SHAP factor mention in risk analysis (regex-normalised for hyphen/underscore variants), recommendation alignment with risk band, grade citation accuracy, and dollar amount invention rate. Virtual Expert constrains LLM responses to SHAP context only.

### Early Warning System (NB11–15)

Two signal streams combined at 60/40 weighting (news-led, macro-confirming). FinBERT (ProsusAI/finbert) scores sector-specific financial headlines monthly. FRED macro signals z-scored against 2015 baseline and clipped at ±3 to prevent near-zero-variance series (FEDFUNDS, DRCCLACBS) from dominating the composite. EWS trigger: score > (2015 mean + 1.5σ), held for minimum 3 months. Backtested against actual test-set default rates by sector and year.

---

## Stack

| Category | Tools |
|---|---|
| Core ML | scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| LLM Inference | Ollama (llama3.2:3b), Groq API |
| NLP | FinBERT (ProsusAI), VADER, sentence-transformers |
| Macro Data | FRED API (fredapi) |
| News Data | GDELT API (synthetic fallback — see RESPONSIBLE_AI.md) |
| Data | pandas, numpy, scipy |
| Visualisation | matplotlib, seaborn |
| Environment | Python 3.14, VS Code, Jupyter |

---

## Responsible AI

All LLM inference runs locally via Ollama. No borrower data is transmitted to external services. SHAP is used throughout to ensure model decisions are auditable. Known limitations — synthetic signal inputs, description coverage bias, small EWS evaluation window — are documented in `RESPONSIBLE_AI.md`.

The credit memo generator includes a hallucination evaluation framework precisely because ungrounded LLM outputs in a credit context carry regulatory and reputational risk. The Virtual Expert is explicitly constrained to refuse answers not supported by the loan context.

---

## Known Limitations

- Borrower descriptions available for only 9.3% of loans, concentrated before the temporal test cutoff — LLM signal models evaluated on internal split only, not directly comparable to structured models
- Description-having subset underrepresents defaulters (7.28% vs 9.86% coverage) — mild selection bias noted
- GDELT API rate-limited during collection — EWS news stream uses synthetic signals based on known macro events; production deployment requires live feed
- EWS evaluated at annual granularity due to test set lacking monthly issue dates — reduces backtest resolution
- EWS flag rates (52–75%) elevated due to synthetic signal bias; real GDELT data would produce lower false positive rates

---

## Setup

```bash
git clone https://github.com/<your-username>/credit-default-prediction
cd credit-default-prediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install and start Ollama
# https://ollama.ai
ollama pull llama3.2:3b
ollama serve

# Set API keys
cp .env.example .env
# Add FRED_API_KEY and GROQ_API_KEY to .env

# Verify environment
jupyter notebook Setup_Check.ipynb
```

Run notebooks in sequence 01 → 15. Each notebook loads artefacts from the previous stage — do not skip steps.

---

## Data

Download the Lending Club loan dataset from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club). Place the raw CSV in `data/raw/`. The `desc` column must be retained — some Kaggle versions drop it.

---

*Built by Sourodeep Roy — [LinkedIn](https://www.linkedin.com/in/sourodeeproy/)*
