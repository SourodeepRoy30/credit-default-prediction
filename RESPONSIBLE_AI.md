# Responsible AI Documentation

**Project:** Credit Risk Intelligence Suite  
**Author:** Sourodeep Roy  
**Last Updated:** March 2026

---

## 1. PURE Framework Mapping

The PURE framework (Purposeful, Unbiased, Respectful, Explainable) is applied across all four project layers.

| Component | Purposeful | Unbiased | Respectful | Explainable |
|---|---|---|---|---|
| PD Model (NB01–06) | Clear business objective: predict probability of default to support lending decisions | WoE encoding applied uniformly across feature bins; temporal train/test split prevents leakage | Model outputs are probabilities, not binary approve/decline decisions — human review retained | SHAP TreeExplainer provides per-loan feature attribution for every prediction |
| LLM Signal Extraction (NB07–09) | Structured extraction only — LLM used to parse borrower intent, not to score creditworthiness directly | Extraction prompt validated for directional correctness; income_stability_mention signal noted as weak and non-discriminatory | Borrower descriptions processed locally; no text transmitted to external services | Five extracted signals are human-interpretable and individually auditable |
| Credit Memo Generator (NB10) | Memos are decision-support tools, not decision-making tools — recommendation field is advisory only | Memo grounded in SHAP values which attribute risk to financial features, not demographic proxies | All inference runs locally via Ollama; zero borrower data transmitted externally | Hallucination evaluation framework built in — SHAP consistency, feature accuracy, non-invention rate all measured |
| Early Warning System (NB11–15) | Designed as a portfolio-level monitoring tool, not an individual loan decision tool | EWS operates on sector-level aggregates, not individual borrower attributes | Macro and news signals sourced from public datasets (FRED, GDELT) | Trigger rule, threshold, weighting, and attribution all documented and configurable |

---

## 2. Risk Mitigations

| Risk | Likelihood | Impact | Mitigation Applied |
|---|---|---|---|
| LLM hallucination in credit memos | Medium | High — incorrect risk statements could influence lending decisions | Hallucination evaluation framework built and run on 200 memos. SHAP consistency 100%, non-invention rate 98%. Virtual Expert constrained to refuse unanswerable questions. |
| Proxy discrimination via text signals | Medium | High — narrative signals could encode demographic information | LLM extraction targets financial behaviour signals only (distress, purpose, repayment). No demographic inference. income_stability_mention noted as weak signal (0.5pp separation) and treated as non-predictive. |
| Data leakage across train/test split | Low | High — inflated performance metrics | Temporal split used throughout. WoE bins fitted on training set only. TF-IDF vectoriser fitted on training descriptions only. SHAP explainer fitted on training set only. |
| Over-reliance on model recommendations | Medium | High — credit decisions require human judgement | Credit memo recommendation field is explicitly labelled advisory. No automated approve/decline pipeline built. Confidence scoring framework designed to flag low-certainty loans for human review. |
| EWS false positives causing unnecessary intervention | High (noted) | Medium | Sensitivity analysis run at 1.0σ, 1.5σ, 2.0σ thresholds. Flag rates documented. High false positive rate attributed to synthetic signal inputs and disclosed prominently. Production deployment requires live GDELT feed. |
| Regulatory non-compliance (SR 11-7, ECOA) | Low | High | SHAP attribution provides model documentation consistent with SR 11-7 model risk management guidance. No protected class variables used in any model. Adverse action explanations derivable from SHAP values. |

---

## 3. Data Governance Statement

**Local inference only.** All LLM inference in this project runs via Ollama on the developer's local machine. No borrower descriptions, loan features, or model outputs are transmitted to any external API or cloud service for inference purposes.

**External APIs used for non-PII data only.** The FRED API is used to retrieve publicly available macroeconomic time series. The Groq API is used only for prompt development and testing on synthetic inputs — never on real loan data.

**No personal identifiable information retained.** The Lending Club dataset used in this project has borrower names, addresses, and full SSNs removed by the data provider prior to public release. Zip codes are retained in the raw data but are not used in any model feature.

**Data sourcing.** Raw loan data sourced from the public Lending Club dataset available on Kaggle. FRED macroeconomic data sourced via the Federal Reserve Economic Data API. GDELT news data collection was attempted via the public GDELT API (rate-limited; synthetic fallback used — see below).

**Synthetic data disclosure.** The GDELT news sentiment stream used in the Early Warning System is synthetic, generated from representative sector headlines based on known 2015–2018 macroeconomic events. This is disclosed in NB11, NB13, NB14, and NB15. Production deployment of the EWS requires replacement of synthetic signals with live GDELT API data.

---

## 4. Known Limitations and Mitigations

| Limitation | Nature | Mitigation / Disclosure |
|---|---|---|
| Description coverage bias | 9.3% of loans have borrower descriptions, concentrated in the training window. Defaulters are underrepresented in the description-having subset (7.28% vs 9.86%). | LLM signal models evaluated on internal split of description-having subset only. Results explicitly noted as not directly comparable to structured models. Bias documented in NB07 results summary. |
| Synthetic EWS signals | GDELT API rate-limited during data collection. News sentiment stream generated synthetically. | Disclosed in NB11, NB13, NB14. Flag rates noted as elevated due to stress-biased synthetic inputs. Production system requires live feed. |
| Temporal test set granularity | Test set loan issue dates available at year level only, not month level. EWS backtest conducted at annual granularity. | Reduces backtest resolution. Noted in NB15 limitations. Monthly granularity would require raw issue_d column preservation in data pipeline. |
| Small sector sample sizes | Small business (n=3,081) and medical (n=3,684) sectors have limited loans in the 2015 training baseline. | Confidence intervals on sector-level default rates are wide. Results interpreted directionally, not as precise estimates. |
| LLM recommendation leniency | Credit memo generator defaults to APPROVE WITH CONDITIONS rather than DECLINE for VERY HIGH risk loans. Recommendation agreement 42.6%. | Noted as prompt limitation. Production system would apply rule-based override layer: VERY HIGH risk band → DECLINE regardless of LLM recommendation. |
| 3-year EWS evaluation window | Backtest covers 2016–2018 only. Three data points per sector is insufficient for robust statistical validation. | Noted in NB15. Longer evaluation window requires earlier loan vintage data or forward-looking deployment. |
| No protected class fairness audit | Fairness analysis across demographic groups not conducted — dataset does not include race, gender, or age. | No protected class variables used in any model feature. Adverse action explanations are derivable from SHAP values for any individual loan. Full fairness audit would require demographic data not present in the Lending Club public dataset. |

---

## 5. Intended and Non-Intended Use

**Intended use:**
- Portfolio-level credit risk monitoring and trend analysis
- Decision-support tooling for credit analysts — augmenting, not replacing, human judgement
- Research and demonstration of LLM integration patterns in financial risk contexts
- Educational reference for responsible ML system design in regulated industries

**Not intended for:**
- Automated approve/decline decisions without human review
- Individual consumer credit scoring as a standalone system
- Deployment in any regulated lending context without full model validation, fairness audit, and regulatory review
- Any use case involving real-time inference on live borrower data without appropriate data governance controls in place

---

*This document should be reviewed and updated before any production deployment of any component of this system.*
