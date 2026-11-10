# Churn Risk Modeling

Two complementary supervised-learning pipelines for telecom decision support: a K-nearest-neighbors classifier for customer churn risk and a random-forest regressor for monthly-charge estimation.

## Decision questions

1. Which customers exhibit characteristics associated with a higher risk of churn?
2. How accurately can monthly charges be estimated from customer, contract, and service attributes?

## What is included

- Leakage-safe preprocessing with train-only imputation, scaling, and one-hot encoding.
- Stratified classification evaluation using accuracy and ROC-AUC.
- Regression evaluation using MAE and RMSE.
- Reusable pipelines that accept mixed numeric and categorical features.
- Synthetic tests that validate the workflow without distributing restricted source data.

## Portfolio artifacts

- [Rendered analysis notebook](notebook/analysis.html)
- [Portfolio report](reports/churn-risk-modeling-report.pdf)

The public repository includes the rendered notebook and report while intentionally excluding the working `.ipynb` file, restricted source data, local paths, and academic grading material.
## Repository structure

```text
├── data/README.md
├── src/models.py
├── tests/test_models.py
├── requirements.txt
└── README.md
```

## Run locally

```bash
python -m venv .venv
pip install -r requirements.txt
pytest tests/ -v --tb=short
```

## Modeling guardrails

- Model scores should be monitored by customer segment, contract type, and service mix.
- Churn predictions are prioritization signals, not deterministic statements about customers.
- Monthly-charge estimates should be compared with pricing rules before operational use.
- Retraining should use a time-aware validation strategy when production timestamps are available.

## Portfolio

More work: [larry-henry-jr.net](https://larry-henry-jr.net)
