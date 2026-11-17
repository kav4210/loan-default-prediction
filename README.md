# Loan Default Prediction (Lending Club)

Logistic regression predicting whether a loan is charged off, evaluated with ROC-AUC on a **time-based holdout** (train on older loans, test on the most recent ones).

## Run
```
# 1. put accepted_2007_to_2018Q4.csv (Lending Club, from Kaggle) in this folder
# 2. Docker, no Python install needed (PowerShell):
docker run --rm -v "${PWD}:/work" -w /work python:3.12-slim bash -c "pip install -q pandas numpy scikit-learn && python train.py --data accepted_2007_to_2018Q4.csv"
# tests (run on a small fake file with the same columns, no download needed):
pip install -r requirements.txt && pytest tests
```
Writes `artifacts/metrics.json`.

## Design choices
- **Target:** Charged Off / Default = 1, Fully Paid = 0. Loans still Current or Late are dropped because their outcome is not known.
- **No leakage:** only features known at origination. Columns like `total_pymnt`, `recoveries` and `last_pymnt_*` reveal the answer and are never used.
- **Time-based split:** a random split lets the model train on loans issued after the ones it is tested on, which overstates performance. The split uses a strict date boundary and the test asserts no overlap.
- **Baseline:** the output includes the ROC-AUC of the lender's own interest rate alone, so the model's number can be read against something.

## Results
Run on the full Lending Club accepted-loans file (2007 to 2018 Q4), produced by `train.py`, saved in `artifacts/metrics.json`:

| | Value |
|---|---|
| Training loans | 1,062,380 (issued 2007-06 to 2016-09) |
| Held-out loans | **282,970** (issued 2016-10 to 2018-12, the most recent) |
| **ROC-AUC on held-out** | **0.693** |
| Baseline: interest rate alone | 0.677 |
| Default rate, train / held-out | 19.5% / 21.8% |

How to read this honestly: the model beats a one-feature interest-rate baseline by only about 0.017 AUC. That is a realistic result for Lending Club, where the lender's own interest rate already encodes most of the risk it knows about at origination. This is a solid, leakage-free evaluation, not a high-performing model.

## Limits
Held-out set is selected by outcome: recent 60-month loans are mostly still current and so are excluded, and recent loans that have already resolved skew toward early charge-offs, which likely inflates the 21.8% held-out default rate; logistic regression only; no hyperparameter tuning; accepted loans only (rejected applicants are not in the data, so there is selection bias); default behavior in 2007 to 2018 may not hold in other rate environments.
