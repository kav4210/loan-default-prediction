"""Loan default prediction on Lending Club, evaluated with ROC-AUC on a time-based holdout.

Usage:  python train.py --data accepted_2007_to_2018Q4.csv [--holdout-frac 0.2] [--out artifacts]

Design choices (each one is something an interviewer can ask about):
- Target: 1 = Charged Off / Default, 0 = Fully Paid. Loans still Current/Late are dropped (outcome not known yet).
- Features: only things known AT ORIGINATION. Post-origination columns (total_pymnt, recoveries, last_pymnt_*, ...) leak
  the answer and are never used.
- Split: by issue date. Train on older loans, test on the most recent ones, which mimics real deployment.
  A random split would let the model see the future and overstate performance.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "open_acc", "revol_bal", "revol_util", "total_acc", "delinq_2yrs", "inq_last_6mths", "pub_rec"]
CATEGORICAL = ["term", "grade", "home_ownership", "verification_status", "purpose"]
GOOD, BAD = {"Fully Paid"}, {"Charged Off", "Default"}

def load(path, chunksize=250_000):
    """Read in chunks and keep only loans with a known outcome, so a 2M-row file fits in a few GB of RAM."""
    cols = NUMERIC + CATEGORICAL + ["loan_status", "issue_d"]; keep = []
    for ch in pd.read_csv(path, usecols=cols, chunksize=chunksize):
        keep.append(ch[ch["loan_status"].isin(GOOD | BAD)])
    return pd.concat(keep, ignore_index=True)

def prepare(df):
    df = df[df["loan_status"].isin(GOOD | BAD)].copy()
    df["default"] = df["loan_status"].isin(BAD).astype(int)
    df["issue_date"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    if not pd.api.types.is_numeric_dtype(df["int_rate"]): df["int_rate"] = df["int_rate"].astype(str).str.rstrip("%").astype(float)
    if not pd.api.types.is_numeric_dtype(df["revol_util"]): df["revol_util"] = df["revol_util"].astype(str).str.rstrip("%").astype(float)
    df["term"] = df["term"].astype(str).str.strip()
    return df.sort_values("issue_date").reset_index(drop=True)

def time_split(df, holdout_frac):
    cut = int(len(df) * (1 - holdout_frac)); cut_date = df.loc[cut, "issue_date"]
    train, test = df[df.issue_date < cut_date], df[df.issue_date >= cut_date]   # strict date boundary, no month straddles
    assert train.issue_date.max() < test.issue_date.min()
    return train, test

def build_model():
    pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
                             ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), CATEGORICAL)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--holdout-frac", type=float, default=0.2); ap.add_argument("--out", default="artifacts")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True, parents=True)
    df = prepare(load(a.data))
    train, test = time_split(df, a.holdout_frac)
    m = build_model().fit(train[NUMERIC + CATEGORICAL], train["default"])
    p = m.predict_proba(test[NUMERIC + CATEGORICAL])[:, 1]
    # sanity baseline: a one-feature model using only the lender's own interest rate
    base = roc_auc_score(test["default"], test["int_rate"])
    res = {"roc_auc": round(float(roc_auc_score(test["default"], p)), 4), "baseline_roc_auc_int_rate_only": round(float(base), 4),
           "train_loans": int(len(train)), "heldout_loans": int(len(test)), "heldout_default_rate": round(float(test["default"].mean()), 4),
           "train_default_rate": round(float(train["default"].mean()), 4),
           "train_period": [str(train.issue_date.min().date()), str(train.issue_date.max().date())],
           "heldout_period": [str(test.issue_date.min().date()), str(test.issue_date.max().date())]}
    (out / "metrics.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))

if __name__ == "__main__": main()
