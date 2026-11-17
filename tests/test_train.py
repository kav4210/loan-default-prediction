import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, subprocess, json, train

def fake(n=6000, seed=0):
    r = np.random.default_rng(seed); dates = pd.to_datetime("2012-01-01") + pd.to_timedelta(r.integers(0, 365*5, n), "D")
    ir = r.uniform(5, 28, n); p = 1 / (1 + np.exp(-(ir - 17) / 3))
    st = np.where(r.random(n) < p * 0.5, "Charged Off", "Fully Paid"); st[r.random(n) < 0.05] = "Current"
    d = pd.DataFrame({"loan_status": st, "issue_d": dates.strftime("%b-%Y"), "int_rate": [f"{x:.2f}%" for x in ir], "term": r.choice([" 36 months", " 60 months"], n),
        "grade": r.choice(list("ABCDEFG"), n), "home_ownership": r.choice(["RENT", "OWN", "MORTGAGE"], n), "verification_status": r.choice(["Verified", "Not Verified"], n),
        "purpose": r.choice(["debt_consolidation", "credit_card", "other"], n), "revol_util": [f"{x:.1f}%" for x in r.uniform(0, 100, n)]})
    for c in ["loan_amnt", "installment", "annual_inc", "dti", "open_acc", "revol_bal", "total_acc", "delinq_2yrs", "inq_last_6mths", "pub_rec"]: d[c] = r.uniform(0, 1000, n)
    d.loc[r.random(n) < .1, "annual_inc"] = np.nan; return d

def test_prepare_drops_unknown_outcomes_and_parses():
    df = train.prepare(fake()); assert set(df.loan_status) <= {"Fully Paid", "Charged Off"} and df.default.isin([0, 1]).all()
    assert df.int_rate.dtype == float and df.issue_date.is_monotonic_increasing

def test_time_split_has_no_overlap_and_right_size():
    df = train.prepare(fake()); tr, te = train.time_split(df, 0.2)
    assert tr.issue_date.max() < te.issue_date.min() and len(tr) + len(te) == len(df) and 0.15 < len(te) / len(df) < 0.25

def test_cli_end_to_end(tmp_path):
    fake().to_csv(tmp_path / "l.csv", index=False)
    subprocess.run([sys.executable, os.path.join(os.path.dirname(train.__file__), "train.py"), "--data", str(tmp_path / "l.csv"), "--out", str(tmp_path / "o")], check=True, capture_output=True)
    m = json.load(open(tmp_path / "o" / "metrics.json")); assert 0.5 < m["roc_auc"] <= 1 and m["heldout_loans"] > 0 and m["train_period"][1] < m["heldout_period"][0]
