"""Synthetic data generators for FinGuard.

Produces two CSVs:
  data/upi_transactions.csv  - UPI-style transactions with injected fraud patterns
  data/credit_risk.csv       - loan applicants with a default label

NOTE: both datasets are SYNTHETIC. No real UPI data is public, so the fraud
patterns below are modelled on commonly reported scam types. State this clearly
in your report. For stronger validation, also test on PaySim / Give Me Some Credit.

Usage:
    python src/generate_data.py                 # defaults
    python src/generate_data.py --n-upi 50000 --n-credit 20000 --seed 7
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------
# UPI transactions
# --------------------------------------------------------------------------
PAYEE_CATS = ["individual", "retail", "utility", "medical", "education", "other"]
LEGIT_CAT_P = [0.45, 0.25, 0.10, 0.05, 0.05, 0.10]
FRAUD_CAT_P = [0.50, 0.20, 0.0, 0.03, 0.0, 0.27]   # fraudsters rarely use utility/education payees


def generate_upi(n=30000, fraud_rate=0.015, seed=42, emergency_rate=0.012):
    """Generate UPI-like transactions.

    Fraud patterns injected:
      1. collect_scam   - fake 'collect request' to a new payee, mid/large amount
      2. fake_qr        - QR/VPA to a very new merchant VPA, odd amounts
      3. mule_passthru  - high velocity burst, new device, round amounts at night
      4. sim_swap       - new device + SIM change flag, large amount

    Also injects GENUINE EMERGENCY withdrawals (fraud_type == 'legit_emergency',
    label 0): night-time, huge amount, account almost emptied, sometimes a young
    account or recent SIM change. These are the hard false-positive cases.
    """
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    def base(m):
        return pd.DataFrame({
            "hour": rng.choice(24, size=m, p=_hour_probs()),
            "day_of_week": rng.integers(0, 7, size=m),
            "account_age_days": rng.gamma(shape=3.0, scale=250.0, size=m).astype(int) + 10,
            "avg_user_amount": np.round(rng.lognormal(6.2, 0.6, size=m), 2),
            "txn_count_1h": rng.poisson(0.3, size=m),
            "txn_count_24h": rng.poisson(2.0, size=m),
            "payee_vpa_age_days": rng.gamma(shape=2.5, scale=300.0, size=m).astype(int) + 5,
            # context signals: how often this user has paid this payee before (0 = first time)
            "payee_prior_txn_count": np.where(rng.random(m) < 0.12, 0, 1 + rng.poisson(6, m)),
            "device_changed": (rng.random(m) < 0.03).astype(int),
            "sim_changed_7d": (rng.random(m) < 0.01).astype(int),
            "is_collect_request": (rng.random(m) < 0.08).astype(int),
            "payee_type": rng.choice(["person", "merchant", "bill"], size=m, p=[0.55, 0.35, 0.10]),
            "payee_category": rng.choice(PAYEE_CATS, size=m, p=LEGIT_CAT_P),
            "location_match": (rng.random(m) < 0.93).astype(int),   # txn from user's usual area
            "max_past_amount": 0.0,                                 # filled below
            "balance_after_ratio": np.clip(rng.beta(4, 2, size=m), 0, 1),
        })

    def set_max_past(d):
        d["max_past_amount"] = np.round(d["avg_user_amount"] * rng.lognormal(1.6, 0.5, size=len(d)), 2)

    n_emerg = int(n * emergency_rate)
    n_legit = n - n_fraud - n_emerg

    legit = base(n_legit)
    legit["amount"] = np.round(legit["avg_user_amount"] * rng.lognormal(0.0, 0.7, size=n_legit), 2)
    set_max_past(legit)
    legit["is_fraud"] = 0
    legit["fraud_type"] = "none"

    # ---- genuine EMERGENCY withdrawals: look risky on the surface but are legitimate.
    # (accident / hospital bill: night, huge amount, account nearly emptied, sometimes a
    #  young account or a recent SIM change). Context (known or medical payee, usual
    #  location, same device) is what separates them from account takeover.
    em = base(n_emerg)
    em["hour"] = rng.choice([0, 1, 2, 3, 4, 5, 22, 23, 10, 15], size=n_emerg,
                            p=[.12, .12, .14, .10, .08, .06, .06, .06, .13, .13])
    em["avg_user_amount"] = np.round(rng.lognormal(6.2, 0.6, size=n_emerg), 2)
    em["amount"] = np.round(em["avg_user_amount"] * rng.uniform(5, 30, n_emerg), 2)
    set_max_past(em)
    em["balance_after_ratio"] = rng.uniform(0.0, 0.12, n_emerg)
    em["account_age_days"] = np.where(rng.random(n_emerg) < 0.35, rng.integers(5, 90, n_emerg),
                                      em["account_age_days"])
    em["sim_changed_7d"] = (rng.random(n_emerg) < 0.35).astype(int)
    em["device_changed"] = (rng.random(n_emerg) < 0.15).astype(int)
    em["txn_count_1h"] = rng.integers(0, 3, n_emerg)
    em["txn_count_24h"] = rng.integers(0, 6, n_emerg)
    em["payee_category"] = rng.choice(PAYEE_CATS, size=n_emerg, p=[.25, .05, .05, .50, .0, .15])
    em["payee_prior_txn_count"] = np.where(rng.random(n_emerg) < 0.35, rng.integers(1, 15, n_emerg), 0)
    em["location_match"] = (rng.random(n_emerg) < 0.88).astype(int)
    em["is_collect_request"] = 0
    em["is_fraud"] = 0
    em["fraud_type"] = "legit_emergency"
    legit = pd.concat([legit, em], ignore_index=True)

    # ---- fraud rows, split across four patterns
    probs = [0.35, 0.25, 0.25, 0.15]
    kinds = rng.choice(["collect_scam", "fake_qr", "mule_passthru", "sim_swap"], size=n_fraud, p=probs)
    fr = base(n_fraud)
    fr["fraud_type"] = kinds
    fr["is_fraud"] = 1
    fr["amount"] = 0.0
    set_max_past(fr)
    fr["payee_category"] = rng.choice(PAYEE_CATS, size=n_fraud, p=FRAUD_CAT_P)
    fr["payee_prior_txn_count"] = np.where(rng.random(n_fraud) < 0.9, 0, rng.integers(1, 3, n_fraud))
    fr["location_match"] = (rng.random(n_fraud) < 0.55).astype(int)

    for kind in ["collect_scam", "fake_qr", "mule_passthru", "sim_swap"]:
        idx = fr.index[fr["fraud_type"] == kind]
        m = len(idx)
        if m == 0:
            continue
        if kind == "collect_scam":
            fr.loc[idx, "is_collect_request"] = 1
            fr.loc[idx, "location_match"] = (rng.random(m) < 0.9).astype(int)  # victim acts from home
            fr.loc[idx, "amount"] = np.round(fr.loc[idx, "avg_user_amount"] * rng.uniform(2, 12, m), 2)
            fr.loc[idx, "payee_vpa_age_days"] = rng.integers(1, 40, m)
        elif kind == "fake_qr":
            fr.loc[idx, "payee_type"] = "merchant"
            fr.loc[idx, "payee_vpa_age_days"] = rng.integers(0, 15, m)
            fr.loc[idx, "payee_prior_txn_count"] = 0
            fr.loc[idx, "payee_category"] = rng.choice(["retail", "other"], size=m, p=[0.7, 0.3])
            fr.loc[idx, "location_match"] = (rng.random(m) < 0.9).astype(int)  # victim scans locally
            fr.loc[idx, "amount"] = np.round(fr.loc[idx, "avg_user_amount"] * rng.uniform(1.5, 8, m), 2)
        elif kind == "mule_passthru":
            fr.loc[idx, "txn_count_1h"] = rng.integers(4, 15, m)
            fr.loc[idx, "txn_count_24h"] = rng.integers(10, 40, m)
            fr.loc[idx, "device_changed"] = (rng.random(m) < 0.6).astype(int)
            fr.loc[idx, "hour"] = rng.choice([0, 1, 2, 3, 4, 23], size=m)
            fr.loc[idx, "account_age_days"] = rng.integers(2, 60, m)
            fr.loc[idx, "amount"] = rng.choice([4999, 9999, 9500, 19999, 24999, 49999], size=m).astype(float)
            fr.loc[idx, "balance_after_ratio"] = rng.uniform(0.0, 0.05, m)
        elif kind == "sim_swap":
            fr.loc[idx, "sim_changed_7d"] = 1
            fr.loc[idx, "device_changed"] = 1
            fr.loc[idx, "location_match"] = (rng.random(m) < 0.3).astype(int)
            fr.loc[idx, "payee_prior_txn_count"] = 0
            fr.loc[idx, "amount"] = np.round(fr.loc[idx, "avg_user_amount"] * rng.uniform(5, 25, m), 2)
            fr.loc[idx, "balance_after_ratio"] = rng.uniform(0.0, 0.15, m)

    df = pd.concat([legit, fr], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # derived features
    df["amount_to_avg_ratio"] = np.round(df["amount"] / df["avg_user_amount"].clip(lower=1), 3)
    df["amount_to_max_past_ratio"] = np.round(df["amount"] / df["max_past_amount"].clip(lower=1), 3)
    df["is_new_payee"] = (df["payee_prior_txn_count"] == 0).astype(int)
    df = df.drop(columns=["max_past_amount"])
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5, 23]).astype(int)
    df["is_round_amount"] = (df["amount"] % 1000 == 999).astype(int) | (df["amount"] % 500 == 0).astype(int)
    df["is_round_amount"] = df["is_round_amount"].astype(int)

    # label noise + missing values so the preprocessing step has real work to do
    flip = rng.random(len(df)) < 0.002
    df.loc[flip, "is_fraud"] = 1 - df.loc[flip, "is_fraud"]
    for col in ["payee_vpa_age_days", "balance_after_ratio"]:
        miss = rng.random(len(df)) < 0.015
        df.loc[miss, col] = np.nan

    # timestamps over a 90-day window (drawn last, so every earlier column is unchanged)
    start = pd.Timestamp("2026-07-01")
    offsets = rng.integers(0, 90 * 24 * 3600, size=len(df))
    df["txn_time"] = (start + pd.to_timedelta(offsets, unit="s")).strftime("%Y-%m-%d %H:%M:%S")

    df.insert(0, "txn_id", [f"TXN{i:07d}" for i in range(len(df))])
    return df


def _hour_probs():
    """Daytime-heavy distribution of transaction hours."""
    w = np.array([1, 1, 1, 1, 1, 2, 4, 6, 8, 9, 9, 9, 9, 9, 8, 8, 8, 9, 10, 10, 9, 6, 3, 2], dtype=float)
    return w / w.sum()


# --------------------------------------------------------------------------
# Credit risk
# --------------------------------------------------------------------------
def generate_credit(n=15000, seed=42):
    """Generate loan applicants with a default label driven by a logistic model."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 70, n)
    income = np.round(rng.lognormal(10.9, 0.55, n), 0)            # annual INR
    employment_years = np.clip(rng.gamma(2.0, 3.0, n), 0, 40).round(1)
    loan_amount = np.round(income * rng.uniform(0.2, 2.5, n), 0)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], n)
    existing_debt = np.round(income * rng.beta(2, 5, n), 0)
    debt_to_income = np.round((existing_debt + loan_amount / (loan_term_months / 12)) / income, 3)
    credit_history_years = np.clip(rng.gamma(2.5, 3.0, n), 0, 35).round(1)
    num_credit_lines = rng.poisson(4, n)
    missed_payments_2y = rng.poisson(0.5, n)
    credit_utilization = np.clip(rng.beta(2, 3, n), 0, 1).round(3)
    num_inquiries_6m = rng.poisson(1.2, n)
    home_ownership = rng.choice(["rent", "own", "mortgage"], n, p=[0.45, 0.25, 0.30])
    loan_purpose = rng.choice(["personal", "education", "home", "business", "vehicle"], n,
                              p=[0.30, 0.15, 0.20, 0.15, 0.20])
    gender = rng.choice(["M", "F"], n, p=[0.6, 0.4])  # kept ONLY for the fairness check

    logit = (-4.9
             + 3.6 * np.clip(debt_to_income, 0, 3) / 3
             + 0.85 * missed_payments_2y
             + 2.6 * credit_utilization
             + 0.25 * num_inquiries_6m
             - 0.10 * credit_history_years
             - 0.06 * employment_years
             - 1.0 * (np.log(income) - 10.9)
             + 0.40 * (home_ownership == "rent")
             + 0.30 * (loan_purpose == "business")
             + rng.normal(0, 0.35, n))
    p_default = _sigmoid(logit)
    default = (rng.random(n) < p_default).astype(int)

    df = pd.DataFrame({
        "applicant_id": [f"APP{i:06d}" for i in range(n)],
        "age": age, "gender": gender, "annual_income": income,
        "employment_years": employment_years, "loan_amount": loan_amount,
        "loan_term_months": loan_term_months, "existing_debt": existing_debt,
        "debt_to_income": debt_to_income, "credit_history_years": credit_history_years,
        "num_credit_lines": num_credit_lines, "missed_payments_2y": missed_payments_2y,
        "credit_utilization": credit_utilization, "num_inquiries_6m": num_inquiries_6m,
        "home_ownership": home_ownership, "loan_purpose": loan_purpose,
        "default": default,
    })
    for col in ["employment_years", "credit_history_years"]:
        miss = rng.random(n) < 0.02
        df.loc[miss, col] = np.nan
    return df


def main():
    ap = argparse.ArgumentParser(description="Generate FinGuard synthetic datasets")
    ap.add_argument("--n-upi", type=int, default=30000)
    ap.add_argument("--fraud-rate", type=float, default=0.015)
    ap.add_argument("--n-credit", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    upi = generate_upi(args.n_upi, args.fraud_rate, args.seed)
    credit = generate_credit(args.n_credit, args.seed)
    upi.to_csv(DATA_DIR / "upi_transactions.csv", index=False)
    credit.to_csv(DATA_DIR / "credit_risk.csv", index=False)
    print(f"upi_transactions.csv : {upi.shape}, fraud rate = {upi['is_fraud'].mean():.4f}")
    print(f"credit_risk.csv      : {credit.shape}, default rate = {credit['default'].mean():.4f}")


if __name__ == "__main__":
    main()
