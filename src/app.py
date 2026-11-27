"""FinGuard Streamlit demo.  Run from the project root:

    streamlit run src/app.py

Requires the models saved by notebooks/01_shared_training_evaluation.ipynb.

Tabs:
  1. UPI transaction check : Allow / Verify / Block decision with reasons for AND against risk
  2. Analyst alert queue   : flagged transactions from a held-out sample, ranked by risk,
                             with a confirm / dismiss feedback log
  3. Loan check            : credit score, band, decision and reason codes
  4. Batch + drift         : upload a CSV, get a scored and ranked file, and a data-drift (PSI) report
  5. Cost trade-off        : move the cost of a missed fraud and see the best threshold change
  6. Mule accounts         : graph-based detection of accounts that pass money on quickly
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import monitoring as mon  # noqa: E402
import train_utils as tu  # noqa: E402

st.set_page_config(page_title="FinGuard", page_icon="🛡️", layout="wide")
st.title("🛡️ FinGuard")
st.caption("Explainable ML for UPI fraud detection and loan credit risk scoring. "
           "Trained on synthetic data: a prototype, not a production decision system.")

REASONS = {
    "debt_to_income": "Debt-to-income ratio", "missed_payments_2y": "Missed payments",
    "credit_utilization": "Credit utilisation", "num_inquiries_6m": "Recent credit enquiries",
    "credit_history_years": "Credit history length", "employment_years": "Employment length",
    "annual_income": "Income", "loan_amount": "Loan amount", "existing_debt": "Existing debt",
    "amount": "Payment amount", "amount_to_avg_ratio": "Amount vs user's average",
    "amount_to_max_past_ratio": "Amount vs user's largest past payment",
    "txn_count_1h": "Transactions in the last hour", "txn_count_24h": "Transactions in 24h",
    "device_changed": "New device", "sim_changed_7d": "SIM changed recently", "is_new_payee": "New payee",
    "payee_prior_txn_count": "Previous payments to this payee", "payee_vpa_age_days": "Payee VPA age",
    "is_collect_request": "Collect request", "balance_after_ratio": "Balance left after payment",
    "is_night": "Night-time payment", "hour": "Hour of day", "account_age_days": "Account age",
    "is_round_amount": "Round amount", "location_match": "Usual location",
    "day_of_week": "Day of week",
}


def label(feature):
    base = feature.split("__", 1)[-1]
    for pre in ("payee_category_", "payee_type_", "home_ownership_", "loan_purpose_"):
        if base.startswith(pre):
            return f"{pre[:-1].replace('_', ' ').capitalize()}: {base[len(pre):]}"
    return REASONS.get(base, base)


def explain(pipe, row, k=4, direction="up"):
    top = tu.top_reasons(pipe, row, k=k, direction=direction)
    return [f"{label(f)} ({v:+.3f})" for f, v in top.items() if (v > 0) == (direction == "up")]


@st.cache_resource
def load(name):
    return tu.load_model(name)


try:
    upi_m, cr_m, sc_m = load("upi_fraud_rf"), load("credit_risk_rf"), load("credit_scorecard_lr")
except FileNotFoundError:
    st.error("Models not found. Run notebooks/01_shared_training_evaluation.ipynb first.")
    st.stop()

TIERS = upi_m["tiers"]
TIER_NOTE = {
    "Allow": ("success", "✅ ALLOW: the payment goes through instantly."),
    "Verify": ("warning", "🟠 VERIFY: pause and ask the customer to confirm (OTP / biometric / callback). "
                          "A genuine customer passes in seconds; a fraudster who only has the SIM does not."),
    "Block": ("error", "🚫 BLOCK / HOLD: very likely fraud. Hold the payment and send to an analyst."),
}

SCENARIOS = {
    "Custom": None,
    "Everyday payment": dict(amount=450.0, avg=800.0, max_past=4000.0, hour=15, payee_type="merchant",
                             category="retail", prior=8, vpa_age=700, acct_age=900, t1h=0, t24=2, bal=0.7,
                             dev=False, sim=False, collect=False, loc=True),
    "Emergency: hospital bill at 2 am": dict(amount=25000.0, avg=800.0, max_past=4000.0, hour=2,
                                             payee_type="merchant", category="medical", prior=0, vpa_age=400,
                                             acct_age=5, t1h=1, t24=2, bal=0.02, dev=False, sim=True,
                                             collect=False, loc=True),
    "Account takeover": dict(amount=25000.0, avg=800.0, max_past=4000.0, hour=2, payee_type="person",
                             category="other", prior=0, vpa_age=4, acct_age=400, t1h=3, t24=6, bal=0.02,
                             dev=True, sim=True, collect=False, loc=False),
}

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["UPI transaction check", "Analyst alert queue", "Loan check",
                                              "Batch + drift", "Cost trade-off", "Mule accounts"])

# ---------------------------------------------------------------- Tab 1
with tab1:
    scn = st.selectbox("Load an example scenario", list(SCENARIOS))
    d = SCENARIOS[scn] or dict(amount=2000.0, avg=800.0, max_past=4000.0, hour=12, payee_type="person",
                               category="individual", prior=3, vpa_age=300, acct_age=400, t1h=0, t24=2,
                               bal=0.5, dev=False, sim=False, collect=False, loc=True)
    k = scn  # changing the scenario resets the widgets

    c1, c2 = st.columns(2)
    amount = c1.number_input("Amount (INR)", 1.0, 1_000_000.0, d["amount"], key=f"amt{k}")
    avg_amt = c2.number_input("User's average amount (INR)", 1.0, 100_000.0, d["avg"], key=f"avg{k}")
    max_past = c1.number_input("User's largest past payment (INR)", 1.0, 1_000_000.0, d["max_past"], key=f"mp{k}")
    hour = c2.slider("Hour of day", 0, 23, d["hour"], key=f"hr{k}")
    cats = ["individual", "retail", "utility", "medical", "education", "other"]
    category = c1.selectbox("Payee category", cats, index=cats.index(d["category"]), key=f"cat{k}")
    ptypes = ["person", "merchant", "bill"]
    payee_type = c2.selectbox("Payee type", ptypes, index=ptypes.index(d["payee_type"]), key=f"pt{k}")
    prior = c1.number_input("Previous payments to this payee", 0, 200, d["prior"], key=f"pr{k}")
    vpa_age = c2.number_input("Payee VPA age (days)", 0, 5000, d["vpa_age"], key=f"va{k}")
    acct_age = c1.number_input("Sender account age (days)", 1, 5000, d["acct_age"], key=f"aa{k}")
    bal = c2.slider("Balance left after payment (fraction)", 0.0, 1.0, d["bal"], key=f"bal{k}")
    t1h = c1.number_input("Payments in last 1h", 0, 50, d["t1h"], key=f"t1{k}")
    t24 = c2.number_input("Payments in last 24h", 0, 100, d["t24"], key=f"t24{k}")
    dev = c1.checkbox("Device changed", d["dev"], key=f"dev{k}")
    sim = c2.checkbox("SIM changed in last 7 days", d["sim"], key=f"sim{k}")
    loc = c1.checkbox("Payment from the user's usual location", d["loc"], key=f"loc{k}")
    collect = c2.checkbox("Collect request", d["collect"], key=f"col{k}")

    if st.button("Check transaction", type="primary"):
        row = pd.DataFrame([{
            "hour": hour, "day_of_week": 2, "account_age_days": acct_age, "avg_user_amount": avg_amt,
            "txn_count_1h": t1h, "txn_count_24h": t24, "payee_vpa_age_days": vpa_age,
            "payee_prior_txn_count": prior, "is_new_payee": int(prior == 0), "device_changed": int(dev),
            "sim_changed_7d": int(sim), "is_collect_request": int(collect), "payee_type": payee_type,
            "payee_category": category, "location_match": int(loc), "balance_after_ratio": bal,
            "amount": amount, "amount_to_avg_ratio": amount / max(avg_amt, 1),
            "amount_to_max_past_ratio": amount / max(max_past, 1),
            "is_night": int(hour in (0, 1, 2, 3, 4, 5, 23)),
            "is_round_amount": int(amount % 1000 == 999 or amount % 500 == 0),
        }])[upi_m["features"]]
        p = float(upi_m["pipeline"].predict_proba(row)[0, 1])
        tier = str(tu.apply_tiers([p], TIERS)[0])
        a, b = st.columns(2)
        a.metric("Fraud probability", f"{p:.1%}")
        b.metric("Decision", tier)
        kind, msg = TIER_NOTE[tier]
        getattr(st, kind)(msg)
        up = explain(upi_m["pipeline"], row, 4, "up")
        down = explain(upi_m["pipeline"], row, 3, "down")
        st.write("**What raises the risk:**")
        for r in up or ["Nothing significant"]:
            st.write("- " + r)
        st.write("**What lowers the risk:**")
        for r in down or ["Nothing significant"]:
            st.write("- " + r)
        st.caption(f"Thresholds: Verify at {TIERS['verify']:.2f}, Block at {TIERS['block']:.2f}. "
                   "Numbers show each factor's push on the score (SHAP); compare them, do not read them as percentages.")

# ---------------------------------------------------------------- Tab 2
FEEDBACK = tu.REPORTS_DIR / "feedback_log.csv"


@st.cache_data(show_spinner="Scoring the held-out transactions...")
def scored_queue():
    df = pd.read_csv(tu.DATA_DIR / "upi_transactions.csv")
    X, y = df[upi_m["features"]], df["is_fraud"]
    _, Xte, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=tu.RANDOM_STATE)
    live, _ = tu.live_eval_split(Xte.index)          # analysts only see the "live" half
    Xte = Xte.loc[live]
    out = df.loc[Xte.index, ["txn_id", "amount", "hour", "payee_category", "is_fraud", "fraud_type"]].copy()
    out["risk"] = upi_m["pipeline"].predict_proba(Xte)[:, 1]
    out["decision"] = tu.apply_tiers(out["risk"].values, TIERS)
    return out, Xte


with tab2:
    q, Xte = scored_queue()
    st.write("Every transaction is scored automatically. Analysts only see the small flagged slice, "
             "ranked by risk. This shows the **live half of the held-out test split** (never used in training); the other half is kept for fairly comparing a retrained model.")
    counts = q["decision"].value_counts().reindex(["Allow", "Verify", "Block"], fill_value=0)
    a, b, c = st.columns(3)
    a.metric("Allowed", f"{counts['Allow']:,}")
    b.metric("Verify", f"{counts['Verify']:,}")
    c.metric("Block", f"{counts['Block']:,}")
    st.caption(f"{100 * (counts['Verify'] + counts['Block']) / len(q):.1f}% of {len(q):,} transactions "
               "need any human or customer attention.")

    show = st.radio("Queue", ["Block", "Verify"], horizontal=True)
    flagged = q[q["decision"] == show].sort_values("risk", ascending=False).head(50)
    st.dataframe(flagged[["txn_id", "amount", "hour", "payee_category", "risk"]].assign(
        risk=lambda x: x["risk"].round(3)), hide_index=True, width="stretch")

    if len(flagged):
        pick = st.selectbox("Open a case", flagged["txn_id"].tolist())
        row = Xte.loc[[flagged.index[flagged["txn_id"] == pick][0]]]
        st.write("**Why it was flagged:**")
        for r in explain(upi_m["pipeline"], row, 4, "up") or ["Nothing significant"]:
            st.write("- " + r)
        st.write("**What speaks for the customer:**")
        for r in explain(upi_m["pipeline"], row, 3, "down") or ["Nothing significant"]:
            st.write("- " + r)
        x, y = st.columns(2)
        for btn, verdict, col in ((x, "confirmed_fraud", "Confirm fraud"), (y, "genuine_customer", "It was the customer")):
            if col and btn.button(col, key=f"{verdict}{pick}"):
                tu.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                rec = pd.DataFrame([{"time": datetime.now().isoformat(timespec="seconds"),
                                     "txn_id": pick, "verdict": verdict}])
                rec.to_csv(FEEDBACK, mode="a", header=not FEEDBACK.exists(), index=False)
                st.success(f"Saved to reports/{FEEDBACK.name}. Analyst labels like these are what you "
                           "would use to retrain the model.")
    if FEEDBACK.exists():
        st.caption(f"Feedback log: {len(pd.read_csv(FEEDBACK))} labelled case(s).")

# ---------------------------------------------------------------- Tab 3
with tab3:
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", 21, 70, 35)
    income = c2.number_input("Annual income (INR)", 50_000, 10_000_000, 600_000, step=10_000)
    loan = c1.number_input("Loan amount (INR)", 10_000, 20_000_000, 800_000, step=10_000)
    term = c2.selectbox("Term (months)", [12, 24, 36, 48, 60], index=2)
    debt = c1.number_input("Existing debt (INR)", 0, 10_000_000, 150_000, step=10_000)
    emp = c2.number_input("Employment (years)", 0.0, 40.0, 4.0)
    hist = c1.number_input("Credit history (years)", 0.0, 35.0, 5.0)
    lines = c2.number_input("Credit lines", 0, 30, 4)
    missed = c1.number_input("Missed payments (2y)", 0, 20, 1)
    util = c2.slider("Credit utilisation", 0.0, 1.0, 0.45)
    inq = c1.number_input("Enquiries (6m)", 0, 20, 1)
    home = c2.selectbox("Home ownership", ["rent", "own", "mortgage"])
    purpose = c1.selectbox("Purpose", ["personal", "education", "home", "business", "vehicle"])
    if st.button("Assess applicant", type="primary"):
        dti = (debt + loan / (term / 12)) / income
        row = pd.DataFrame([{
            "age": age, "annual_income": income, "employment_years": emp, "loan_amount": loan,
            "loan_term_months": term, "existing_debt": debt, "debt_to_income": dti,
            "credit_history_years": hist, "num_credit_lines": lines, "missed_payments_2y": missed,
            "credit_utilization": util, "num_inquiries_6m": inq, "home_ownership": home,
            "loan_purpose": purpose,
        }])[cr_m["features"]]
        pd_rf = float(cr_m["pipeline"].predict_proba(row)[0, 1])
        pd_lr = float(sc_m["pipeline"].predict_proba(row)[0, 1])
        score = float(tu.pd_to_score(pd_lr))
        band = str(tu.risk_band(score).iloc[0])
        decision = "Approve" if score >= 650 else "Manual review" if score >= 550 else "Reject"
        a, b, c = st.columns(3)
        a.metric("Credit score", f"{score:.0f}")
        b.metric("Risk band", band)
        c.metric("Decision", decision)
        st.caption(f"Scorecard default probability {pd_lr:.1%} | Random Forest estimate {pd_rf:.1%}")
        st.write("**Reason codes (factors raising risk):**")
        for r in explain(cr_m["pipeline"], row, 4, "up") or ["Nothing significant"]:
            st.write("- " + r)


# ---------------------------------------------------------------- Tab 4: batch + drift
@st.cache_data(show_spinner=False)
def reference_data():
    df = pd.read_csv(tu.DATA_DIR / "upi_transactions.csv")
    X, y = df[upi_m["features"]], df["is_fraud"]
    Xtr, _, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=tu.RANDOM_STATE)
    return Xtr.sample(8000, random_state=1), df.head(200)


with tab4:
    feats = upi_m["features"]
    ref, sample = reference_data()
    st.write("Upload a CSV of transactions (same columns as `data/upi_transactions.csv`). Every row is scored, "
             "ranked by risk, and compared with the training data to detect **drift**.")
    st.download_button("Download a sample CSV", sample.to_csv(index=False), "sample_transactions.csv")
    up = st.file_uploader("Transactions CSV", type="csv")
    if up is not None:
        b = pd.read_csv(up)
        missing = [c for c in feats if c not in b.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            scores = upi_m["pipeline"].predict_proba(b[feats])[:, 1]
            out = b.copy()
            out["fraud_risk"] = scores.round(4)
            out["decision"] = tu.apply_tiers(scores, TIERS)
            out = out.sort_values("fraud_risk", ascending=False)
            cnt = out["decision"].value_counts().reindex(["Allow", "Verify", "Block"], fill_value=0)
            a, b2, c = st.columns(3)
            a.metric("Allow", f"{cnt['Allow']:,}")
            b2.metric("Verify", f"{cnt['Verify']:,}")
            c.metric("Block", f"{cnt['Block']:,}")
            st.dataframe(out.head(100), hide_index=True, width="stretch")
            st.download_button("Download scored file", out.to_csv(index=False), "scored_transactions.csv")

            st.subheader("Data drift vs training data (PSI)")
            ref_scores = upi_m["pipeline"].predict_proba(ref)[:, 1]
            sd = mon.score_drift(ref_scores, scores)
            (st.error if sd["status"] == "DRIFT" else st.warning if sd["status"] == "watch" else st.success)(
                f"Model score distribution: PSI {sd['psi']} ({sd['status']})")
            rep = mon.drift_report(ref, b, feats)
            st.dataframe(rep, hide_index=True, width="stretch")
            st.caption("PSI < 0.10 stable | 0.10 to 0.25 watch | above 0.25 drift: investigate or retrain.")

# ---------------------------------------------------------------- Tab 5: cost trade-off
@st.cache_data(show_spinner=False)
def test_scores():
    df = pd.read_csv(tu.DATA_DIR / "upi_transactions.csv")
    X, y = df[upi_m["features"]], df["is_fraud"]
    _, Xte, _, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=tu.RANDOM_STATE)
    return (yte.to_numpy(), upi_m["pipeline"].predict_proba(Xte)[:, 1],
            (df.loc[Xte.index, "fraud_type"] == "legit_emergency").to_numpy())


with tab5:
    y_t, s_t, emerg = test_scores()
    st.write("A missed fraud costs far more than a false alarm, but how much more is a **business decision**. "
             "Move the slider and watch the best single threshold change.")
    cfn = st.slider("Cost of one MISSED fraud (one false alarm = 1)", 1, 200, 10)
    ths = np.linspace(0.02, 0.98, 97)
    pred = s_t[:, None] >= ths[None, :]
    fn = ((~pred) & (y_t[:, None] == 1)).sum(0)
    fp = (pred & (y_t[:, None] == 0)).sum(0)
    cost = cfn * fn + fp
    best = int(np.argmin(cost))
    st.line_chart(pd.DataFrame({"total cost": cost}, index=np.round(ths, 2)), x_label="threshold", y_label="cost")
    a, b2, c, d = st.columns(4)
    a.metric("Best threshold", f"{ths[best]:.2f}")
    b2.metric("Frauds caught", f"{int(y_t.sum() - fn[best])} of {int(y_t.sum())}")
    c.metric("Missed frauds", f"{int(fn[best])}")
    d.metric("False alarms", f"{int(fp[best])}")
    st.caption("A higher cost of a miss pushes the best threshold down (catch more fraud, accept more false alarms). "
               "On this synthetic data fraud scores separate cleanly from legitimate ones, so the best threshold "
               "barely moves; on real, overlapping data it moves a lot. The operating points below show the "
               "trade-off directly.")
    pts = []
    for t in (0.05, 0.10, 0.20, 0.30, 0.45, 0.60):
        flagged = s_t >= t
        pts.append({"threshold": t, "frauds caught": int((flagged & (y_t == 1)).sum()),
                    "frauds missed": int((~flagged & (y_t == 1)).sum()),
                    "legit customers flagged": int((flagged & (y_t == 0)).sum()),
                    "of which genuine emergencies": int((flagged & emerg).sum()),
                    "share of all traffic flagged %": round(100 * flagged.mean(), 2)})
    st.write("**Operating points** (what each cut-off would do on the held-out test set)")
    st.dataframe(pd.DataFrame(pts), hide_index=True, width="stretch")

# ---------------------------------------------------------------- Tab 6: mule accounts
with tab6:
    sc_path = tu.REPORTS_DIR / "mule_scored_accounts.csv"
    if not sc_path.exists():
        st.info("Run `python src/graph_module.py` first to build the account network and the detector.")
    else:
        sc = pd.read_csv(sc_path, index_col=0)
        st.write("Single payments look harmless; the **pattern across accounts** gives mules away: money "
                 "arrives from several people and is passed on within minutes, down a chain or round a cycle. "
                 "Scores below are **cross-validated** (each account scored by a model that never saw it).")
        m_path = tu.REPORTS_DIR / "mule_metrics.csv"
        if m_path.exists():
            st.dataframe(pd.read_csv(m_path, index_col=0), width="stretch")
        thr = st.slider("Flag accounts with mule risk above", 0.1, 0.95, 0.5, 0.05)
        fl = sc[sc["mule_risk_oof"] >= thr].sort_values("mule_risk_oof", ascending=False)
        a, b2, c = st.columns(3)
        a.metric("Accounts", f"{len(sc):,}")
        b2.metric("Flagged", f"{len(fl):,}")
        c.metric("Real mules among flagged", f"{int(fl['is_mule'].sum())} of {int(sc['is_mule'].sum())}")
        show = ["mule_risk_oof", "fast_fwd_count", "fast_fwd_share", "passthrough", "median_dwell_min",
                "ff_component_size", "in_fast_cycle", "role", "is_mule"]
        st.dataframe(fl[show].round(3).head(50), width="stretch")
        st.caption("`role` shows what each account really is in the synthetic data: flagged accounts that are "
                   "'payroll' or 'family_forwarder' are the false alarms (legitimate look-alikes).")
        img = tu.REPORTS_DIR / "mule_network.png"
        if img.exists():
            st.image(str(img), caption="Largest detected ring")
