"""Retrain the UPI fraud model using analyst feedback.

The Streamlit alert queue writes reports/feedback_log.csv  (txn_id, verdict) where verdict is
'confirmed_fraud' or 'genuine_customer'. This script:

  1. adds those analyst-labelled transactions to the TRAINING data (weighted, since a human checked them),
  2. retrains, re-chooses the Allow / Verify / Block thresholds on out-of-fold predictions,
  3. evaluates old vs new on the held-out test rows that are NOT in the feedback set (no leakage),
  4. replaces models/upi_fraud_rf.joblib only if the new model is not worse (the old one is kept as *_prev).

Usage:  python src/retrain.py            # retrain and promote if not worse
        python src/retrain.py --dry-run  # compare only, change nothing
"""
import argparse
import shutil

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

import train_utils as tu

FEEDBACK = tu.REPORTS_DIR / "feedback_log.csv"
LABEL = {"confirmed_fraud": 1, "genuine_customer": 0}


def load_feedback(path=FEEDBACK):
    if not path.exists():
        return pd.DataFrame(columns=["txn_id", "label"])
    fb = pd.read_csv(path)
    fb = fb[fb["verdict"].isin(LABEL)].drop_duplicates("txn_id", keep="last")   # latest verdict wins
    return pd.DataFrame({"txn_id": fb["txn_id"], "label": fb["verdict"].map(LABEL)})


def retrain(feedback=None, weight=5.0, dry_run=False, promote_margin=0.005, verbose=True):
    old = tu.load_model("upi_fraud_rf")
    feats = old["features"]
    fb = load_feedback() if feedback is None else feedback
    if fb.empty:
        return {"status": "no feedback yet", "n_feedback": 0}

    df = pd.read_csv(tu.DATA_DIR / "upi_transactions.csv")
    X, y = df[feats], df["is_fraud"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=tu.RANDOM_STATE)

    fb = fb[fb["txn_id"].isin(df["txn_id"])].copy()
    fb_idx = df.index[df["txn_id"].isin(fb["txn_id"])]
    label_map = dict(zip(fb["txn_id"], fb["label"]))
    fb_y = df.loc[fb_idx, "txn_id"].map(label_map).astype(int)

    # training set = original training rows + analyst-labelled rows (analyst label overrides)
    Xtr2 = pd.concat([Xtr.drop(index=Xtr.index.intersection(fb_idx)), X.loc[fb_idx]])
    ytr2 = pd.concat([ytr.drop(index=ytr.index.intersection(fb_idx)), fb_y])
    w = np.where(Xtr2.index.isin(fb_idx), weight, 1.0)

    # fair comparison: the evaluation half of the held-out data (analysts never see it, we never train on it)
    _, eval_idx = tu.live_eval_split(Xte.index)
    Xev, yev = Xte.loc[eval_idx], yte.loc[eval_idx]
    fb_idx = fb_idx.difference(eval_idx)                     # safety: never train on evaluation rows
    fb = fb[fb["txn_id"].isin(df.loc[fb_idx, "txn_id"])]
    Xtr2 = pd.concat([Xtr.drop(index=Xtr.index.intersection(fb_idx)), X.loc[fb_idx]])
    ytr2 = pd.concat([ytr.drop(index=ytr.index.intersection(fb_idx)),
                      df.loc[fb_idx, "txn_id"].map(label_map).astype(int)])
    w = np.where(Xtr2.index.isin(fb_idx), weight, 1.0)

    pre, _ = tu.make_preprocessor(df, "is_fraud", ["txn_id", "fraud_type"])
    new_pipe = tu.build_pipeline(clone(pre), tu.get_models("class_weight")["Random Forest"])
    new_pipe.fit(Xtr2, ytr2, clf__sample_weight=w)

    m_old = tu.holdout_metrics(yev, old["pipeline"].predict_proba(Xev)[:, 1])
    m_new = tu.holdout_metrics(yev, new_pipe.predict_proba(Xev)[:, 1])
    result = {"status": "compared", "n_feedback": len(fb_idx),
              "old_PR_AUC": round(m_old["PR-AUC"], 4), "new_PR_AUC": round(m_new["PR-AUC"], 4),
              "old_F1": round(m_old["F1"], 4), "new_F1": round(m_new["F1"], 4)}
    ok = m_new["PR-AUC"] >= m_old["PR-AUC"] - promote_margin
    result["promote"] = bool(ok)

    if ok and not dry_run:
        oof = tu.oof_scores(tu.build_pipeline(clone(pre), tu.get_models("class_weight")["Random Forest"]),
                            Xtr2, ytr2)
        tiers = tu.choose_tiers(ytr2, oof)
        src = tu.MODELS_DIR / "upi_fraud_rf.joblib"
        shutil.copy(src, tu.MODELS_DIR / "upi_fraud_rf_prev.joblib")
        tu.save_model({"pipeline": new_pipe, "threshold": tiers["verify"], "tiers": tiers, "features": feats},
                      "upi_fraud_rf")
        result["status"] = "promoted (old model kept as upi_fraud_rf_prev.joblib)"
        result["new_tiers"] = {k: round(v, 3) for k, v in tiers.items()}
    elif not ok:
        result["status"] = "NOT promoted: new model is worse"
    if verbose:
        print(result)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    retrain(dry_run=ap.parse_args().dry_run)
