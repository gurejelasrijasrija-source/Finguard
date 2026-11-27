"""Fairness audit for the loan model.

The model never sees `gender`, but it can still treat groups differently through proxies, so we audit
outcomes by group. All numbers are computed on the held-out test set.

Terms (positive = predicted to default = loan rejected / sent to review):
  approval rate          share NOT flagged
  disparate impact       group approval rate / reference-group approval rate  (the "4/5 rule": >= 0.80 is OK)
  TPR (equal opportunity) share of real defaulters correctly flagged
  FPR                    share of good borrowers wrongly flagged (the harm to a genuine customer)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score


def age_group(age):
    return pd.cut(pd.Series(age), bins=[0, 29, 44, 59, 200], labels=["<30", "30-44", "45-59", "60+"]).astype(str)


def group_report(y_true, flagged, groups, reference=None):
    d = pd.DataFrame({"y": np.asarray(y_true), "flag": np.asarray(flagged).astype(int), "g": np.asarray(groups)})
    rows = []
    for g, x in d.groupby("g"):
        pos, neg = x[x.y == 1], x[x.y == 0]
        rows.append({"group": g, "n": len(x), "actual_default_rate": x.y.mean(),
                     "approval_rate": 1 - x.flag.mean(),
                     "TPR": pos.flag.mean() if len(pos) else np.nan,
                     "FPR": neg.flag.mean() if len(neg) else np.nan})
    r = pd.DataFrame(rows).set_index("group")
    ref = reference if reference in r.index else r["approval_rate"].idxmax()
    r["disparate_impact"] = r["approval_rate"] / r.loc[ref, "approval_rate"]
    r["TPR_gap_vs_ref"] = r["TPR"] - r.loc[ref, "TPR"]
    r["FPR_gap_vs_ref"] = r["FPR"] - r.loc[ref, "FPR"]
    r["passes_4/5_rule"] = r["disparate_impact"] >= 0.80
    return r.round(3), ref


def proxy_auc(X_numeric, sensitive, seed=42):
    """Can the model's inputs predict the sensitive attribute? AUC ~ 0.5 means no proxy leak;
    well above 0.5 means other features are standing in for it."""
    y = pd.factorize(pd.Series(np.asarray(sensitive)))[0]
    rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    p = cross_val_predict(rf, X_numeric, y, cv=3, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))
