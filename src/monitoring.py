"""Drift monitoring: has the live data stopped looking like the data the model was trained on?

Uses the Population Stability Index (PSI), the standard measure in banking model monitoring:
    PSI < 0.10        stable
    0.10 - 0.25       watch (moderate shift)
    > 0.25            drift (investigate / retrain)
"""
import numpy as np
import pandas as pd


def psi_numeric(ref, new, bins=10):
    ref, new = pd.Series(ref).dropna(), pd.Series(new).dropna()
    if ref.nunique() <= 1 or len(new) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                               # (almost) constant / binary column
        edges = np.unique(np.concatenate([[ref.min() - 1e-9], edges, [ref.max() + 1e-9]]))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    n = np.histogram(new, edges)[0] / len(new)
    r, n = np.clip(r, 1e-4, None), np.clip(n, 1e-4, None)
    return float(np.sum((n - r) * np.log(n / r)))


def psi_categorical(ref, new):
    ref, new = pd.Series(ref).dropna().astype(str), pd.Series(new).dropna().astype(str)
    cats = sorted(set(ref) | set(new))
    r = ref.value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=1e-4).to_numpy()
    n = new.value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=1e-4).to_numpy()
    return float(np.sum((n - r) * np.log(n / r)))


def status(psi):
    return "stable" if psi < 0.10 else "watch" if psi < 0.25 else "DRIFT"


def drift_report(reference, new, features=None):
    """PSI per feature, sorted worst first. `reference` = training data, `new` = recent live data."""
    features = features or [c for c in reference.columns if c in new.columns]
    rows = []
    for c in features:
        numeric = pd.api.types.is_numeric_dtype(reference[c])
        p = psi_numeric(reference[c], new[c]) if numeric else psi_categorical(reference[c], new[c])
        rows.append({"feature": c, "psi": round(p, 4), "status": status(p)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def score_drift(ref_scores, new_scores):
    """PSI on the model's own output: a sudden change in the share of high scores is an early warning."""
    p = psi_numeric(ref_scores, new_scores)
    return {"psi": round(p, 4), "status": status(p)}
