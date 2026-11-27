"""Mule-account detection from the transaction NETWORK.

Fraud money rarely stays where it lands. Scam victims pay a "collector" account, which forwards almost
all of it within minutes down a chain of mule accounts (or round and round a cycle) before cash-out.
A single transaction looks harmless, but the *pattern across accounts* is not. This module:

  1. generates a synthetic account network with mule chains and mule cycles, plus realistic look-alikes
     that are LEGITIMATE (merchants, payroll accounts, family members who forward money),
  2. builds graph features per account (pass-through ratio, dwell time, fast-forward links, cycles,
     size of the connected fast-forward component),
  3. trains a classifier and compares it with a simple rule, and
  4. draws one detected ring.

Run:  python src/graph_module.py      (writes data/account_edges.csv, data/accounts.csv,
                                       models/mule_rf.joblib, reports/mule_*.png/csv)

NOTE: the network is SYNTHETIC. Real bank graphs are far larger and noisier.
"""
from pathlib import Path

import joblib
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, MODELS_DIR, REPORTS_DIR = ROOT / "data", ROOT / "models", ROOT / "reports"

# Behavioural features only. Raw money totals (total_in / total_out) are deliberately NOT used: in this
# synthetic network mules happen to move larger sums, which would be a shortcut rather than a real signal.
FEATURES = ["account_age_days", "n_in", "n_out", "in_deg", "out_deg", "passthrough",
            "median_dwell_min", "fast_fwd_count", "fast_fwd_share", "ff_component_size", "in_fast_cycle"]


# --------------------------------------------------------------------------
# 1. Synthetic network
# --------------------------------------------------------------------------
def generate_network(n_accounts=3000, n_chains=18, n_cycles=6, seed=42):
    rng = np.random.default_rng(seed)
    ids = np.array([f"ACC{i:05d}" for i in range(n_accounts)])
    acc = pd.DataFrame({"account": ids,
                        "account_age_days": rng.gamma(3.0, 250.0, n_accounts).astype(int) + 10,
                        "is_mule": 0, "role": "normal"})
    role = acc["role"].to_numpy(dtype=object)
    perm = rng.permutation(n_accounts)
    merchants = perm[: int(0.06 * n_accounts)]
    payroll = perm[int(0.06 * n_accounts): int(0.07 * n_accounts)]
    family = perm[int(0.07 * n_accounts): int(0.09 * n_accounts)]
    role[merchants], role[payroll], role[family] = "merchant", "payroll", "family_forwarder"
    normal = np.array([i for i in range(n_accounts) if role[i] in ("normal", "family_forwarder")])

    start = pd.Timestamp("2026-09-01")
    edges = []

    def add(s, d, amt, minutes):
        edges.append((ids[s], ids[d], float(round(amt, 2)), start + pd.Timedelta(minutes=float(minutes))))

    span = 30 * 24 * 60
    # ---- ordinary activity: people pay merchants (60%) and each other (40%)
    for i in normal:
        for _ in range(rng.poisson(8)):
            d = rng.choice(merchants) if rng.random() < 0.6 else rng.choice(normal)
            if d != i:
                add(i, d, rng.lognormal(6.3, 1.0), rng.uniform(0, span))
    # ---- legitimate look-alike 1: payroll accounts receive one big sum, pay many people over hours/days
    for p in payroll:
        t0 = rng.uniform(0, span - 4000)
        add(rng.choice(normal), p, rng.uniform(300000, 900000), t0)
        for e in rng.choice(normal, size=rng.integers(10, 30), replace=False):
            add(p, e, rng.uniform(15000, 40000), t0 + rng.uniform(180, 2880))
    # ---- legitimate look-alike 2: family members forward money; 30% do it quickly (hard cases)
    for f in family:
        for _ in range(rng.integers(2, 5)):
            t0 = rng.uniform(0, span - 600)
            amt = rng.uniform(2000, 20000)
            add(rng.choice(normal), f, amt, t0)
            fast = rng.random() < 0.3
            add(f, rng.choice(normal), amt * rng.uniform(0.85, 0.97),
                t0 + (rng.uniform(10, 50) if fast else rng.uniform(70, 400)))

    # ---- mule chains and cycles (mules are drawn from ordinary accounts, so they also behave normally)
    pool = [i for i in normal if role[i] == "normal"]
    pool = list(rng.permutation(pool))
    mule_ids = set()

    def take(k, mule=True):
        got = [pool.pop() for _ in range(k)]
        if mule:                      # victims are ordinary customers, NOT mules
            mule_ids.update(got)
        return got

    for _ in range(n_chains):
        chain = take(int(rng.integers(2, 5)))
        victims = take(int(rng.integers(6, 20)), mule=False)
        exit_acct = rng.choice(merchants)
        slow = rng.random() < 0.2                      # 20% of chains are slower and leakier (harder)
        day0 = rng.uniform(0, span - 4320)
        for v in victims:
            amt = rng.lognormal(9.0, 0.7)
            t = day0 + rng.uniform(0, 3 * 1440)
            add(v, chain[0], amt, t)
            for hop in range(len(chain)):
                amt *= rng.uniform(0.75, 0.90) if slow else rng.uniform(0.92, 0.99)
                t += rng.uniform(60, 240) if slow else rng.uniform(1, 25)
                nxt = chain[hop + 1] if hop + 1 < len(chain) else exit_acct
                add(chain[hop], nxt, amt, t)
    for _ in range(n_cycles):
        ring = take(int(rng.integers(3, 5)))
        victims = take(int(rng.integers(1, 4)), mule=False)
        exit_acct = rng.choice(merchants)
        t = rng.uniform(0, span - 3000)
        amt = rng.uniform(30000, 120000)
        for v in victims:
            add(v, ring[0], amt / len(victims), t)
        for _round in range(int(rng.integers(2, 4))):
            for j in range(len(ring)):
                amt *= rng.uniform(0.95, 0.99)
                t += rng.uniform(3, 20)
                add(ring[j], ring[(j + 1) % len(ring)], amt, t)
        add(ring[-1], exit_acct, amt * 0.97, t + rng.uniform(3, 20))

    acc.loc[list(mule_ids), "is_mule"] = 1
    acc.loc[list(mule_ids), "role"] = "mule"
    acc["role"] = [("mule" if i in mule_ids else role[i]) for i in range(n_accounts)]
    ed = pd.DataFrame(edges, columns=["src", "dst", "amount", "ts"]).sort_values("ts").reset_index(drop=True)
    return ed, acc


# --------------------------------------------------------------------------
# 2. Graph features
# --------------------------------------------------------------------------
def _forward_links(edges):
    """For each outgoing payment, find the latest payment INTO the same account before it.
    A 'fast forward' = at least 70% of that money leaves within 60 minutes."""
    e = edges.copy()
    e["ts"] = pd.to_datetime(e["ts"])
    inc = e.rename(columns={"dst": "account", "src": "from_acct", "amount": "amt_in", "ts": "t_in"})[
        ["account", "from_acct", "amt_in", "t_in"]].sort_values("t_in")
    out = e.rename(columns={"src": "account", "dst": "to_acct", "amount": "amt_out", "ts": "t_out"})[
        ["account", "to_acct", "amt_out", "t_out"]].sort_values("t_out")
    m = pd.merge_asof(out, inc, left_on="t_out", right_on="t_in", by="account", direction="backward")
    m["dwell_min"] = (m["t_out"] - m["t_in"]).dt.total_seconds() / 60
    m["fast_fwd"] = ((m["dwell_min"] <= 60) & (m["amt_out"] >= 0.7 * m["amt_in"])
                     & (m["amt_out"] <= m["amt_in"] * 1.001))
    return m


def graph_features(edges, accounts):
    """One row per account with the features in FEATURES (plus is_mule if present in `accounts`)."""
    e = edges.copy()
    e["ts"] = pd.to_datetime(e["ts"])
    m = _forward_links(e)

    out_g = e.groupby("src").agg(n_out=("amount", "size"), out_deg=("dst", "nunique"), total_out=("amount", "sum"))
    in_g = e.groupby("dst").agg(n_in=("amount", "size"), in_deg=("src", "nunique"), total_in=("amount", "sum"))
    f = accounts.set_index("account").join(in_g).join(out_g)
    for c in ["n_in", "n_out", "in_deg", "out_deg", "total_in", "total_out"]:
        f[c] = f[c].fillna(0)
    f["passthrough"] = (f["total_out"] / (f["total_in"] + 1)).clip(0, 2)

    f["median_dwell_min"] = m.groupby("account")["dwell_min"].median().reindex(f.index).fillna(10080)
    ff = m[m["fast_fwd"]]
    f["fast_fwd_count"] = ff.groupby("account").size().reindex(f.index).fillna(0)
    f["fast_fwd_share"] = f["fast_fwd_count"] / f["n_out"].clip(lower=1)

    # graph of "money was quickly passed on": from_acct -> account -> to_acct
    g = nx.DiGraph()
    g.add_edges_from(zip(ff["from_acct"], ff["account"]))
    g.add_edges_from(zip(ff["account"], ff["to_acct"]))
    comp = {}
    for c in nx.weakly_connected_components(g):
        for n in c:
            comp[n] = len(c)
    f["ff_component_size"] = pd.Series(comp).reindex(f.index).fillna(1)
    cyc_nodes = set()
    for cyc in nx.simple_cycles(g, length_bound=5):
        cyc_nodes.update(cyc)
    f["in_fast_cycle"] = f.index.isin(cyc_nodes).astype(int)
    return f


# --------------------------------------------------------------------------
# 3. Detection
# --------------------------------------------------------------------------
def rule_baseline(f):
    """Simple hand-written rule: forwards money fast, often, and nearly all of it."""
    return ((f["fast_fwd_count"] >= 3) & (f["passthrough"] >= 0.8)).astype(int).to_numpy()


def _metrics(y, pred, score=None):
    d = {"Precision": precision_score(y, pred, zero_division=0), "Recall": recall_score(y, pred, zero_division=0),
         "F1": f1_score(y, pred, zero_division=0)}
    if score is not None:
        d.update({"PR-AUC": average_precision_score(y, score), "ROC-AUC": roc_auc_score(y, score)})
    return d


def train_detector(f):
    X, y = f[FEATURES], f["is_mule"]
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                n_jobs=-1, random_state=42)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = cross_val_predict(rf, X, y, cv=skf, method="predict_proba")[:, 1]
    rows = {"Rule (>=3 fast forwards, >=80% passed on)": _metrics(y, rule_baseline(f)),
            "Random Forest on graph features (5-fold CV)": _metrics(y, (oof >= 0.5).astype(int), oof)}
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return rf, pd.DataFrame(rows).T, imp, oof


def score_accounts(f, model):
    out = f.copy()
    out["mule_risk"] = model.predict_proba(f[FEATURES])[:, 1]
    return out.sort_values("mule_risk", ascending=False)


# --------------------------------------------------------------------------
# 4. Plot one detected ring
# --------------------------------------------------------------------------
def plot_ring(edges, scored, fname="mule_network.png", min_risk=0.5):
    """Draw the biggest connected group of high-risk accounts with the victims who paid into it."""
    e = edges.copy()
    risky = set(scored.index[scored["mule_risk"] >= min_risk])
    g = nx.Graph()
    g.add_edges_from(zip(e.loc[e["src"].isin(risky) & e["dst"].isin(risky), "src"],
                         e.loc[e["src"].isin(risky) & e["dst"].isin(risky), "dst"]))
    if not len(g):
        return None
    ring = max(nx.connected_components(g), key=len)
    sub = e[e["src"].isin(ring) | e["dst"].isin(ring)]
    sub = sub[sub["src"].isin(ring) | sub["dst"].isin(ring)]
    G = nx.DiGraph()
    for s, d, a in zip(sub["src"], sub["dst"], sub["amount"]):
        if G.has_edge(s, d):
            G[s][d]["amount"] += a
        else:
            G.add_edge(s, d, amount=a)
    colors = ["#d62728" if n in ring else ("#1f77b4" if scored.loc[n, "is_mule"] == 0 else "#ff7f0e")
              for n in G.nodes if n in scored.index]
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=3, k=0.6)
    nx.draw_networkx(G, pos, ax=ax, node_color=colors, node_size=140, with_labels=False,
                     edge_color="#999999", arrows=True, arrowsize=8, width=0.8)
    ax.set_title("Detected mule ring (red = flagged mules, blue = paying victims / other accounts)")
    ax.axis("off")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return len(ring)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    edges, accounts = generate_network()
    edges.to_csv(DATA_DIR / "account_edges.csv", index=False)
    accounts.to_csv(DATA_DIR / "accounts.csv", index=False)
    f = graph_features(edges, accounts)
    rf, table, imp, oof = train_detector(f)
    joblib.dump({"model": rf, "features": FEATURES}, MODELS_DIR / "mule_rf.joblib")
    table.round(4).to_csv(REPORTS_DIR / "mule_metrics.csv")
    imp.round(4).to_csv(REPORTS_DIR / "mule_feature_importance.csv", header=["importance"])
    scored = score_accounts(f, rf)
    scored.assign(mule_risk_oof=pd.Series(oof, index=f.index)).to_csv(REPORTS_DIR / "mule_scored_accounts.csv")
    ring_n = plot_ring(edges, scored.assign(mule_risk=pd.Series(oof, index=f.index).reindex(scored.index)))
    print(f"accounts={len(accounts)}  transfers={len(edges)}  mules={int(accounts.is_mule.sum())}  "
          f"largest flagged ring={ring_n}")
    print(table.round(3).to_string())
    print(imp.round(3).head(6).to_string())


if __name__ == "__main__":
    main()
