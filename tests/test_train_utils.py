import numpy as np
import pandas as pd
import train_utils as tu
import generate_data as gd


def test_score_is_monotonic_and_bounded():
    p = np.array([0.01, 0.05, 0.2, 0.5, 0.9])
    s = tu.pd_to_score(p)
    assert (np.diff(s) < 0).all()                         # higher default risk -> lower score
    assert s.min() >= 300 and s.max() <= 900


def test_tiers_ordered_and_cover_everything():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.03).astype(int)
    s = np.clip(rng.normal(0.1 + 0.6 * y, 0.15), 0, 1)
    t = tu.choose_tiers(y, s)
    assert t["verify"] <= t["block"]
    tier = tu.apply_tiers(s, t)
    assert set(tier) <= {"Allow", "Verify", "Block"}
    assert (tier != "Allow").mean() <= 0.06               # review-capacity cap respected (5% + block share)


def test_tier_report_adds_up():
    tier = np.array(["Allow", "Allow", "Verify", "Block"])
    rep = tu.tier_report(tier, [0, 0, 1, 1], ["none", "legit_emergency", "none", "none"])
    assert rep["transactions"].sum() == 4
    assert rep.loc["Allow", "genuine_emergencies"] == 1


def test_time_split_is_chronological():
    df = gd.generate_upi(n=2000, seed=6)
    tr, te = tu.time_split(df)
    assert df.loc[tr, "txn_time"].max() <= df.loc[te, "txn_time"].min()
    assert len(set(tr) & set(te)) == 0


def test_identifiers_never_become_features():
    df = gd.generate_upi(n=1000, seed=7)
    pre, feats = tu.make_preprocessor(df, "is_fraud", [])
    for bad in ("txn_id", "txn_time", "fraud_type", "is_fraud"):
        assert bad not in feats


def test_live_eval_halves_are_disjoint():
    live, ev = tu.live_eval_split(pd.RangeIndex(100))
    assert len(set(live) & set(ev)) == 0 and len(live) + len(ev) == 100


def test_pipeline_trains_and_scores():
    df = gd.generate_upi(n=3000, seed=8)
    pre, feats = tu.make_preprocessor(df, "is_fraud", ["txn_id", "fraud_type"])
    pipe = tu.build_pipeline(pre, tu.get_models()["Random Forest"])
    pipe.fit(df[feats], df["is_fraud"])
    p = pipe.predict_proba(df[feats])[:, 1]
    assert p.shape == (3000,) and ((p >= 0) & (p <= 1)).all()
