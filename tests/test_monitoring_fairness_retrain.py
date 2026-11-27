import numpy as np
import pandas as pd
import fairness as fa
import monitoring as mon
import retrain as rt


def test_psi_stable_vs_shifted():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 5000), "c": rng.choice(["a", "b"], 5000)})
    same = pd.DataFrame({"x": rng.normal(0, 1, 2000), "c": rng.choice(["a", "b"], 2000)})
    shifted = pd.DataFrame({"x": rng.normal(2, 1, 2000), "c": rng.choice(["a", "b"], 2000, p=[.9, .1])})
    assert (mon.drift_report(ref, same)["status"] == "stable").all()
    assert (mon.drift_report(ref, shifted)["status"] == "DRIFT").all()


def test_fairness_equal_groups_pass():
    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.1).astype(int)
    flag = (rng.random(4000) < 0.12).astype(int)
    g = rng.choice(["F", "M"], 4000)
    r, _ = fa.group_report(y, flag, g)
    assert r["passes_4/5_rule"].all() and (r["disparate_impact"] > 0.9).all()


def test_fairness_detects_unequal_treatment():
    y = np.zeros(2000, dtype=int)
    g = np.array(["A"] * 1000 + ["B"] * 1000)
    flag = np.array([0] * 1000 + [1] * 500 + [0] * 500)    # group B flagged 50% vs 0%
    r, ref = fa.group_report(y, flag, g)
    assert ref == "A" and not r.loc["B", "passes_4/5_rule"]


def test_age_groups():
    assert list(fa.age_group([25, 35, 50, 70])) == ["<30", "30-44", "45-59", "60+"]


def test_feedback_log_parsing(tmp_path):
    p = tmp_path / "fb.csv"
    pd.DataFrame({"time": ["t1", "t2", "t3"], "txn_id": ["T1", "T1", "T2"],
                  "verdict": ["confirmed_fraud", "genuine_customer", "confirmed_fraud"]}).to_csv(p, index=False)
    fb = rt.load_feedback(p).set_index("txn_id")["label"]
    assert fb["T1"] == 0 and fb["T2"] == 1                # latest verdict wins
