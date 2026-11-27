import graph_module as gm


def test_network_labels_mules_not_victims():
    edges, acc = gm.generate_network(n_accounts=800, n_chains=5, n_cycles=2, seed=3)
    n_mules = int(acc["is_mule"].sum())
    assert 5 * 2 <= n_mules <= 5 * 4 + 2 * 4               # only the chain / ring accounts, not victims
    assert set(["src", "dst", "amount", "ts"]) <= set(edges.columns)


def test_mules_pass_money_on_fast():
    edges, acc = gm.generate_network(n_accounts=800, n_chains=5, n_cycles=2, seed=3)
    f = gm.graph_features(edges, acc)
    assert set(gm.FEATURES) <= set(f.columns)
    assert f.loc[f["is_mule"] == 1, "fast_fwd_count"].mean() > 3 * f.loc[f["is_mule"] == 0, "fast_fwd_count"].mean()
    assert f.loc[f["is_mule"] == 1, "median_dwell_min"].median() < f.loc[f["is_mule"] == 0, "median_dwell_min"].median()


def test_detector_beats_chance():
    edges, acc = gm.generate_network(n_accounts=1500, n_chains=10, n_cycles=3, seed=4)
    f = gm.graph_features(edges, acc)
    _, table, _, _ = gm.train_detector(f)
    assert table.loc["Random Forest on graph features (5-fold CV)", "ROC-AUC"] > 0.9
