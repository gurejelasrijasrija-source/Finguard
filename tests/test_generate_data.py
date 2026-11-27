import generate_data as gd


def test_upi_shape_and_labels():
    df = gd.generate_upi(n=4000, seed=1)
    assert len(df) == 4000
    assert {"txn_id", "txn_time", "is_fraud", "fraud_type", "payee_category", "location_match"} <= set(df.columns)
    assert 0.005 < df["is_fraud"].mean() < 0.05          # rare-event problem
    assert df["txn_id"].is_unique


def test_emergencies_are_legitimate_and_context_separates_them():
    df = gd.generate_upi(n=20000, seed=2)
    em = df[df["fraud_type"] == "legit_emergency"]
    assert len(em) > 100
    assert em["is_fraud"].mean() < 0.02                    # labelled legitimate (apart from label noise)
    fr = df[df["fraud_type"].isin(["collect_scam", "fake_qr", "mule_passthru", "sim_swap"])]
    # context should differ: emergencies more often go to medical payees / usual location
    assert (em["payee_category"] == "medical").mean() > (fr["payee_category"] == "medical").mean()


def test_new_payee_consistent_with_prior_count():
    df = gd.generate_upi(n=3000, seed=3)
    assert ((df["payee_prior_txn_count"] == 0).astype(int) == df["is_new_payee"]).all()


def test_credit_default_rate_reasonable():
    df = gd.generate_credit(n=5000, seed=4)
    assert 0.03 < df["default"].mean() < 0.30
    assert df["applicant_id"].is_unique


def test_reproducible():
    a, b = gd.generate_upi(n=1000, seed=5), gd.generate_upi(n=1000, seed=5)
    assert a.equals(b)
