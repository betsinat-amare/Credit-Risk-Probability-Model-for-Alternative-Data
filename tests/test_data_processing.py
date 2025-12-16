def test_is_high_risk_binary(sample_raw_data):
    df = process_data_end_to_end(sample_raw_data)
    assert set(df["is_high_risk"].unique()).issubset({0, 1})
