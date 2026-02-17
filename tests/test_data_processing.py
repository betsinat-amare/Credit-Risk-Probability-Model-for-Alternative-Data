import pytest
import pandas as pd
import numpy as np
from src.data_processing import (
    DateFeatureExtractor,
    CustomerAggregator,
    calculate_rfm,
    process_data_end_to_end
)

def test_date_feature_extractor(sample_transaction_data):
    extractor = DateFeatureExtractor(datetime_col="TransactionStartTime")
    df = extractor.transform(sample_transaction_data)
    
    assert "TransactionHour" in df.columns
    assert "TransactionDay" in df.columns
    assert "TransactionMonth" in df.columns
    assert "TransactionYear" in df.columns
    assert df["TransactionYear"].iloc[0] == 2018

def test_customer_aggregator(sample_transaction_data):
    aggregator = CustomerAggregator(customer_id="CustomerId")
    df = aggregator.transform(sample_transaction_data)
    
    assert "TotalTransactionAmount" in df.columns
    assert len(df) == 3  # C1, C2, C3
    assert df.loc[df["CustomerId"] == "C1", "TotalTransactionAmount"].values[0] == 350.0

def test_calculate_rfm(sample_transaction_data, snapshot_date):
    rfm = calculate_rfm(sample_transaction_data, snapshot_date)
    
    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary" in rfm.columns
    # C3's last trans was 2018-12-30. Snapshot 2019-01-01. Recency should be 2.
    assert rfm.loc[rfm["CustomerId"] == "C3", "Recency"].values[0] == 2

def test_process_data_end_to_end(sample_transaction_data, snapshot_date):
    df = process_data_end_to_end(sample_transaction_data, snapshot_date)
    
    assert "is_high_risk" in df.columns
    assert "TotalTransactionAmount" in df.columns
    assert "CustomerId" in df.columns
    assert len(df) == 3
    assert set(df["is_high_risk"].unique()).issubset({0, 1})
