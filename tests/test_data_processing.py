import pytest
import pandas as pd
import numpy as np
from src.data_processing import (
    DateFeatureExtractor,
    CustomerAggregator,
    calculate_rfm,
    scale_rfm,
    cluster_customers,
    identify_high_risk_cluster,
    assign_high_risk_label,
    process_data_end_to_end
)
from src.config import config

def test_date_feature_extractor(sample_transaction_data):
    extractor = DateFeatureExtractor()
    df = extractor.transform(sample_transaction_data)
    
    assert "TransactionHour" in df.columns
    assert "TransactionDay" in df.columns
    assert "TransactionMonth" in df.columns
    assert "TransactionYear" in df.columns
    assert df["TransactionYear"].iloc[0] == 2018

def test_customer_aggregator(sample_transaction_data):
    aggregator = CustomerAggregator()
    df = aggregator.transform(sample_transaction_data)
    
    assert "TotalTransactionAmount" in df.columns
    assert "StdTransactionAmount" in df.columns
    assert len(df) == 3  # C1, C2, C3
    assert df.loc[df[config.data.customer_id_col] == "C1", "TotalTransactionAmount"].values[0] == 350.0

def test_customer_aggregator_single_transaction():
    # Test that Std Dev is 0 for a single transaction instead of NaN
    data = pd.DataFrame({
        config.data.customer_id_col: ["C4"],
        "Amount": [100.0],
        "TransactionId": ["T4"],
        config.data.datetime_col: ["2018-12-01"]
    })
    aggregator = CustomerAggregator()
    df = aggregator.transform(data)
    assert df.loc[df[config.data.customer_id_col] == "C4", "StdTransactionAmount"].values[0] == 0.0

def test_calculate_rfm(sample_transaction_data, snapshot_date):
    rfm = calculate_rfm(sample_transaction_data, snapshot_date)
    
    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary" in rfm.columns
    # C3's last trans was 2018-12-30 18:00:00. Snapshot 2019-01-01 00:00:00. 
    # Difference is 1 day 6 hours. .days returns 1.
    assert rfm.loc[rfm[config.data.customer_id_col] == "C3", "Recency"].values[0] == 1

def test_scaling_and_clustering():
    rfm = pd.DataFrame({
        config.data.customer_id_col: ["C1", "C2", "C3"],
        "Recency": [1, 10, 50],
        "Frequency": [10, 5, 1],
        "Monetary": [1000, 500, 50]
    })
    
    scaled = scale_rfm(rfm)
    assert scaled.shape == (3, 3)
    
    clusters = cluster_customers(scaled, n_clusters=2)
    assert len(clusters) == 3
    assert len(np.unique(clusters)) == 2

def test_identify_high_risk_cluster():
    # Create a deterministic RFM where one cluster is clearly high risk
    # High Risk = High Recency, Low Frequency, Low Monetary
    rfm = pd.DataFrame({
        "cluster": [0, 1],
        "Recency": [2, 50],
        "Frequency": [20, 1],
        "Monetary": [2000, 50]
    })
    
    high_risk = identify_high_risk_cluster(rfm)
    assert high_risk == 1

def test_assign_high_risk_label():
    rfm = pd.DataFrame({
        config.data.customer_id_col: ["C1", "C2"],
        "cluster": [0, 1]
    })
    labeled = assign_high_risk_label(rfm, high_risk_cluster=1)
    
    assert labeled.loc[labeled[config.data.customer_id_col] == "C2", config.data.target_col].values[0] == 1
    assert labeled.loc[labeled[config.data.customer_id_col] == "C1", config.data.target_col].values[0] == 0

def test_process_data_end_to_end(sample_transaction_data, snapshot_date):
    df = process_data_end_to_end(sample_transaction_data, snapshot_date)
    
    assert config.data.target_col in df.columns
    assert "TotalTransactionAmount" in df.columns
    assert config.data.customer_id_col in df.columns
    assert len(df) == 3
    assert set(df[config.data.target_col].unique()).issubset({0, 1})
