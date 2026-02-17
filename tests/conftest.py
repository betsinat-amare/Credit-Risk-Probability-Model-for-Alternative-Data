import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_transaction_data():
    """
    Provides a small sample of transaction data for testing.
    """
    data = {
        "TransactionId": ["T1", "T2", "T3", "T4", "T5", "T6"],
        "CustomerId": ["C1", "C1", "C1", "C2", "C2", "C3"],
        "Amount": [100.0, 200.0, 50.0, 500.0, 10.0, 1000.0],
        "Value": [100.0, 200.0, 50.0, 500.0, 10.0, 1000.0],  # Added for compatibility with some legacy functions if any
        "TransactionStartTime": [
            "2018-12-01 10:00:00",
            "2018-12-10 11:00:00",
            "2018-12-20 12:00:00",
            "2018-11-01 09:00:00",
            "2018-11-15 10:00:00",
            "2018-12-30 18:00:00",
        ],
        "ProductCategory": ["A", "B", "A", "C", "A", "B"],
        "ChannelId": ["CH1", "CH2", "CH1", "CH3", "CH1", "CH2"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def snapshot_date():
    return "2019-01-01"
