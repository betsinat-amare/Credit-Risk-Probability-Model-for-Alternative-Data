from dataclasses import dataclass, field
from typing import List
import os

@dataclass(frozen=True)
class DataConfig:
    raw_data_path: str = "data/raw/data.csv"
    processed_data_path: str = "data/processed/final_df.csv"
    snapshot_date: str = "2019-01-01"
    customer_id_col: str = "CustomerId"
    datetime_col: str = "TransactionStartTime"
    target_col: str = "is_high_risk"
    numerical_features: List[str] = field(default_factory=lambda: [
        "TotalTransactionAmount", 
        "AvgTransactionAmount", 
        "TransactionCount", 
        "StdTransactionAmount"
    ])
    categorical_features: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class ModelConfig:
    test_size: float = 0.2
    random_state: int = 42
    n_clusters: int = 3
    lr_max_iter: int = 1000
    rf_n_estimators: int = 100
    model_name: str = os.getenv("MODEL_NAME", "credit-risk-model")
    model_stage: str = os.getenv("MODEL_STAGE", "Production")

@dataclass(frozen=True)
class UIConfig:
    risk_threshold_low: float = 0.3
    risk_threshold_high: float = 0.7
    app_title: str = "Credit Risk Intelligence Dashboard"

@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ui: UIConfig = field(default_factory=UIConfig)

# Default global config instance
config = AppConfig()
