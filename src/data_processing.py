import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Union, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from xverse.transformer import WOE
from src.config import config


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts temporal features from a datetime column.
    """

    def __init__(self, datetime_col: str = config.data.datetime_col):
        self.datetime_col = datetime_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DateFeatureExtractor":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col], errors="coerce")

        X["TransactionHour"] = X[self.datetime_col].dt.hour
        X["TransactionDay"] = X[self.datetime_col].dt.day
        X["TransactionMonth"] = X[self.datetime_col].dt.month
        X["TransactionYear"] = X[self.datetime_col].dt.year

        return X


class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates transaction data at the customer level.
    """

    def __init__(self, customer_id: str = config.data.customer_id_col):
        self.customer_id = customer_id

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CustomerAggregator":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        agg_df = (
            X.groupby(self.customer_id)
            .agg(
                TotalTransactionAmount=("Amount", "sum"),
                AvgTransactionAmount=("Amount", "mean"),
                TransactionCount=("TransactionId", "count"),
                StdTransactionAmount=("Amount", "std"),
            )
            .reset_index()
        )
        
        # Fill NaN for Std Dev if only one transaction
        agg_df["StdTransactionAmount"] = agg_df["StdTransactionAmount"].fillna(0.0)

        return agg_df


def build_feature_pipeline(
    categorical_features: List[str] = config.data.categorical_features, 
    numerical_features: List[str] = config.data.numerical_features
) -> ColumnTransformer:
    """
    Builds a scikit-learn preprocessing pipeline.
    """
    # Numerical pipeline
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features),
        ]
    )

    return preprocessor


def apply_woe(
    X: pd.DataFrame, y: pd.Series, categorical_features: List[str]
) -> Tuple[pd.DataFrame, WOE]:
    """
    Apply Weight of Evidence (WoE) transformation.
    """
    woe = WOE(
        cols=categorical_features,
        monotonic_trend="auto",
        min_bin_size=0.05,
        treat_missing="separate",
    )

    X_woe = woe.fit_transform(X, y)
    return X_woe, woe


def calculate_rfm(df: pd.DataFrame, snapshot_date: str = config.data.snapshot_date) -> pd.DataFrame:
    """
    Calculate Recency, Frequency, and Monetary metrics per customer.
    """
    df = df.copy()
    df[config.data.datetime_col] = pd.to_datetime(df[config.data.datetime_col])
    snapshot_date_dt = pd.to_datetime(snapshot_date)

    rfm = (
        df.groupby(config.data.customer_id_col)
        .agg(
            Recency=(config.data.datetime_col, lambda x: (snapshot_date_dt - x.max()).days),
            Frequency=("TransactionId", "count"),
            Monetary=("Amount", "sum"),
        )
        .reset_index()
    )

    return rfm


def scale_rfm(rfm: pd.DataFrame) -> np.ndarray:
    """
    Scales RFM metrics for clustering.
    """
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    return rfm_scaled


def cluster_customers(rfm_scaled: np.ndarray, n_clusters: int = config.model.n_clusters) -> np.ndarray:
    """
    Groups customers into clusters based on transaction behavior.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.model.random_state, n_init=10)
    clusters = kmeans.fit_predict(rfm_scaled)
    return clusters


def identify_high_risk_cluster(rfm: pd.DataFrame) -> int:
    """
    Identifies the cluster representing the highest credit risk (low engagement).
    """
    cluster_summary = (
        rfm.groupby("cluster")
        .agg({"Recency": "mean", "Frequency": "mean", "Monetary": "mean"})
        .reset_index()
    )

    # Sort by worst engagement: Low Frequency, Low Monetary, High Recency
    high_risk_cluster = (
        cluster_summary.sort_values(
            by=["Frequency", "Monetary", "Recency"], ascending=[True, True, False]
        )
        .iloc[0]["cluster"]
    )

    return int(high_risk_cluster)


def assign_high_risk_label(rfm: pd.DataFrame, high_risk_cluster: int) -> pd.DataFrame:
    """
    Creates a binary risk flag based on the identified high-risk cluster.
    """
    rfm[config.data.target_col] = np.where(rfm["cluster"] == high_risk_cluster, 1, 0)
    return rfm[[config.data.customer_id_col, config.data.target_col]]


def create_proxy_target(
    df: pd.DataFrame, snapshot_date: str = config.data.snapshot_date
) -> pd.DataFrame:
    """
    Full pipeline to create the proxy credit risk labels.
    """
    rfm = calculate_rfm(df, snapshot_date)
    rfm_scaled = scale_rfm(rfm)
    rfm["cluster"] = cluster_customers(rfm_scaled)
    high_risk_cluster = identify_high_risk_cluster(rfm)
    target = assign_high_risk_label(rfm, high_risk_cluster)

    return target


def process_data_end_to_end(
    df: pd.DataFrame, snapshot_date: str = config.data.snapshot_date
) -> pd.DataFrame:
    """
    Complete end-to-end processing pipeline as expected by training script.
    """
    # 1. Extract Date Features
    date_extractor = DateFeatureExtractor()
    df_with_dates = date_extractor.transform(df)

    # 2. Aggregate Features per Customer
    aggregator = CustomerAggregator()
    customer_features = aggregator.transform(df_with_dates)

    # 3. Create Proxy Labels
    proxy_target = create_proxy_target(df, snapshot_date)

    # 4. Merge Features and Target
    final_df = customer_features.merge(proxy_target, on=config.data.customer_id_col, how="inner")

    return final_df


