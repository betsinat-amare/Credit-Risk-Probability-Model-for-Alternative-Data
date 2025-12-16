import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

from xverse.transformer import WOE


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col="TransactionStartTime"):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col], errors="coerce")

        X["TransactionHour"] = X[self.datetime_col].dt.hour
        X["TransactionDay"] = X[self.datetime_col].dt.day
        X["TransactionMonth"] = X[self.datetime_col].dt.month
        X["TransactionYear"] = X[self.datetime_col].dt.year

        return X


class CustomerAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id="CustomerId"):
        self.customer_id = customer_id

    def fit(self, X, y=None):
        return self

    def transform(self, X):
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

        return agg_df


def build_feature_pipeline(categorical_features, numerical_features):
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


def apply_woe(X, y, categorical_features):
    """
    Apply Weight of Evidence (WoE) transformation
    """
    woe = WOE(
        cols=categorical_features,
        monotonic_trend="auto",
        min_bin_size=0.05,
        treat_missing="separate"
    )

    X_woe = woe.fit_transform(X, y)
    return X_woe, woe


def process_data(raw_df, target=None):
    """
    Full feature engineering pipeline
    """

    # Step 1: Date features
    date_extractor = DateFeatureExtractor()
    df = date_extractor.transform(raw_df)

    # Step 2: Aggregate per customer
    aggregator = CustomerAggregator()
    customer_df = aggregator.transform(df)

    # Define features
    categorical_features = []
    numerical_features = [
        "TotalTransactionAmount",
        "AvgTransactionAmount",
        "TransactionCount",
        "StdTransactionAmount",
    ]

    # Step 3: Preprocessing pipeline
    preprocessor = build_feature_pipeline(
        categorical_features, numerical_features
    )

    X_processed = preprocessor.fit_transform(customer_df)

    # Step 4: WoE (if target exists)
    if target is not None:
        X_woe, woe_model = apply_woe(
            customer_df[numerical_features],
            target,
            categorical_features
        )
        return X_processed, X_woe, preprocessor, woe_model

    return X_processed, preprocessor


# task-4
def calculate_rfm(df: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    """
    Calculate Recency, Frequency, and Monetary metrics per customer.
    """

    df = df.copy()
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    snapshot_date = pd.to_datetime(snapshot_date)

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=("TransactionStartTime",
                     lambda x: (snapshot_date - x.max()).days),
            Frequency=("TransactionId", "count"),
            Monetary=("Value", "sum")
        )
        .reset_index()
    )

    return rfm


def scale_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    return rfm_scaled


def cluster_customers(rfm_scaled: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    clusters = kmeans.fit_predict(rfm_scaled)
    return clusters


def identify_high_risk_cluster(rfm: pd.DataFrame) -> int:
    cluster_summary = (
        rfm.groupby("cluster")
        .agg({
            "Recency": "mean",
            "Frequency": "mean",
            "Monetary": "mean"
        })
        .reset_index()
    )

    # Sort by worst engagement
    high_risk_cluster = (
        cluster_summary
        .sort_values(
            by=["Frequency", "Monetary", "Recency"],
            ascending=[True, True, False]
        )
        .iloc[0]["cluster"]
    )

    return int(high_risk_cluster)


def assign_high_risk_label(rfm: pd.DataFrame, high_risk_cluster: int) -> pd.DataFrame:
    rfm["is_high_risk"] = np.where(
        rfm["cluster"] == high_risk_cluster,
        1,
        0
    )
    return rfm[["CustomerId", "is_high_risk"]]


def create_proxy_target(
    df: pd.DataFrame,
    snapshot_date: str = "2019-01-01"
) -> pd.DataFrame:

    rfm = calculate_rfm(df, snapshot_date)
    rfm_scaled = scale_rfm(rfm)

    rfm["cluster"] = cluster_customers(rfm_scaled)

    high_risk_cluster = identify_high_risk_cluster(rfm)

    target = assign_high_risk_label(rfm, high_risk_cluster)

    return target


def merge_target(df_processed: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    return df_processed.merge(
        target,
        on="CustomerId",
        how="left"
    )


