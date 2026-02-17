import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple, Any, Optional

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from src.data_processing import process_data_end_to_end
from src.config import config


def load_and_prepare_data(
    path: str = config.data.raw_data_path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads raw data, processes it into features and target, and splits into train/test sets.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")

    df = pd.read_csv(path)
    df_processed = process_data_end_to_end(df)

    # Drop non-feature columns
    X = df_processed.drop(columns=[config.data.target_col, config.data.customer_id_col])
    y = df_processed[config.data.target_col]

    return train_test_split(
        X, y,
        test_size=config.model.test_size,
        random_state=config.model.random_state,
        stratify=y
    )


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluates the model using various classification metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba))
    }


def log_shap_summary(model: Any, X_train: pd.DataFrame, run_name: str) -> None:
    """
    Computes and logs a SHAP summary plot to MLflow.
    """
    try:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_train, show=False)
        plot_path = f"shap_summary_{run_name}.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not log SHAP summary for {run_name}: {e}")


def train_model(
    model_type: str, 
    X_train: pd.DataFrame, y_train: pd.Series, 
    X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """
    Generic model training and logging function.
    """
    run_name = model_type.replace("_", " ").title()
    
    with mlflow.start_run(run_name=run_name):
        if model_type == "logistic_regression":
            model = LogisticRegression(
                max_iter=config.model.lr_max_iter, 
                random_state=config.model.random_state
            )
        elif model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=config.model.rf_n_estimators, 
                random_state=config.model.random_state
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        # MLflow logging
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        # Log feature importance via SHAP
        log_shap_summary(model, X_train, run_name)

        return metrics


def run_training_pipeline():
    """
    Main entry point for the training pipeline.
    """
    try:
        print("Loading and preparing data...")
        X_train, X_test, y_train, y_test = load_and_prepare_data()

        print("Training Logistic Regression...")
        lr_metrics = train_model("logistic_regression", X_train, y_train, X_test, y_test)
        print(f"LR Metrics: {lr_metrics}")

        print("Training Random Forest...")
        rf_metrics = train_model("random_forest", X_train, y_train, X_test, y_test)
        print(f"RF Metrics: {rf_metrics}")

    except Exception as e:
        print(f"An error occurred during the training pipeline: {e}")


if __name__ == "__main__":
    run_training_pipeline()


