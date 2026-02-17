import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from src.data_processing import process_data_end_to_end


def load_and_prepare_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads raw data, processes it into features and target, and splits into train/test sets.
    """
    df = pd.read_csv(path)

    df_processed = process_data_end_to_end(df)

    # Drop non-feature columns
    X = df_processed.drop(columns=["is_high_risk", "CustomerId"])
    y = df_processed["is_high_risk"]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
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


def log_shap_summary(model: Any, X_train: pd.DataFrame, run_name: str):
    """
    Computes and logs a SHAP summary plot to MLflow.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, show=False)
    plot_path = f"shap_summary_{run_name}.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.close()


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, 
                              X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Trains and logs a Logistic Regression model.
    """
    with mlflow.start_run(run_name="Logistic_Regression"):
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        # Log feature importance via SHAP
        log_shap_summary(model, X_train, "Logistic_Regression")

        return metrics


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Trains and logs a Random Forest Classifier.
    """
    with mlflow.start_run(run_name="Random_Forest"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        # Log feature importance via SHAP
        log_shap_summary(model, X_train, "Random_Forest")

        return metrics


if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test = load_and_prepare_data(
            "data/raw/data.csv"
        )

        print("Training Logistic Regression...")
        lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test)
        print(f"LR Metrics: {lr_metrics}")

        print("Training Random Forest...")
        rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
        print(f"RF Metrics: {rf_metrics}")

    except FileNotFoundError:
        print("Error: CSV data file not found. Ensure 'data/raw/data.csv' exists.")
    except Exception as e:
        print(f"An error occurred: {e}")


