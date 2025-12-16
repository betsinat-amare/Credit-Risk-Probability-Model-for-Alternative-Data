import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from src.data_processing import process_data_end_to_end


def load_and_prepare_data(path: str):
    df = pd.read_csv(path)

    df_processed = process_data_end_to_end(df)

    X = df_processed.drop(columns=["is_high_risk"])
    y = df_processed["is_high_risk"]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }


def train_logistic_regression(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="Logistic_Regression"):
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        return metrics


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        "data/processed/processed_data.csv"
    )

    print("Training Logistic Regression...")
    train_logistic_regression(X_train, y_train, X_test, y_test)

    print("Training Random Forest...")
    train_random_forest(X_train, y_train, X_test, y_test)


