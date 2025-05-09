import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from datetime import datetime

# Set MLflow tracking URI to DagsHub
MLFLOW_TRACKING_URI = "https://dagshub.com/agummds/Stunting-Toddler.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Set experiment name
EXPERIMENT_NAME = "Stunting_Prediction"
mlflow.set_experiment(EXPERIMENT_NAME)

def load_data():
    """Load preprocessed data"""
    # Adjust path according to your structure
    data_path = "../preprocessing/namadataset_preprocessing/preprocessed_data.csv"
    return pd.read_csv(data_path)

def prepare_data(df):
    """Prepare data for training"""
    # Assuming 'target' is your target column
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, X_test, y_train, y_test):
    """Train model with MLflow tracking"""
    with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42
        }
        mlflow.log_params(params)

        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log additional custom metrics
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        })
        mlflow.log_text(feature_importance.to_string(), "feature_importance.txt")

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log training data info
        mlflow.log_metric("training_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        mlflow.log_metric("feature_count", X_train.shape[1])

        return model, metrics

def main():
    # Load and prepare data
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Train model and log metrics
    model, metrics = train_model(X_train, X_test, y_train, y_test)

    # Print results
    print("\nModel Performance Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main() 