import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set DagsHub credentials
os.environ['MLFLOW_TRACKING_USERNAME'] = 'agummds'  # Your DagsHub username
os.environ['MLFLOW_TRACKING_PASSWORD'] = '48ac90871d91db0a55ba38bee5c924b6ed95ec25'  # Replace with your DagsHub access token

# Set MLflow tracking URI to your DagsHub repository
mlflow.set_tracking_uri("https://dagshub.com/agummds/Stunting-Toddler.mlflow")

def load_data():
    """Load preprocessed data"""
    # Load the preprocessed data
    X_train = pd.read_csv('../preprocessing/data_balita_preprocessing/X_train.csv')
    X_test = pd.read_csv('../preprocessing/data_balita_preprocessing/X_test.csv')
    y_train = pd.read_csv('../preprocessing/data_balita_preprocessing/y_train.csv')
    y_test = pd.read_csv('../preprocessing/data_balita_preprocessing/y_test.csv')
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train the model"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train.values.ravel())  # Added .values.ravel() to handle the DataFrame
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    # Additional custom metrics
    metrics['prediction_confidence'] = np.mean(model.predict_proba(X_test).max(axis=1))
    metrics['feature_importance_mean'] = np.mean(model.feature_importances_)
    
    return metrics, y_pred

def plot_confusion_matrix(y_test, y_pred, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Start MLflow run
    with mlflow.start_run(run_name="model_training"):
        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("test_size", 0.2)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Create and log confusion matrix
        plot_confusion_matrix(y_test, y_pred, "confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
        # Log feature importance plot
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
        plt.title('Top 10 Feature Importance')
        plt.savefig("feature_importance.png")
        plt.close()
        
        mlflow.log_artifact("feature_importance.png")

if __name__ == "__main__":
    main() 