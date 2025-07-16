import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set MLflow tracking URI (public URL)
MLFLOW_TRACKING_URI = "https://dagshub.com/agummds/Stunting-Toddler.mlflow"

# Set MLflow credentials from environment variables
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')

# Set tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --- Define Paths ---
# Sebaiknya definisikan semua path di satu tempat agar mudah dikelola
RAW_DATA_PATH = 'data/raw_data.csv'
PROCESSED_DATA_DIR = 'data/processed'
PREPROCESSOR_DIR = 'preprocessors'
MODEL_DIR = 'models'
REPORTS_DIR = 'reports/figures'
TARGET_COLUMN = 'Status' # GANTI DENGAN NAMA KOLOM TARGET ANDA

def load_raw_data():
    """Load raw data"""
    # Pastikan direktori ada sebelum membuat file
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    # Gunakan path yang sudah didefinisikan
    data = pd.read_csv(RAW_DATA_PATH)
    return data

def preprocess_data(data, test_size=0.2, random_state=42):
    """Apply comprehensive preprocessing pipeline"""
    
    # 1. Identify numerical and categorical columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target column from features
    if TARGET_COLUMN in numerical_cols:
        numerical_cols.remove(TARGET_COLUMN)
    if TARGET_COLUMN in categorical_cols:
        categorical_cols.remove(TARGET_COLUMN)
    
    # Define preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_cols)
    feature_names = numerical_cols + list(cat_feature_names)

    # Apply feature selection
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
    X_train_selected = selector.fit_transform(X_train_processed, y_train)
    X_test_selected = selector.transform(X_test_processed)
    
    # Get selected feature names
    selected_feature_names = [name for name, supported in zip(feature_names, selector.get_support()) if supported]
    
    # Convert to DataFrame
    X_train_df = pd.DataFrame(X_train_selected, columns=selected_feature_names)
    X_test_df = pd.DataFrame(X_test_selected, columns=selected_feature_names)
    y_train_df = pd.DataFrame(y_train, columns=['target'])
    y_test_df = pd.DataFrame(y_test, columns=['target'])
    
    # Save preprocessed data
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    X_train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'), index=False)
    X_test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'), index=False)
    y_train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'), index=False)
    y_test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'), index=False)
    
    # Save preprocessors
    os.makedirs(PREPROCESSOR_DIR, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(PREPROCESSOR_DIR, 'main_preprocessor.joblib'))
    joblib.dump(selector, os.path.join(PREPROCESSOR_DIR, 'feature_selector.joblib'))
    joblib.dump(le, os.path.join(PREPROCESSOR_DIR, 'target_encoder.joblib'))
    
    return X_train_df, X_test_df, y_train_df, y_test_df, preprocessor, selector

def load_preprocessed_data():
    """Load preprocessed data"""
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'))
    y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'))
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train the model"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train.values.ravel())
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    y_test_np = y_test.values.ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_test_np, y_pred),
        'precision': precision_score(y_test_np, y_pred, average='weighted'),
        'recall': recall_score(y_test_np, y_pred, average='weighted'),
        'f1': f1_score(y_test_np, y_pred, average='weighted')
    }
    
    if len(np.unique(y_test_np)) == 2:
        metrics['roc_auc'] = roc_auc_score(y_test_np, y_pred_proba[:, 1])
    
    return metrics, y_pred

def plot_confusion_matrix(y_test, y_pred, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()
    
def plot_roc_curve(model, X_test, y_test, save_path):
    """Plot ROC curve for binary classification"""
    y_test_np = y_test.values.ravel()
    
    if len(np.unique(y_test_np)) == 2:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_np, y_pred_proba)
        auc_score = roc_auc_score(y_test_np, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(save_path)
        plt.close()
        return auc_score
    return None

def plot_feature_importance(model, features, save_path):
    """Plot and save feature importance"""
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Top 15 Feature Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    """Main execution function"""
    # Pastikan direktori untuk menyimpan output ada
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    preprocess_path = os.path.join(PROCESSED_DATA_DIR, 'X_train.csv')
    
    if not os.path.exists(preprocess_path):
        print("Preprocessed data not found. Running preprocessing pipeline...")
        raw_data = load_raw_data()
        X_train, X_test, y_train, y_test, _, _ = preprocess_data(raw_data)
        print("Preprocessing complete and data saved.")
    else:
        print("Loading preprocessed data...")
        X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    with mlflow.start_run(run_name="model_training_v3"):
        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save plots
        cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
        roc_path = os.path.join(REPORTS_DIR, "roc_curve.png")
        fi_path = os.path.join(REPORTS_DIR, "feature_importance.png")
        
        plot_confusion_matrix(y_test, y_pred, cm_path)
        plot_roc_curve(model, X_test, y_test, roc_path)
        plot_feature_importance(model, X_train.columns, fi_path)

        # Log artifacts (plots)
        mlflow.log_artifact(cm_path)
        if os.path.exists(roc_path):
            mlflow.log_artifact(roc_path)
        mlflow.log_artifact(fi_path)

        print(f"Model and artifacts logged to MLflow. Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    main()