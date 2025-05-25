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

def load_raw_data():
    """Load raw data"""
    # Adjust path as needed
    data = pd.read_csv('../Eksperimen_SML_Agum Medisa/data/raw_data.csv') 
    return data

def preprocess_data(data, test_size=0.2, random_state=42):
    """Apply comprehensive preprocessing pipeline"""
    
    # 1. Identify numerical and categorical columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target column from features if it exists
    if 'target_column' in numerical_cols:  # Replace with actual target column name
        numerical_cols.remove('target_column')
    if 'target_column' in categorical_cols:
        categorical_cols.remove('target_column')
    
    # Define preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
        ('scaler', StandardScaler())  # Standardize features
    ])
    
    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Encode categorical variables
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Assume the last column is the target (adjust as needed)
    target_column = 'target_column'  # Replace with actual target column name
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Encode target if needed
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transform the test data
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding
    feature_names = []
    
    # Add numerical feature names
    feature_names.extend(numerical_cols)
    
    # Add encoded categorical feature names
    for i, col in enumerate(categorical_cols):
        encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        categories = encoder.categories_[i]
        feature_names.extend([f"{col}_{category}" for category in categories])
    
    # Apply feature selection
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
    X_train_selected = selector.fit_transform(X_train_processed, y_train)
    X_test_selected = selector.transform(X_test_processed)
    
    # Get selected feature indices
    selected_indices = selector.get_support()
    selected_feature_names = [name for i, name in enumerate(feature_names) if i < len(selected_indices) and selected_indices[i]]
    
    # Convert to DataFrame for better handling
    X_train_df = pd.DataFrame(X_train_selected, columns=selected_feature_names)
    X_test_df = pd.DataFrame(X_test_selected, columns=selected_feature_names)
    y_train_df = pd.DataFrame(y_train, columns=['target'])
    y_test_df = pd.DataFrame(y_test, columns=['target'])
    
    # Save preprocessed data
    os.makedirs('../Eksperimen_SML_Agum Medisa/preprocessing/data_balita_preprocessing', exist_ok=True)
    X_train_df.to_csv('../Eksperimen_SML_Agum Medisa/preprocessing/data_balita_preprocessing/X_train.csv', index=False)
    X_test_df.to_csv('../Eksperimen_SML_Agum Medisa/preprocessing/data_balita_preprocessing/X_test.csv', index=False)
    y_train_df.to_csv('../Eksperimen_SML_Agum Medisa/preprocessing/data_balita_preprocessing/y_train.csv', index=False)
    y_test_df.to_csv('../Eksperimen_SML_Agum Medisa/preprocessing/data_balita_preprocessing/y_test.csv', index=False)
    
    # Save preprocessor for later use
    import joblib
    os.makedirs('preprocessors', exist_ok=True)
    joblib.dump(preprocessor, 'preprocessors/main_preprocessor.joblib')
    joblib.dump(selector, 'preprocessors/feature_selector.joblib')
    joblib.dump(le, 'preprocessors/target_encoder.joblib')
    
    return X_train_df, X_test_df, y_train_df, y_test_df, preprocessor, selector

def load_preprocessed_data():
    """Load preprocessed data"""
    X_train = pd.read_csv('../Eksperimen_SML_Agum Medisa/preprocessing/data_balita_preprocessing/X_train.csv')
    X_test = pd.read_csv('../Eksperimen_SML_Agum Medisa/preprocessing/data_balita_preprocessing/X_test.csv')
    y_train = pd.read_csv('../Eksperimen_SML_Agum Medisa/preprocessing/data_balita_preprocessing/y_train.csv')
    y_test = pd.read_csv('../Eksperimen_SML_Agum Medisa/preprocessing/data_balita_preprocessing/y_test.csv')
    
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
    
    # Convert to numpy arrays to handle multi-class cases properly
    y_test_np = y_test.values.ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_test_np, y_pred),
        'precision': precision_score(y_test_np, y_pred, average='weighted'),
        'recall': recall_score(y_test_np, y_pred, average='weighted'),
        'f1': f1_score(y_test_np, y_pred, average='weighted')
    }
    
    # Additional custom metrics
    metrics['prediction_confidence'] = np.mean(model.predict_proba(X_test).max(axis=1))
    metrics['feature_importance_mean'] = np.mean(model.feature_importances_)
    
    # Add ROC AUC if binary classification
    if len(np.unique(y_test_np)) == 2:
        metrics['roc_auc'] = roc_auc_score(y_test_np, y_pred_proba[:, 1])
    
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
    
def plot_roc_curve(model, X_test, y_test, save_path):
    """Plot ROC curve for binary classification"""
    y_test_np = y_test.values.ravel()
    
    # Only for binary classification
    if len(np.unique(y_test_np)) == 2:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_np, y_pred_proba)
        auc_score = roc_auc_score(y_test_np, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')  # Random classifier line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(save_path)
        plt.close()
        return auc_score
    return None

def main():
    """Main execution function"""
    # Check if preprocessed data exists
    preprocess_path = '../Eksperimen_SML_Agum Medisa/preprocessing/data_balita_preprocessing/X_train.csv'
    
    if not os.path.exists(preprocess_path):
        print("Preprocessed data not found. Running preprocessing pipeline...")
        data = load_raw_data()
        X_train, X_test, y_train, y_test, preprocessor, selector = preprocess_data(data)
        print("Preprocessing complete and data saved.")
    else:
        print("Loading preprocessed data...")
        X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Create local model directory
    local_model_dir = "models"
    os.makedirs(local_model_dir, exist_ok=True)
    
    # Start MLflow run
    with mlflow.start_run(run_name="model_training_v2"):
        # Log preprocessing parameters
        mlflow.log_param("preprocessing", "StandardScaler + OneHotEncoder + FeatureSelection")
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
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        local_model_path = os.path.join(local_model_dir, "model")
        mlflow.sklearn.save_model(model, local_model_path)
        
        # Create and log confusion matrix
        plot_confusion_matrix(y_test, y_pred, "confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
        # Create and log ROC curve (if binary classification)
        roc_auc = plot_roc_curve(model, X_test, y_test, "roc_curve.png")
        if roc_auc:
            mlflow.log_artifact("roc_curve.png")
        
        # Log feature importance plot
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
        plt.title('Top 15 Feature Importance')
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.close()
        
        mlflow.log_artifact("feature_importance.png")

if __name__ == "__main__":
    main()