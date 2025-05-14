import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load data from file"""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the dataset"""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    return df

def preprocess_features(df, target_column):
    """Preprocess features for model training"""
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def save_processed_data(X, y, output_path):
    """Save processed data"""
    processed_df = pd.DataFrame(X)
    processed_df['target'] = y
    processed_df.to_csv(output_path, index=False) 