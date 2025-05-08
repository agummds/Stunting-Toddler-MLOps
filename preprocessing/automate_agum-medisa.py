import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(url=None, file_path=None):
    """
    Load the dataset from either a URL or local file path
    
    Parameters:
    -----------
    url : str, optional
        URL to the dataset
    file_path : str, optional
        Local path to the dataset file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded dataset
    """
    if url:
        return pd.read_csv(url)
    elif file_path:
        return pd.read_csv(file_path)
    else:
        raise ValueError("Either url or file_path must be provided")

def preprocess_data(df, save_path=None):
    """
    Preprocess the dataset for model training
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    save_path : str, optional
        Path to save preprocessed data and preprocessors
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test, preprocessors
    """
    # Create copies of encoders for later use
    label_encoders = {}
    
    # Encode categorical variables
    categorical_columns = ['Jenis Kelamin', 'Status Gizi']
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Separate features and target
    X = df.drop('Status Gizi', axis=1)
    y = df['Status Gizi']
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_columns = ['Umur (bulan)', 'Tinggi Badan (cm)']
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Store all preprocessors
    preprocessors = {
        'label_encoders': label_encoders,
        'scaler': scaler
    }
    
    # Save preprocessed data and preprocessors if save_path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save preprocessed data
        X_train.to_csv(os.path.join(save_path, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(save_path, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(save_path, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(save_path, 'y_test.csv'), index=False)
        
        # Save preprocessors
        joblib.dump(preprocessors, os.path.join(save_path, 'preprocessors.joblib'))
        
        print(f"Preprocessed data and preprocessors saved to {save_path}")
    
    return X_train, X_test, y_train, y_test, preprocessors

def inverse_transform_predictions(y_pred, preprocessors):
    """
    Convert encoded predictions back to original labels
    
    Parameters:
    -----------
    y_pred : array-like
        Encoded predictions
    preprocessors : dict
        Dictionary containing the preprocessors
        
    Returns:
    --------
    array-like
        Original label predictions
    """
    return preprocessors['label_encoders']['Status Gizi'].inverse_transform(y_pred)

def preprocess_new_data(new_data, preprocessors):
    """
    Preprocess new data using the same transformations as training data
    
    Parameters:
    -----------
    new_data : pandas.DataFrame
        New data to preprocess
    preprocessors : dict
        Dictionary containing the preprocessors
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed data ready for prediction
    """
    # Create a copy to avoid modifying the original
    processed_data = new_data.copy()
    
    # Apply label encoding to categorical variables
    for col, encoder in preprocessors['label_encoders'].items():
        if col in processed_data.columns:
            processed_data[col] = encoder.transform(processed_data[col])
    
    # Scale numerical features
    numerical_columns = ['Umur (bulan)', 'Tinggi Badan (cm)']
    processed_data[numerical_columns] = preprocessors['scaler'].transform(
        processed_data[numerical_columns]
    )
    
    return processed_data

def load_preprocessed_data(load_path):
    """
    Load preprocessed data and preprocessors from saved files
    
    Parameters:
    -----------
    load_path : str
        Path to the directory containing saved preprocessed data
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test, preprocessors
    """
    # Load preprocessed data
    X_train = pd.read_csv(os.path.join(load_path, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(load_path, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(load_path, 'y_train.csv'))
    y_test = pd.read_csv(os.path.join(load_path, 'y_test.csv'))
    
    # Load preprocessors
    preprocessors = joblib.load(os.path.join(load_path, 'preprocessors.joblib'))
    
    return X_train, X_test, y_train, y_test, preprocessors

if __name__ == "__main__":
    # Example usage
    url = 'https://raw.githubusercontent.com/agummds/Predictive-Analytics/master/Dataset/data_balita.csv'
    save_path = 'data_balita_preprocessing'
    
    # Load and preprocess data
    df = load_data(url=url)
    X_train, X_test, y_train, y_test, preprocessors = preprocess_data(df, save_path=save_path)
    
    print("Data preprocessing completed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}") 