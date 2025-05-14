import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class MLModel:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()
        
    def load_data(self, data_dir):
        """Load preprocessed data"""
        X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
        y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
        y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))
        
        return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()
    
    def train(self, X_train, y_train):
        """Train the model"""
        X_train_scaled = self.preprocess_data(X_train)
        self.model.fit(X_train_scaled, y_train)
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        X_test_scaled = self.preprocess_data(X_test)
        return self.model.score(X_test_scaled, y_test)
    
    def preprocess_data(self, data):
        """Preprocess the input data"""
        return self.scaler.fit_transform(data)
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.preprocess_data(X)
        return self.model.predict(X_scaled)
    
    def save_model(self, path):
        """Save the model to disk"""
        joblib.dump(self.model, path)
    
    def load_model(self, path):
        """Load the model from disk"""
        self.model = joblib.load(path)

if __name__ == "__main__":
    # Path ke data (menggunakan path relatif)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_balita_preprocessing')
    
    # Inisialisasi model
    model = MLModel()
    
    # Load data
    X_train, X_test, y_train, y_test = model.load_data(data_dir)
    
    # Train model
    model.train(X_train, y_train)
    
    # Evaluate model
    score = model.evaluate(X_test, y_test)
    print(f"Model accuracy: {score}")
    
    # Simpan model
    model.save_model('model_balita.joblib') 