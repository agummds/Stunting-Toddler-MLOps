from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import mlflow.pyfunc
import joblib
import os
import requests
import tempfile

app = Flask(__name__)

# Initialize paths
model_path = os.getenv('MODEL_PATH', '../Membangun_model/models/model')
preprocessor_url = 'https://github.com/agummds/Stunting-Toddler-MLOps/raw/automated-preprocessing/preprocessing/data_balita_preprocessing/preprocessors.joblib'

print(f"Initializing Flask app with model path: {model_path}")
print(f"Preprocessor URL: {preprocessor_url}")

try:
    # Load MLflow model
    model = mlflow.pyfunc.load_model(model_path)
    print("✅ MLflow model loaded successfully!")
    
    # Download and load preprocessor from GitHub
    print("Downloading preprocessor from GitHub...")
    response = requests.get(preprocessor_url)
    response.raise_for_status()  # Raises an HTTPError for bad responses
    
    # Save to temporary file and load
    with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as temp_file:
        temp_file.write(response.content)
        temp_path = temp_file.name
    
    preprocessor = joblib.load(temp_path)
    os.unlink(temp_path)  # Delete temp file
    
    print("✅ Preprocessor loaded successfully!")
    print(f"Preprocessor type: {type(preprocessor)}")
    print(f"Preprocessor content: {preprocessor}")
    
    # Check if it's a dictionary containing multiple preprocessors
    if isinstance(preprocessor, dict):
        print("Preprocessor is a dictionary with keys:", list(preprocessor.keys()))
        for key, value in preprocessor.items():
            print(f"  {key}: {type(value)}")
    
except Exception as e:
    print(f"Failed to initialize model: {str(e)}")
    import traceback
    traceback.print_exc()
    raise

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'GET':
            return jsonify({
                'message': 'Predict endpoint is working. Send a POST request with JSON data.',
                'example': {
                    'method': 'POST',
                    'url': '/predict',
                    'body': {
                        'input': {
                            'Umur (bulan)': 24,
                            'Jenis Kelamin': 1,  # 0 untuk Perempuan, 1 untuk Laki-laki
                            'Tinggi Badan (cm)': 87.5
                        }
                    }
                },
                'description': 'Send preprocessed data. Jenis Kelamin: 0=Perempuan, 1=Laki-laki',
                'status': 'info'
            })
            
        data = request.get_json()
        if not data or 'input' not in data:
            return jsonify({
                'error': 'Invalid input format. Expected JSON with "input" field',
                'status': 'error'
            }), 400
        
        input_data = data['input']
        
        # Convert input data to DataFrame with proper column names
        if isinstance(input_data, dict):
            # Handle dictionary input (recommended format)
            # Convert text gender to numeric if needed
            if 'Jenis Kelamin' in input_data:
                if isinstance(input_data['Jenis Kelamin'], str):
                    if input_data['Jenis Kelamin'].lower() in ['laki-laki', 'laki', 'male', 'm']:
                        input_data['Jenis Kelamin'] = 1
                    elif input_data['Jenis Kelamin'].lower() in ['perempuan', 'female', 'f']:
                        input_data['Jenis Kelamin'] = 0
            
            df_input = pd.DataFrame([input_data])
        elif isinstance(input_data, list) and len(input_data) == 3:
            # Handle list input [umur, jenis_kelamin, tinggi_badan]
            df_input = pd.DataFrame([{
                'Umur (bulan)': input_data[0],
                'Jenis Kelamin': input_data[1],
                'Tinggi Badan (cm)': input_data[2]
            }])
        else:
            return jsonify({
                'error': 'Invalid input format. Expected dict with keys: "Umur (bulan)", "Jenis Kelamin" (0/1), "Tinggi Badan (cm)" or list with 3 values',
                'status': 'error'
            }), 400
        
        print(f"Input DataFrame: {df_input}")
        
        # Apply preprocessing using the loaded preprocessor
        if isinstance(preprocessor, dict):
            # If preprocessor is a dictionary, we need to handle it differently
            print("Preprocessor is a dictionary, checking available keys...")
            
            # Common keys that might contain the actual preprocessor
            possible_keys = ['preprocessor', 'scaler', 'transformer', 'pipeline']
            actual_preprocessor = None
            
            for key in possible_keys:
                if key in preprocessor:
                    actual_preprocessor = preprocessor[key]
                    print(f"Using preprocessor from key: {key}")
                    break
            
            if actual_preprocessor is None:
                # Try the first item in the dictionary
                first_key = list(preprocessor.keys())[0]
                actual_preprocessor = preprocessor[first_key]
                print(f"Using first preprocessor from key: {first_key}")
            
            X_processed = actual_preprocessor.transform(df_input)
        else:
            # Direct preprocessor object
            X_processed = preprocessor.transform(df_input)
        
        print(f"After preprocessing shape: {X_processed.shape}")
        
        # Make prediction
        prediction = model.predict(X_processed)
        print(f"Prediction result: {prediction}")
        
        return jsonify({
            'prediction': prediction.tolist(),
            'status': 'success',
            'input_processed_shape': X_processed.shape
        })
            
    except Exception as e:
        print(f"Prediction error details: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("Starting Flask server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True) 