from flask import Flask, request, jsonify
from prometheus_exporter import ModelExporter
import numpy as np
import mlflow.pyfunc
import os

app = Flask(__name__)

# Initialize the model exporter
model_path = os.getenv('MODEL_PATH', '../Membangun_model/models/model')  # Updated path to local model
print(f"Initializing Flask app with model path: {model_path}")

try:
    model_exporter = ModelExporter(model_path)
except Exception as e:
    print(f"Failed to initialize model: {str(e)}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'input' not in data:
            return jsonify({
                'error': 'Invalid input format. Expected JSON with "input" field',
                'status': 'error'
            }), 400
            
        input_data = np.array(data['input'])
        
        # Make prediction using the model exporter
        prediction = model_exporter.predict(input_data)
        
        if prediction is not None:
            return jsonify({
                'prediction': prediction.tolist(),
                'status': 'success'
            })
        else:
            return jsonify({
                'error': 'Prediction failed',
                'status': 'error'
            }), 500
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("Starting Flask server on port 5000...")
    app.run(host='0.0.0.0', port=5000) 