from prometheus_client import start_http_server, Counter, Gauge, Histogram
import time
import numpy as np
from mlflow.pyfunc import load_model
import psutil
import os

# Initialize metrics
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total number of predictions made')
PREDICTION_LATENCY = Histogram('model_prediction_latency_seconds', 'Time spent processing predictions')
MODEL_LOAD_TIME = Gauge('model_load_time_seconds', 'Time taken to load the model')
CPU_USAGE = Gauge('model_cpu_usage_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('model_memory_usage_bytes', 'Memory usage in bytes')
PREDICTION_PROBABILITY = Gauge('model_prediction_probability', 'Prediction probability for the latest prediction')
BATCH_SIZE = Gauge('model_batch_size', 'Current batch size being processed')
ERROR_COUNTER = Counter('model_errors_total', 'Total number of errors encountered')
MODEL_VERSION = Gauge('model_version', 'Current version of the model')
REQUEST_QUEUE_SIZE = Gauge('model_request_queue_size', 'Current size of the request queue')

class ModelExporter:
    def __init__(self, model_path):
        self.model = None
        self.model_path = model_path
        print(f"Initializing ModelExporter with model path: {model_path}")
        self.load_model()
        
    def load_model(self):
        start_time = time.time()
        try:
            # Try to load the model from MLflow
            print(f"Loading model from: {self.model_path}")
            self.model = load_model(self.model_path)
            MODEL_LOAD_TIME.set(time.time() - start_time)
            MODEL_VERSION.set(1.0)  # Set your model version here
            print("Model loaded successfully")
        except Exception as e:
            ERROR_COUNTER.inc()
            print(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, input_data):
        start_time = time.time()
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
                
            # Update system metrics
            CPU_USAGE.set(psutil.cpu_percent())
            MEMORY_USAGE.set(psutil.Process(os.getpid()).memory_info().rss)
            
            # Make prediction
            prediction = self.model.predict(input_data)
            
            # Update metrics
            PREDICTION_COUNTER.inc()
            PREDICTION_LATENCY.observe(time.time() - start_time)
            PREDICTION_PROBABILITY.set(float(prediction[0]))  # Assuming binary classification
            BATCH_SIZE.set(len(input_data))
            
            return prediction
        except Exception as e:
            ERROR_COUNTER.inc()
            print(f"Error during prediction: {str(e)}")
            return None

def main():
    # Start Prometheus metrics server
    print("Starting Prometheus metrics server on port 8000...")
    start_http_server(8000)
    
    # Initialize model exporter
    model_path = os.getenv('MODEL_PATH', 'MLProject/models/model')  # Updated path to local model
    print(f"Using model path: {model_path}")
    model_exporter = ModelExporter(model_path)
    
    print("Server is running. Press Ctrl+C to stop.")
    # Keep the server running
    while True:
        time.sleep(1)

if __name__ == '__main__':
    main() 