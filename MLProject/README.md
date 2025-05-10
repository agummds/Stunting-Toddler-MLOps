# Stunting Prediction MLflow Project

This project implements a machine learning pipeline for stunting prediction using MLflow for experiment tracking and model management.

## Project Structure

```
MLProject/
├── MLProject          # MLflow project configuration
├── conda.yaml         # Conda environment specification
├── modelling.py       # Main training script
├── Dockerfile         # Docker configuration
└── README.md         # This file
```

## Setup

1. Install dependencies:
```bash
conda env create -f conda.yaml
```

2. Activate the environment:
```bash
conda activate stunting-prediction
```

## Running the Project

### Local Execution
```bash
mlflow run . --experiment-name "stunting-prediction"
```

### Docker Execution
```bash
docker build -t stunting-prediction .
docker run -e MLFLOW_TRACKING_USERNAME=<username> -e MLFLOW_TRACKING_PASSWORD=<password> stunting-prediction
```

## CI/CD

The project includes GitHub Actions workflow for continuous integration. The workflow:
1. Triggers on push to main branch
2. Sets up Python environment
3. Installs dependencies
4. Runs MLflow project
5. Tracks experiments in DagsHub

## Model Artifacts

The trained model and artifacts are stored in DagsHub MLflow tracking server:
- Model artifacts
- Metrics
- Parameters
- Confusion matrix
- Feature importance plots

## Environment Variables

Required environment variables:
- MLFLOW_TRACKING_USERNAME: DagsHub username
- MLFLOW_TRACKING_PASSWORD: DagsHub access token 