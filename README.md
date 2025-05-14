# ML Project with CI/CD Workflow

This project implements a machine learning pipeline with continuous integration and deployment workflow.

## Project Structure

```
Workflow-CI
├── .workflow
├── MLProject
    ├── modelling.py
    ├── conda.yaml
    ├── MLProject
    ├── dataset_preprocessing.py
```

## Setup

1. Create a conda environment:
```bash
conda env create -f MLProject/conda.yaml
```

2. Activate the environment:
```bash
conda activate ml-project
```

## Usage

1. Preprocess your data:
```python
from dataset_preprocessing import load_data, clean_data, preprocess_features

# Load and preprocess data
df = load_data('your_data.csv')
df_clean = clean_data(df)
X, y = preprocess_features(df_clean, 'target_column')
```

2. Train the model:
```python
from modelling import MLModel

model = MLModel()
model.train(X, y)
```

## Docker Hub

To use the Docker image:
```bash
docker pull your-username/ml-project:latest
```

## CI/CD Pipeline

The project includes a GitHub Actions workflow that:
- Runs on push to main branch
- Sets up Python environment
- Installs dependencies
- Runs tests
- Builds and pushes Docker image 