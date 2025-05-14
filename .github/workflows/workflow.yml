name: ML Workflow CI
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Set up Conda
      uses: conda-incubator/setup-miniconda@v2
      with:
        python-version: 3.8
        activate-environment: ml-project
        environment-file: MLProject/conda.yaml
    
    - name: Install dependencies
      shell: bash -l {0}
      run: |
        conda activate ml-project
        pip install -r requirements.txt
    
    - name: Run tests
      shell: bash -l {0}
      run: |
        conda activate ml-project
        python -m pytest tests/
    
    - name: Train and evaluate model
      shell: bash -l {0}
      run: |
        conda activate ml-project
        python MLProject/modelling.py
    
    - name: Upload model artifact
      uses: actions/upload-artifact@v2
      with:
        name: trained-model
        path: model_balita.joblib 