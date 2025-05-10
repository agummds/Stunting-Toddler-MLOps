FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy project files
COPY . /app/

# Create conda environment
RUN conda env create -f conda.yaml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "stunting-prediction", "/bin/bash", "-c"]

# Set environment variables
ENV MLFLOW_TRACKING_URI="https://dagshub.com/agummds/Stunting-Toddler.mlflow"

# Command to run when container starts
CMD ["conda", "run", "-n", "stunting-prediction", "python", "modelling.py"] 