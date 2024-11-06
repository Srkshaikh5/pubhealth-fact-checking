Here's a comprehensive **README.md** file that includes all the necessary information for setting up and running the project. This README covers system requirements, installation, data setup, running the various scripts, and troubleshooting.

---

# PUBHEALTH Fact-Checking System

This repository provides a machine learning pipeline for fact-checking public health claims using the PUBHEALTH dataset. The system includes data ingestion, processing, model training, and a deployment-ready API built with FastAPI and Docker.

## Table of Contents

- [System Requirements](#system-requirements)
- [Setup and Installation](#setup-and-installation)
- [Project Structure](#project-structure)
- [Running the Pipeline](#running-the-pipeline)
  - [1. Data Ingestion](#1-data-ingestion)
  - [2. Data Processing](#2-data-processing)
  - [3. Model Training](#3-model-training)
  - [4. Model Deployment with FastAPI](#4-model-deployment-with-fastapi)
  - [5. Docker Deployment](#5-docker-deployment)
- [Kubernetes Deployment (Optional)](#kubernetes-deployment-optional)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)

## System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: 3.8 or later
- **CUDA** (optional): GPU with CUDA support if you want to use GPU for faster training
- **Docker**: Needed for containerized deployment
- **Kubernetes** (Optional): If deploying on Kubernetes

## Setup and Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Srkshaikh5/pubhealth-fact-checking.git
cd pubhealth-fact-checking
```

### Step 2: Install Python Dependencies

It is recommended to set up a virtual environment for the project.

```bash
python -m venv env
source env/bin/activate   # On Windows use `env\Scripts\activate`
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

#### Main Libraries:

- `transformers`: For model loading and fine-tuning
- `datasets`: For handling and loading the PUBHEALTH dataset
- `fastapi`: For deploying the API
- `uvicorn`: For running the FastAPI server
- `scikit-learn`, `pandas`, `torch`: Core libraries for data processing and model training

### Step 3: Download the Data

The PUBHEALTH dataset is available through the Hugging Face `datasets` library. This will be handled in the `ingest.py` script.

## Project Structure

```
├── data/                   # Folder for raw and processed datasets
├── ingest.py               # Script to download and save dataset
├── prepare.py              # Script to clean and process the data
├── train.py                # Script to train the model
├── serve.py                # FastAPI application for model deployment
├── Dockerfile              # Docker configuration for deployment
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Running the Pipeline

### 1. Data Ingestion

To download and save the dataset, run the `ingest.py` script. By default, it saves the dataset in the `data` folder.

```bash
python ingest.py
```

### 2. Data Processing

Process the data to prepare it for training. This script cleans the data, handles missing values, combines text fields, and encodes labels.

```bash
python prepare.py
```

Output: This will create `train_processed.csv` and `test_processed.csv` in the `data` folder.

### 3. Model Training

Train the model with the `train.py` script. This will load the processed data, fine-tune a pretrained model, and save the output.

```bash
python train.py
```

### 4. Model Deployment with FastAPI

To deploy the model as an API, use `serve.py`. This script will load the trained model and provide an endpoint to make predictions on new claims.

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000/claim/v1/predict`.

**Sample Request**:

You can test the API by sending a POST request:

```bash
curl -X POST "http://localhost:8000/claim/v1/predict" -H "Content-Type: application/json" -d '{"claim": "This is a health claim to check."}'
```
```
# Payload
{
  "claim": "This is a health-related fact."
}

# Output
{
  "veracity": "true"
}
```
### 5. Docker Deployment

To deploy the application in a containerized environment:

1. **Build the Docker image**:

   ```bash
   docker build -t health-claim-serve .
   ```

2. **Run the Docker container**:

   ```bash
   docker run -p 8000:80 health-claim-serve
   ```

The API should now be accessible at `http://localhost:8000/claim/v1/predict`.

## Kubernetes Deployment (Optional)

To deploy on Kubernetes, use the provided `kubernetes-deployment.yaml` configuration file.

1. Apply the deployment:

   ```bash
   kubectl apply -f kubernetes-deployment.yaml
   ```

2. Check the deployment status:

   ```bash
   kubectl get pods
   ```

Once deployed, you may need to expose the service using a LoadBalancer or NodePort to access it.

## Troubleshooting

- **CUDA Error**: If you encounter errors like `CUDA error: device-side assert triggered`, verify that all labels are within the correct range (`[0, 3]`).
- **Out of Range Labels**: Ensure the labels in the dataset are integers within the expected range by checking them in `prepare.py` (see the `validate_labels` function).
- **Environment Variables**: If you see issues related to environment variables, ensure they are set up correctly (e.g., `TF_ENABLE_ONEDNN_OPTS=0` to disable oneDNN optimizations for TensorFlow).

## Contributors

- **Saruk Shaikh** - Project setup and development
- **Onclusive** - Original problem statement