# Boston Housing Price Predictor | MLOps Challenge

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Docker](https://img.shields.io/badge/Docker-Available-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-green)
![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)

A **delivery-ready, end-to-end MLOps pipeline** designed to predict housing prices using the Boston Housing dataset. This project implements a Cloud Agnostic and Self-Hosted architecture, prioritizing reproducibility, modular software design, and automated CI/CD workflows, strictly avoiding vendor lock-in from managed services.

---

## Table of Contents

- [Key Features](#-key-features)
- [Architecture & Design](#-architecture--design)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [MLOps Pipeline](#-mlops-pipeline)
- [Makefile Commands](#-makefile-commands)
- [Deployment (Docker)](#-deployment-docker)
- [API Reference](#-api-reference)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Project Structure](#-project-structure)
- [AI Tools Disclosure](#-ai-tools-disclosure)
- [Future Improvements & Scalability](#-future-improvements--scalability)


---

## Key Features

- **Reproducible Training Pipeline**  
  Automated workflow to train, evaluate, and compare multiple models:
  - Linear Regression  
  - Random Forest  
  - XGBoost  

- **Champion / Challenger Strategy**  
  Automatically selects and promotes the best-performing model based on **RMSE** and **R²**.

- **Experiment Tracking**  
  Integrated **MLflow** (SQLite backend) for parameters, metrics, and artifacts.

- **Production-Ready API**  
  High-performance REST API built with **FastAPI**, including:
  - Input validation
  - Health checks
  - Typed schemas

- **Containerized Deployment**  
  Fully Dockerized using **multi-stage builds** for optimized production images.

- **Developer Experience First**  
  Unified workflows via **Makefile** and **Poetry**.

---

## 🏗 Architecture & Design

The solution follows a modular and maintainable architecture:

1. **Data Processing**  
   Feature preprocessing (imputation, scaling) embedded in the training pipeline.
2. **Model Training**  
   Multiple algorithms trained under a unified interface.
3. **Evaluation**  
   Models evaluated on a hold-out test set.
4. **Model Registry**  
   Best model serialized (`.pkl`) and logged to MLflow.
5. **Serving Layer**  
   FastAPI loads the serialized pipeline for real-time inference.

### Design Choice — Cloud Agnostic

Managed ML platforms were intentionally avoided to demonstrate how to build a **portable**, **vendor-neutral** MLOps system that can run:
- On-premise
- In any cloud provider
- In local or CI/CD environments

---

## 🧰 Tech Stack

| Component | Tool | Description |
|---------|------|-------------|
| Language | Python 3.11+ | Core programming language |
| Package Manager | Poetry | Dependency management & virtual environments |
| ML Frameworks | Scikit-learn, XGBoost | Training pipelines & models |
| Experiment Tracking | MLflow | Metrics, artifacts, model lifecycle |
| API | FastAPI + Uvicorn | Async inference service |
| Containerization | Docker, Docker Compose | Deployment & isolation |
| Testing | Pytest | Unit & integration testing |
| Quality | Black, Isort, Ruff, Black | Formatting, linting, typing |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Docker Desktop** (running)
- **Poetry**

### Installation

1. **Clone the repository**
   ```bash
   git clone git@github.com:mhgualdron/challenge_mlops.git
   cd mlops-challenge

2.  **Install Dependencies**
    Using the Makefile (Recommended):
    ```bash
    make install
    ```
    Or manually via Poetry:
    ```bash
    poetry install
    ```


### 🛠 Makefile Commands

All common development and operational tasks are centralized via Makefile:


| Command | Description |
|---------|-------------|
make install	|Install dependencies with Poetry
make train	|Train and evaluate models
make serve	|Start FastAPI server locally
make test	|Run tests with coverage
make docker-build	|Build Docker image
make docker-up	|Start services with Docker Compose
make docker-down	|Stop Docker services
make mlflow	S|tart MLflow UI
make clean	|Remove caches and build artifacts
make lint	|Run Flake8 and Mypy
make format	|Auto-format code (Black + Isort)
make help | Help

---

## MLOps Pipeline

The training script (`src/models/train.py`) orchestrates the lifecycle of the model.

### 1. Run Training
Execute the pipeline to train models and select the champion:
```bash
make train
# or
poetry run python src/models/train.py
```

### 2. View Experiments
Launch the MLflow UI to visualize metrics (RMSE, MAE, R²) and compare runs:
```bash
make mlflow
```
*Access UI at: http://127.0.0.1:5000*

---

## 🐳 Deployment (Docker)

The application is production-ready with Docker Compose.

### Build and Run
```bash
docker compose up --build
```

The API will be available at **http://localhost:8000**.

---

## API Reference

### Endpoints

- **`GET /health`**: Health check. Returns status and model loading state.
- **`POST /predict`**: Inference endpoint.

### Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CRIM": 0.00632, "ZN": 18.0, "INDUS": 2.31, "CHAS": 0.0,
    "NOX": 0.538, "RM": 6.575, "AGE": 65.2, "DIS": 4.09,
    "RAD": 1.0, "TAX": 296.0, "PTRATIO": 15.3, "B": 396.9, "LSTAT": 4.98
  }'
```

### Example Response
```json
{
  "predicted_price": 24.5
}
```

---

## Testing & Quality Assurance

Unit tests cover API endpoints, payload validation, and model integrity.

Run the test suite:
```bash
make test
# or
poetry run pytest
```

---

## Project Structure

```text
mlops-challenge/
├── .github/workflows/   # CI/CD pipelines (Linting, Testing)
├── docker/              # Dockerfile and related configs
├── models/              # Storage for serialized champion models
├── src/
│   ├── api/             # FastAPI application logic
│   ├── models/          # Training and evaluation scripts
│   └── utils/           # Shared utility functions
├── tests/               # Pytest unit and integration tests
├── .env                 # Environment configuration
├── docker-compose.yml   # Orchestration for API and MLflow
├── Makefile             # Shortcuts for common tasks
├── poetry.lock          # Locked dependencies for reproducibility
├── pyproject.toml       # Project metadata and dependencies
└── README.md            # Project documentation
```

---

## 🤖 AI Tools Disclosure

This project leveraged Generative AI to accelerate development:

- **Boilerplate Generation**: Pydantic schemas and initial test scaffolding.
- **Code Quality Assistance**: Refactoring suggestions and style improvements.

**All code has been manually reviewed, tested, and validated by me**


## Future Improvements & Scalability
While this project serves as a robust functional baseline, the following enhancements are recommended for a *production-grade environment*:


1. **Data Version Control (DVC)**: Integrate DVC to track dataset versions alongside code, ensuring 100% reproducibility of experiments when data changes over time.


2. **Centralized Model Registry**: Transition from a local MLflow instance to a remote server (e.g., hosted on an EC2/GCE instance) to allow team collaboration and formal "Champion vs. Challenger" model management.


3. **Advanced Monitoring & Observability**: Implement a sidecar container with Prometheus and Grafana to track real-time inference latency, throughput, and data drift detection (using libraries like EvidentlyAI).


4. **Infrastructure as Code (IaC)**: Use Terraform or Ansible to automate the provisioning of the virtual machines where the Docker containers are deployed, maintaining the cloud-agnostic principle.


5. **API Security & Rate Limiting**: Add an OAuth2/API Key authentication layer and implement rate limiting to protect the inference endpoint from abuse in a public environment.


6. **Automated Retraining Trigger**: Configure a CI/CD trigger or a CronJob that initiates the train.py script automatically when performance metrics fall below a specific threshold.

# -----------

7. **Ecosystem Integration (Fury)**: In a real Mercado Libre environment, the next step would be migrating this agnostic containerized solution to Fury. This would allow taking advantage of Meli's internal metrics, standardized CI/CD pipelines, and high-availability infrastructure (AWS / GCP).