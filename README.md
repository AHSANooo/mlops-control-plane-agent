# Production-Ready MLOps AI Agent

[![MLOps AI Agent CI](https://github.com/AHSANooo/assignment-03-mlops-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/AHSANooo/assignment-03-mlops-agent/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, multi-tool conversational AI assistant that interfaces directly with production MLOps infrastructure: **DVC** for data lineage, **MLflow** for experiment tracking and model registration, **Apache Airflow 3** for training workflow orchestration, and **Kubernetes (`kind`)** for live microservice diagnostics and remediation.

Complete architectural details and flowcharts are documented in [architecture.md](file:///d:/Uni/MLOps/Assignment03/architecture.md).  
Detailed empirical evaluation and failure injection logs are documented in [evaluation_report.md](file:///d:/Uni/MLOps/Assignment03/reports/evaluation_report.md).

---

## Key Capabilities & DevOps Guardrails

1. **Strict Type Safety & Schemas**:
   - 100% type-annotated codebase enforced with `mypy` strict mode on `agent/`.
   - All tool inputs and outputs validate against Pydantic V2 schemas.
2. **Subprocess Isolation**:
   - Zero usage of `shell=True` across the codebase. All subprocesses invoke explicit argument lists.
3. **Structured Tool Returns & Error Resilience**:
   - Every tool returns a standardized `ToolResult(success, tool_name, data, error, duration_ms)`.
   - Exceptions are caught internally and returned safely; the LLM never receives unhandled raw tracebacks.
4. **State Mutation Safeguards**:
   - Kubernetes cluster mutations (rollout restarts, image rollbacks) strictly require:
     - Target `namespace == "mlops-prod"`.
     - Explicit user confirmation (`user_confirmed is True`).
   - Mutations targeting forbidden namespaces (such as `kube-system` or `default`) or without confirmation immediately fail with security rejections.
   - Apache Airflow DAG triggers also strictly require `user_confirmed is True`.
5. **Grounded Multi-Tool Reasoning (Zero Hallucination)**:
   - The agent strictly differentiates between **Known Facts** (retrieved directly from live tools), **Uncertainties** (missing, unavailable, or ambiguous telemetry), and **Recommended Actions**.
   - Highest $F_1$ score alone never implies production readiness without verifying DVC dataset lineage and Kubernetes deployment state.
6. **Telemetry & Secret Redaction**:
   - Every invocation generates structured JSON telemetry logging `request_id`, `tool_name`, `status`, and `duration_ms`.
   - Secret redaction automatically masks Bearer JWT tokens, API keys, and passwords before logging.

---

## Repository Structure

```text
.
├── .github/workflows/
│   └── ci.yml                     # GitHub Actions CI (Ruff, Mypy, Pytest, Docker)
├── agent/                         # Core MLOps AI Agent Package
│   ├── config.py                  # Pydantic BaseSettings (URIs, allowlists, tokens)
│   ├── main.py                    # Programmatic agent entrypoint (`chat()`)
│   ├── orchestrator.py            # Grounded multi-tool reasoning orchestrator
│   ├── schemas.py                 # Pydantic V2 schemas (ToolResult, AgentChatResponse)
│   ├── telemetry.py               # JSON structured logger with regex secret masking
│   └── tools/                     # Four modular toolkits
│       ├── airflow_tools.py       # Airflow 3 REST API v2 & token auth tools
│       ├── dvc_tools.py           # DVC dataset versioning & lineage tools
│       ├── kubernetes_tools.py    # Official Kubernetes client tools & diagnostics
│       └── mlflow_tools.py        # MLflow Client experiment & registry tools
├── api/
│   └── main.py                    # FastAPI Gateway (`POST /api/v1/chat`, `GET /health`)
├── dags/
│   └── training_pipeline.py       # 5-stage Airflow 3 TaskFlow training DAG
├── docker/
│   └── mlflow/                    # MLflow 3.18.0 server Dockerfile
├── docker-compose.yml             # Local MLflow & Airflow 3 stack
├── inference_service/             # Containerized FastAPI customer churn inference API
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── kubernetes/                    # Kubernetes manifests for kind cluster
│   ├── configmap.yaml             # Model version & MLflow run reference
│   ├── deployment.yaml            # 2-replica deployment with liveness/readiness probes
│   ├── namespace.yaml             # mlops-prod isolated namespace
│   └── service.yaml               # ClusterIP service
├── params.yaml                    # Hyperparameter & quality gate configuration
├── pyproject.toml                 # Dependencies, Ruff, Mypy, and Pytest configuration
├── reports/                       # Evaluation scenario reports and JSON telemetry
│   ├── evaluation_report.md
│   ├── scenario_1_data_failure.json
│   ├── scenario_2_k8s_failure.json
│   ├── scenario_3_model_audit.json
│   ├── scenario_4_safety_rejection.json
│   └── scenario_5_reproduction.json
├── scripts/
│   └── run_failure_injections.py  # Automated 5-scenario evaluation runner
├── tests/                         # Complete Pytest test suite (21 unit & integration tests)
└── training/                      # Training & validation scripts
    ├── generate_data.py           # Synthetic dataset generator with null-injection flag
    ├── preprocess.py              # Data preprocessing and parquet partitioning
    ├── train.py                   # Model training, MLflow logging & quality gate
    └── validate.py                # Schema & missing value validation gate
```

---

## Local Setup & Quickstart

### 1. Prerequisites (WSL2 / Linux)
- Windows with WSL2 Ubuntu 24.04 (or Linux host).
- Docker Engine & Docker Compose.
- `uv` Python package manager:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- `kubectl` and `kind` (Kubernetes in Docker):
  ```bash
  # Install kind
  curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.24.0/kind-linux-amd64
  chmod +x ./kind && sudo mv ./kind /usr/local/bin/kind

  # Install kubectl
  curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
  chmod +x kubectl && sudo mv kubectl /usr/local/bin/kubectl
  ```

### 2. Environment Setup
Create the virtual environment and install all packages in editable mode:
```bash
uv venv ~/.venvs/mlops-agent
source ~/.venvs/mlops-agent/bin/activate
uv pip install -e ".[dev]"
```

### 3. Start the MLOps Stack
Spin up the MLflow tracking server and Apache Airflow 3 standalone:
```bash
docker compose up -d
```
- **MLflow Tracking UI**: [http://localhost:5000](http://localhost:5000)
- **Airflow 3 UI**: [http://localhost:8080](http://localhost:8080) (Password available in `/opt/airflow/simple_auth_manager_passwords.json.generated`).

### 4. Deploy Kubernetes Inference Cluster
Create the local `kind` cluster and deploy the microservice:
```bash
kind create cluster --name mlops-cluster
docker build -t mlops-inference:v1.0 ./inference_service
kind load docker-image mlops-inference:v1.0 --name mlops-cluster

kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/
kubectl get pods -n mlops-prod -w
```

---

## DVC Dataset Versioning

The churn dataset is tracked deterministically via DVC pointers:
- **Version 1.0**: Tagged `v1.0` (commit `80f37db`, hash `6afc194079c45accc3e7b9aacdb6bafa`).
- **Version 2.0**: Tagged `v2.0` (commit `e83c7b2`, hash `55a2a6ac70acd49a6fae9efbbf3e2fa3`).

To reproduce any historical dataset state:
```bash
git checkout v1.0
dvc checkout
```

---

## Running the Agent API

Start the FastAPI conversational gateway:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
```
Swagger UI documentation is available at [http://localhost:8001/docs](http://localhost:8001/docs).

### Example Conversational Request
```bash
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is our best-performing model, and is it ready for production?"}'
```

---

## Verification & Testing

### 1. Execute the 5 Failure Injection Scenarios
Run the automated end-to-end evaluation runner:
```bash
python scripts/run_failure_injections.py
```
This tests:
1. **Pipeline Null Corruption**: Injects 40% missing values; verifies validation failure and grounded root cause diagnosis.
2. **Kubernetes Image Fault**: Injects `broken-tag`; detects `ErrImagePull` and verifies deployment rollback.
3. **Production Readiness Audit**: Correlates MLflow best $F_1$ runs with DVC lineage and K8s deployed version.
4. **DevOps Guardrail Rejection**: Validates rejection of unconfirmed mutations and unauthorized namespaces.
5. **DVC Reproduction Lineage**: Verifies generation of exact deterministic reproduction commands.

### 2. Run Test Suite & Static Analysis
```bash
# Static Analysis
ruff check .
ruff format --check .
mypy agent

# Pytest Suite (21 tests)
pytest -v
```

---

## Continuous Integration (CI)

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every pull request and push to `main` and `feature/**` branches, executing:
- **Linting & Code Quality**: `ruff check .`
- **Formatting**: `ruff format --check .`
- **Strict Typing**: `mypy agent`
- **Automated Tests**: `pytest -v`
- **Container Build**: `docker build -t mlops-inference:test ./inference_service`
