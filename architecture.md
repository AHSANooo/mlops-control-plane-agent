# MLOps AI Agent System Architecture

## 1. High-Level System Architecture

The following diagram illustrates the interaction between the User/Client, the FastAPI Gateway, the Multi-Tool LLM Orchestrator, and the four integrated infrastructure backends: **DVC**, **MLflow**, **Apache Airflow 3**, and **Kubernetes (`kind`)**.

```mermaid
flowchart TD
    Client["Client / Engineer (HTTP / Chat API)"] --> Gateway["FastAPI Gateway (/api/v1/chat, /health)"]
    Gateway --> Orchestrator["MLOps Agent Orchestrator"]

    subgraph Security_And_Telemetry["DevOps Guardrails & Observability"]
        Guardrails{"Namespace & Confirmation Guardrails"}
        TelemetryLogger["JSON Structured Telemetry & Secret Masker"]
    end

    Orchestrator --> Guardrails
    Orchestrator --> TelemetryLogger

    subgraph Tool_Modules["Agent Tool Modules (Shell=False, Pydantic V2)"]
        DVC["DVC Tools (Lineage, Hash Diff, Reproduction)"]
        MLFLOW["MLflow Tools (Runs, Metrics, Registry Status)"]
        AIRFLOW["Airflow Tools (DAGs, Runs, Task Logs, Triggers)"]
        K8S["Kubernetes Tools (Pods, Deployments, Diagnostics, Rollback)"]
    end

    Guardrails -->|Allowed & Confirmed| Tool_Modules

    subgraph Infrastructure_Layer["Local MLOps Infrastructure"]
        DVCStorage[("DVC Storage (/tmp/dvc-storage & Git Tags)")]
        MLflowServer["MLflow Tracking Server (:5000)"]
        AirflowServer["Apache Airflow 3 Standalone (:8080)"]
        KindCluster["Kubernetes kind Cluster (Namespace: mlops-prod)"]
    end

    DVC --> DVCStorage
    MLFLOW --> MLflowServer
    AIRFLOW --> AirflowServer
    K8S --> KindCluster
```

---

## 2. Airflow 3 5-Stage Training Pipeline Architecture

The model training pipeline is implemented using the Apache Airflow 3 TaskFlow API. It guarantees data quality, pre-processing consistency, MLflow lineage tracking, and automated quality gate enforcement before registering models in the MLflow Model Registry.

```mermaid
flowchart LR
    Start([Pipeline Trigger]) --> S1["1. validate_dataset"]
    S1 -->|Null ratio <= 5% & Schema Valid| S2["2. preprocess_data"]
    S1 -.->|Null ratio > 5%| F1["Pipeline Aborted (ValueError)"]
    
    S2 -->|Clean Parquet Splits| S3["3. train_model"]
    
    S3 -->|Sklearn Model & Hyperparams| S4["4. evaluate_and_log_mlflow"]
    S4 -->|Logs Metrics, Tags & Cloudpickle Artifacts| S5["5. apply_model_quality_gate"]
    
    S5 -->|F1 >= 0.45 & ROC-AUC >= 0.70| Reg["Register in MLflow Model Registry"]
    S5 -.->|F1 < 0.45| F2["Quality Gate Rejection (Model Blocked)"]
    Reg --> End([Complete])
```

### Pipeline Stage Details:
1. **`validate_dataset`**: Inspects `data/dataset.csv`. Verifies all required columns exist (`customer_id`, `tenure_months`, `monthly_charges`, `total_charges`, `contract_type`, etc.) and ensures missing value ratio is $\le 5\%$.
2. **`preprocess_data`**: Implements stratified 80/20 train/test splitting, one-hot encodes categorical features, scales numeric values, and stores clean partitions into `data/train.parquet` and `data/test.parquet`.
3. **`train_model`**: Trains a Scikit-Learn `RandomForestClassifier` parameterized via `params.yaml`.
4. **`evaluate_and_log_mlflow`**: Computes evaluation metrics ($F_1$, precision, recall, ROC-AUC) and logs Git commit hash, dataset Git tag, and DVC MD5 data hash as searchable MLflow tags.
5. **`apply_model_quality_gate`**: Compares candidate model $F_1$ score against `params.yaml:quality_gate.min_f1`. Only approved models are promoted to `CustomerChurnClassifier` in the MLflow Model Registry.

---

## 3. Grounded Multi-Tool Reasoning & Guardrail Execution Sequence

The agent enforces zero-hallucination policies and strict state mutation safeguards. State-mutating commands (e.g., triggering workflows or restarting Kubernetes deployments) are rejected if `user_confirmed` is not `True` or if targeting outside `mlops-prod`.

```mermaid
sequenceDiagram
    autonumber
    actor User as User / DevOps Engineer
    participant API as FastAPI Gateway
    participant Orch as Agent Orchestrator
    participant Guard as Safety & Redaction Guardrail
    participant Tools as MLOps Tools (DVC/MLflow/K8s/Airflow)
    participant Stack as Live Infrastructure Stack

    User->>API: POST /api/v1/chat {"message": "Why is my prediction API degraded?"}
    API->>Orch: plan_and_execute_grounded(request)
    Orch->>Guard: Validate Namespace ("mlops-prod") & Mutation Rules
    Guard-->>Orch: Validation Passed (Read-only Diagnostic)
    
    Orch->>Tools: k8s_get_deployed_app_version("mlops-prod")
    Tools->>Stack: Query ConfigMap & Deployment Spec
    Stack-->>Tools: Model version v1.0, Run run-v1-prod
    Tools-->>Orch: ToolResult(success=True)

    Orch->>Tools: k8s_get_unhealthy_pods_and_events("mlops-prod")
    Tools->>Stack: List Pods & Warning Events
    Stack-->>Tools: Pod in ErrImagePull (broken-tag)
    Tools-->>Orch: ToolResult(success=True, unhealthy_pods=[...])

    Orch->>Orch: Synthesize Evidence into Known Facts, Uncertainties & Actions
    Orch->>Guard: Log JSON Structured Telemetry (Secret Redacted)
    Orch-->>API: AgentChatResponse(reply, known_facts, uncertainties, recommended_actions)
    API-->>User: 200 OK Response with Grounded Diagnostics & Rollback Command
```

---

## 4. Security & Safety Model

1. **Subprocess Isolation**: Zero instances of `shell=True`. All shell binaries (`git`, `dvc`, `kubectl`) are invoked via explicit, sanitized argument lists.
2. **Namespace Guardrail**: Only the `mlops-prod` namespace is accessible for write operations. Any request targeting `kube-system`, `default`, or other namespaces fails immediately with a security rejection.
3. **Double Confirmation**: Workflows mutating state (`restart_deployment`, `trigger_training_dag`) require `user_confirmed=True`.
4. **Data Redaction**: Telemetry engine uses regex patterns to filter Bearer JWT tokens, API keys (`sk-...`), and passwords before emitting JSON logs.
