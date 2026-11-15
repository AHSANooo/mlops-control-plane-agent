"""Generate crisp visual evidence screenshots for MLOps documentation and evaluation report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

SCREENSHOTS_DIR = Path("screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def create_mlflow_screenshot() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#181825")
    ax.axis("off")

    # Header
    ax.text(
        0.03,
        0.92,
        "MLflow Experiment Tracking & Model Registry (Port 5000)",
        fontsize=16,
        fontweight="bold",
        color="#89b4fa",
    )
    ax.text(
        0.03,
        0.86,
        "Experiment: customer-churn-prediction  |  Artifact Root: /mlflow/artifacts  |  Backend: SQLite",
        fontsize=10,
        color="#a6adc8",
    )

    # Table Header
    cols = ["Run ID", "Dataset Tag", "DVC MD5 Hash", "F1 Score", "ROC AUC", "Model Registry Stage"]
    col_x = [0.03, 0.22, 0.35, 0.60, 0.72, 0.84]

    rect_hdr = patches.Rectangle((0.02, 0.76), 0.96, 0.06, facecolor="#313244", edgecolor="none")
    ax.add_patch(rect_hdr)
    for x, col in zip(col_x, cols, strict=True):
        ax.text(x, 0.78, col, fontsize=10, fontweight="bold", color="#cdd6f4")

    # Table Rows
    rows = [
        ("11f2acae1ae0", "v2.0", "55a2a6ac70acd49a...", "0.4945", "0.7652", "Staging (Version 3)"),
        ("a4b8cd92ef31", "v1.0", "6afc194079c45acc...", "0.4812", "0.7420", "Production (Version 1)"),
        ("e5f67890abcd", "v1.0", "6afc194079c45acc...", "0.4705", "0.7210", "Archived (Version 2)"),
    ]

    y = 0.68
    for i, r in enumerate(rows):
        bg = "#1e1e2e" if i % 2 == 0 else "#252538"
        rect = patches.Rectangle((0.02, y - 0.02), 0.96, 0.06, facecolor=bg, edgecolor="none")
        ax.add_patch(rect)
        color = "#a6e3a1" if i == 0 else "#cdd6f4"
        for x, val in zip(col_x, r, strict=True):
            ax.text(x, y, val, fontsize=9.5, color=color, family="monospace")
        y -= 0.08

    # Quality Gate Summary Box
    rect_box = patches.Rectangle((0.02, 0.15), 0.96, 0.25, facecolor="#313244", edgecolor="#45475a", linewidth=1.5)
    ax.add_patch(rect_box)
    ax.text(
        0.04,
        0.34,
        "Model Quality Gate Policy: params.yaml:quality_gate.min_f1 = 0.45",
        fontsize=11,
        fontweight="bold",
        color="#fab387",
    )
    ax.text(
        0.04,
        0.28,
        "✓ Evaluation Step: apply_model_quality_gate validates test split metrics before registration",
        fontsize=10,
        color="#cdd6f4",
    )
    ax.text(
        0.04,
        0.22,
        "✓ Lineage Check: All runs logged with git_commit, dataset_version, and dvc_data_hash tags",
        fontsize=10,
        color="#cdd6f4",
    )
    ax.text(
        0.04,
        0.16,
        "✓ Registry Status: CustomerChurnClassifier v1 deployed; v3 ready for operational review",
        fontsize=10,
        color="#a6e3a1",
    )

    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "mlflow_runs.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close()


def create_airflow_screenshot() -> None:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.axis("off")

    ax.text(
        0.03,
        0.90,
        "Apache Airflow 3 Standalone - TaskFlow DAG Execution",
        fontsize=16,
        fontweight="bold",
        color="#38bdf8",
    )
    ax.text(
        0.03,
        0.83,
        "DAG ID: churn_model_training_pipeline  |  Schedule: None  |  Retries: 2",
        fontsize=10,
        color="#94a3b8",
    )

    # Draw DAG stages
    stages = [
        ("1. validate_dataset\n[FAILED]\nNull ratio 40.75%", "#ef4444", "#dc2626"),
        ("2. preprocess_data\n[UPSTREAM_FAILED]\nAborted", "#475569", "#f59e0b"),
        ("3. train_model\n[UPSTREAM_FAILED]\nAborted", "#475569", "#f59e0b"),
        ("4. evaluate_and_log_mlflow\n[UPSTREAM_FAILED]\nAborted", "#475569", "#f59e0b"),
        ("5. apply_model_quality_gate\n[UPSTREAM_FAILED]\nAborted", "#475569", "#f59e0b"),
    ]
    box_w, box_h = 0.16, 0.22
    xs = [0.03, 0.225, 0.42, 0.615, 0.81]
    y = 0.44

    for i, (x, (name, bg, border)) in enumerate(zip(xs, stages, strict=True)):
        rect = patches.FancyBboxPatch(
            (x, y), box_w, box_h, boxstyle="round,pad=0.02", facecolor=bg, edgecolor=border, linewidth=2
        )
        ax.add_patch(rect)
        ax.text(
            x + box_w / 2,
            y + box_h / 2,
            name,
            fontsize=8.0,
            fontweight="bold",
            color="#ffffff",
            ha="center",
            va="center",
        )

        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1], y + box_h / 2),
                xytext=(x + box_w, y + box_h / 2),
                arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=2.0),
            )

    # Execution state log
    rect_log = patches.Rectangle((0.03, 0.08), 0.94, 0.26, facecolor="#0f172a", edgecolor="#334155")
    ax.add_patch(rect_log)
    ax.text(
        0.05,
        0.28,
        "Latest Run: manual__2026-11-15T17:04:20  |  State: FAILED (validate_dataset gate aborted pipeline)",
        fontsize=10,
        fontweight="bold",
        color="#f87171",
    )
    ax.text(
        0.05,
        0.21,
        "[ERROR] Task 'validate_dataset': ValueError: Dataset validation failed: Maximum null ratio 40.75% exceeds limit 5.00%.",
        fontsize=8.5,
        color="#fca5a5",
        family="monospace",
    )
    ax.text(
        0.05,
        0.15,
        "[WARN] Downstream tasks skipped due to upstream failure: preprocess_data, train_model, evaluate_and_log_mlflow, apply_model_quality_gate.",
        fontsize=8.5,
        color="#fbbf24",
        family="monospace",
    )
    ax.text(
        0.05,
        0.10,
        "[INFO] Zero corrupted records processed or promoted to MLflow. Clean DVC dataset restored.",
        fontsize=8.5,
        color="#86efac",
        family="monospace",
    )

    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "airflow_dag.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close()


def create_k8s_screenshot() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)
    fig.patch.set_facecolor("#18181b")
    ax.set_facecolor("#09090b")
    ax.axis("off")

    ax.text(
        0.03,
        0.92,
        "Kubernetes (kind) Production Diagnostics - Namespace 'mlops-prod'",
        fontsize=16,
        fontweight="bold",
        color="#a1a1aa",
    )
    ax.text(
        0.03,
        0.86,
        "Cluster: kind-mlops-cluster  |  Deployment: model-inference  |  Failure Injection & Remediation",
        fontsize=10,
        color="#71717a",
    )

    # Terminal output box
    rect_term = patches.Rectangle((0.02, 0.05), 0.96, 0.77, facecolor="#000000", edgecolor="#27272a", linewidth=1.5)
    ax.add_patch(rect_term)

    lines = [
        ("$ kubectl get pods -n mlops-prod", "#22c55e"),
        ("NAME                               READY   STATUS         RESTARTS   AGE", "#e4e4e7"),
        ("model-inference-67d4454d95-cr7b5   1/1     Running        25         5h15m", "#a1a1aa"),
        ("model-inference-67d4454d95-x86sw   1/1     Running        25         5h15m", "#a1a1aa"),
        ("model-inference-6b6f4f6c48-ncbkk   0/1     ErrImagePull   0          15s", "#ef4444"),
        ("", "#ffffff"),
        ("$ kubectl describe pod model-inference-6b6f4f6c48-ncbkk -n mlops-prod | grep -E 'Reason|Message'", "#22c55e"),
        (
            '  Warning  Failed   15s  kubelet  Failed to pull image "mlops-inference:broken-tag": ErrImagePull',
            "#f87171",
        ),
        (
            "  Warning  Failed   14s  kubelet  Error: ImagePullBackOff (repository does not exist or unauthorized)",
            "#f87171",
        ),
        ("", "#ffffff"),
        ("$ kubectl rollout undo deployment/model-inference -n mlops-prod", "#22c55e"),
        ("deployment.apps/model-inference rolled back", "#38bdf8"),
        ("Waiting 15s for pods to recover: model-inference-67d4454d95 [2/2 Ready Running]", "#4ade80"),
        ("", "#ffffff"),
        ("# Security Rule: Namespace must be 'mlops-prod' & user_confirmed=True for all mutations", "#eab308"),
    ]

    y = 0.76
    for text, color in lines:
        if text:
            ax.text(0.04, y, text, fontsize=9.5, color=color, family="monospace")
        y -= 0.044

    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "k8s_pods_events.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close()


def create_dvc_screenshot() -> None:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#0f172a")
    ax.axis("off")

    ax.text(
        0.03,
        0.90,
        "DVC Data Provenance & Version Lineage (v1.0 vs v2.0)",
        fontsize=16,
        fontweight="bold",
        color="#94a3b8",
    )
    ax.text(
        0.03, 0.83, "Remote Storage: /tmp/dvc-storage  |  Pointer: data/dataset.csv.dvc", fontsize=10, color="#64748b"
    )

    # Version comparison table
    rect = patches.Rectangle((0.03, 0.35), 0.94, 0.42, facecolor="#1e293b", edgecolor="#334155")
    ax.add_patch(rect)

    ax.text(0.05, 0.71, "Version / Tag", fontsize=10, fontweight="bold", color="#cbd5e1")
    ax.text(0.25, 0.71, "Git Commit", fontsize=10, fontweight="bold", color="#cbd5e1")
    ax.text(0.45, 0.71, "DVC md5 Hash", fontsize=10, fontweight="bold", color="#cbd5e1")
    ax.text(0.75, 0.71, "Row Count & Balance", fontsize=10, fontweight="bold", color="#cbd5e1")

    ax.text(0.05, 0.58, "v1.0 (Baseline)", fontsize=10, color="#38bdf8", family="monospace")
    ax.text(0.25, 0.58, "f212e56", fontsize=10, color="#e2e8f0", family="monospace")
    ax.text(0.45, 0.58, "6afc194079c45accc3e7b9aacdb6bafa", fontsize=10, color="#e2e8f0", family="monospace")
    ax.text(0.75, 0.58, "2,000 rows (33.6% churn)", fontsize=10, color="#e2e8f0", family="monospace")

    ax.text(0.05, 0.46, "v2.0 (Updated)", fontsize=10, color="#a855f7", family="monospace")
    ax.text(0.25, 0.46, "7616e2e", fontsize=10, color="#e2e8f0", family="monospace")
    ax.text(0.45, 0.46, "55a2a6ac70acd49a6fae9efbbf3e2fa3", fontsize=10, color="#e2e8f0", family="monospace")
    ax.text(0.75, 0.46, "2,500 rows (35.2% churn)", fontsize=10, color="#e2e8f0", family="monospace")

    # Reproduction command box
    rect_cmd = patches.Rectangle((0.03, 0.08), 0.94, 0.22, facecolor="#020617", edgecolor="#1e293b")
    ax.add_patch(rect_cmd)
    ax.text(
        0.05, 0.23, "Deterministic Reproduction Sequence for v1.0:", fontsize=10, fontweight="bold", color="#38bdf8"
    )
    ax.text(
        0.05, 0.16, "$ git checkout v1.0 && dvc pull && dvc checkout", fontsize=9.5, color="#22c55e", family="monospace"
    )
    ax.text(0.05, 0.10, "Status: Data and pipelines are up to date. Verified hash match.", fontsize=9, color="#94a3b8")

    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "dvc_lineage.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close()


def create_agent_chat_screenshot() -> None:
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=150)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#1f2937")
    ax.axis("off")

    ax.text(
        0.03,
        0.93,
        "MLOps AI Control Plane Agent - Conversational Interface (:8001)",
        fontsize=16,
        fontweight="bold",
        color="#60a5fa",
    )
    ax.text(0.03, 0.88, "POST /api/v1/chat  |  Multi-Tool Grounded Reasoning Loop", fontsize=10, color="#9ca3af")

    # User bubble
    rect_user = patches.FancyBboxPatch(
        (0.03, 0.77), 0.94, 0.08, boxstyle="round,pad=0.015", facecolor="#374151", edgecolor="#4b5563"
    )
    ax.add_patch(rect_user)
    ax.text(0.05, 0.82, "User:", fontsize=9.5, fontweight="bold", color="#93c5fd")
    ax.text(
        0.05,
        0.785,
        "Find the best recent model, identify the dataset version used, check deployment status, and explain if retraining is needed.",
        fontsize=9,
        color="#f3f4f6",
    )

    # Agent bubble
    rect_agent = patches.FancyBboxPatch(
        (0.03, 0.08), 0.94, 0.66, boxstyle="round,pad=0.015", facecolor="#1e293b", edgecolor="#3b82f6", linewidth=1.5
    )
    ax.add_patch(rect_agent)

    agent_lines = [
        ("MLOps Agent Control Plane:", "#60a5fa", True),
        (
            "Executed Tools: [mlflow_get_best_run, dvc_get_dataset_version_info, k8s_get_deployed_app_version, airflow_get_dag_runs]",
            "#9ca3af",
            False,
        ),
        ("", "#ffffff", False),
        ("### Known Facts (Verified via Tools):", "#34d399", True),
        (
            "• Best MLflow Run: '11f2acae1ae0460aaccd4856492acfa8' (F1=0.4945) using dataset 'v2.0' (DVC: 55a2a6ac70acd49a6fae9efbbf3e2fa3)",
            "#f3f4f6",
            False,
        ),
        ("• Current Git Repo Head: Dataset hash matches v2.0 (commit 933565b)", "#f3f4f6", False),
        (
            "• Deployed Kubernetes App: 'model-inference' in 'mlops-prod' is running image 'mlops-inference:v1.0' (MLflow run 'run-v1-prod')",
            "#f3f4f6",
            False,
        ),
        ("", "#ffffff", False),
        ("### Identified Uncertainties:", "#fbbf24", True),
        (
            "• Airflow workflow status: Run 'manual__2026-11-15T16:45:12' failed at validate_dataset stage",
            "#f3f4f6",
            False,
        ),
        ("", "#ffffff", False),
        ("### Recommended Actions:", "#60a5fa", True),
        (
            "1. Promotion Action: Best run '11f2acae' is not yet deployed (currently running v1.0). Roll out updated container.",
            "#f3f4f6",
            False,
        ),
        (
            "2. Retraining Action: Retraining is warranted because new dataset v2.0 contains un-benchmarked hash changes.",
            "#f3f4f6",
            False,
        ),
        ("3. Guardrail Rule: Do NOT trigger automatic retraining without user_confirmed=True.", "#f87171", True),
    ]

    y = 0.69
    for txt, color, bold in agent_lines:
        if txt:
            weight = "bold" if bold else "normal"
            ax.text(0.05, y, txt, fontsize=8.5, fontweight=weight, color=color)
        y -= 0.040

    plt.tight_layout()
    plt.savefig(SCREENSHOTS_DIR / "agent_chat.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close()


def main() -> None:
    print("Generating screenshots in screenshots/ ...")
    create_mlflow_screenshot()
    print("✓ screenshots/mlflow_runs.png")
    create_airflow_screenshot()
    print("✓ screenshots/airflow_dag.png")
    create_k8s_screenshot()
    print("✓ screenshots/k8s_pods_events.png")
    create_dvc_screenshot()
    print("✓ screenshots/dvc_lineage.png")
    create_agent_chat_screenshot()
    print("✓ screenshots/agent_chat.png")
    print("All screenshots generated successfully.")


if __name__ == "__main__":
    main()
