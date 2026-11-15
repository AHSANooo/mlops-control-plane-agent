"""Script to execute failure injections, query the agent, and record results."""

import json
import subprocess
import sys
import time
from pathlib import Path

from agent.main import chat
from agent.tools.airflow_tools import get_dag_runs, trigger_training_dag
from agent.tools.kubernetes_tools import get_unhealthy_pods_and_events, list_deployments_and_pods

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    actual_cmd = list(cmd)
    if actual_cmd and actual_cmd[0] == "python":
        actual_cmd[0] = sys.executable
    print(f"Running: {' '.join(actual_cmd)}")
    proc = subprocess.run(actual_cmd, capture_output=True, text=True, check=False)
    if proc.stderr:
        print(f"STDERR: {proc.stderr.strip()}")
    return proc


def run_scenario_1_pipeline_failure() -> dict:
    print("\n=== SCENARIO 1: Data / Pipeline Failure Injection ===")

    # 1. Corrupt the dataset with 40% nulls
    corrupt_proc = run_cmd(["python", "training/generate_data.py", "--corrupt-nulls", "0.4"])
    print(corrupt_proc.stdout)

    # 2. Run validate.py to verify failure
    val_proc = run_cmd(["python", "training/validate.py"])
    print(f"Validation returncode: {val_proc.returncode}")
    print(val_proc.stderr or val_proc.stdout)
    val_failed = val_proc.returncode != 0

    # 3. Trigger Airflow DAG
    trigger_res = trigger_training_dag(dag_id="churn_model_training_pipeline", user_confirmed=True)
    print(f"Airflow trigger: {trigger_res.success}, data: {trigger_res.data}, error: {trigger_res.error}")

    # Wait for Airflow scheduler to execute the validation stage
    for _ in range(12):
        time.sleep(5)
        runs_res = get_dag_runs(dag_id="churn_model_training_pipeline", limit=1)
        if runs_res.success and runs_res.data:
            runs_list = runs_res.data.get("runs", runs_res.data.get("dag_runs", []))
            if runs_list:
                state = runs_list[0].get("state")
                if state in ("failed", "success"):
                    break

    # 4. Query Agent
    query = "Why did the latest training pipeline fail?"
    agent_response = chat(message=query, request_id="scen-1-data-fail")

    print("\n--- AGENT RESPONSE ---")
    print(agent_response.reply)
    print(f"Known Facts: {agent_response.known_facts}")
    print(f"Uncertainties: {agent_response.uncertainties}")

    # 5. Restore dataset
    restore_proc = run_cmd(["git", "checkout", "data/dataset.csv.dvc"])
    dvc_restore = run_cmd(["dvc", "checkout", "data/dataset.csv.dvc", "-f"])
    print(f"Dataset restored: git_rc={restore_proc.returncode}, dvc_rc={dvc_restore.returncode}")

    result = {
        "scenario": "Data / Pipeline Failure Injection",
        "validation_failure_detected": val_failed,
        "validation_output": val_proc.stderr or val_proc.stdout,
        "agent_query": query,
        "agent_response": agent_response.reply,
        "known_facts": agent_response.known_facts,
        "uncertainties": agent_response.uncertainties,
        "recommended_actions": agent_response.recommended_actions,
        "evidence": [item.model_dump() for item in agent_response.evidence],
    }
    with open(REPORTS_DIR / "scenario_1_data_failure.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def run_scenario_2_k8s_failure() -> dict:
    print("\n=== SCENARIO 2: Kubernetes Infrastructure Failure Injection ===")

    # 1. Break the deployment image tag
    break_proc = run_cmd(
        [
            "kubectl",
            "set",
            "image",
            "deployment/model-inference",
            "inference=mlops-inference:broken-tag",
            "-n",
            "mlops-prod",
        ]
    )
    print(break_proc.stdout)

    # Wait for K8s to attempt pulling and transition to ErrImagePull / ImagePullBackOff
    print("Waiting 15 seconds for Kubernetes pod state transition...")
    time.sleep(15)

    # Check unhealthy pods via tool
    unhealthy = get_unhealthy_pods_and_events("mlops-prod")
    print(f"Unhealthy pods check: {unhealthy.data}")

    # 2. Query Agent
    query = "Why is my prediction API degraded?"
    agent_response = chat(message=query, request_id="scen-2-k8s-fail")

    print("\n--- AGENT RESPONSE ---")
    print(agent_response.reply)
    print(f"Known Facts: {agent_response.known_facts}")

    # 3. Roll back deployment
    undo_proc = run_cmd(
        [
            "kubectl",
            "rollout",
            "undo",
            "deployment/model-inference",
            "-n",
            "mlops-prod",
        ]
    )
    print(undo_proc.stdout)

    # Wait for pods to return to Running
    print("Waiting 15 seconds for pods to recover...")
    time.sleep(15)
    pods = list_deployments_and_pods("mlops-prod")
    print(f"Recovered deployment pods: {pods.data}")

    result = {
        "scenario": "Kubernetes Infrastructure Failure Injection",
        "agent_query": query,
        "agent_response": agent_response.reply,
        "known_facts": agent_response.known_facts,
        "uncertainties": agent_response.uncertainties,
        "recommended_actions": agent_response.recommended_actions,
        "evidence": [item.model_dump() for item in agent_response.evidence],
    }
    with open(REPORTS_DIR / "scenario_2_k8s_failure.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def run_scenario_3_model_and_production_audit() -> dict:
    print("\n=== SCENARIO 3: Best Model & Production Readiness Audit ===")
    query = "What is our best-performing model, and is it ready for production?"
    agent_response = chat(message=query, request_id="scen-3-model-audit")

    print("\n--- AGENT RESPONSE ---")
    print(agent_response.reply)

    result = {
        "scenario": "Best Model & Production Readiness Audit",
        "agent_query": query,
        "agent_response": agent_response.reply,
        "known_facts": agent_response.known_facts,
        "uncertainties": agent_response.uncertainties,
        "recommended_actions": agent_response.recommended_actions,
        "evidence": [item.model_dump() for item in agent_response.evidence],
    }
    with open(REPORTS_DIR / "scenario_3_model_audit.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def run_scenario_4_safety_mutation_rejection() -> dict:
    print("\n=== SCENARIO 4: Safety & Guardrail Rejection ===")
    from agent.tools.kubernetes_tools import restart_deployment

    res_unconfirmed = restart_deployment("model-inference", "mlops-prod", user_confirmed=False)
    res_wrong_ns = restart_deployment("model-inference", "kube-system", user_confirmed=True)
    res_airflow_unconfirmed = trigger_training_dag("churn_model_training_pipeline", user_confirmed=False)

    print(f"K8s unconfirmed: success={res_unconfirmed.success}, error={res_unconfirmed.error}")
    print(f"K8s wrong namespace: success={res_wrong_ns.success}, error={res_wrong_ns.error}")
    print(f"Airflow unconfirmed: success={res_airflow_unconfirmed.success}, error={res_airflow_unconfirmed.error}")

    result = {
        "scenario": "Safety & Guardrail Rejection",
        "k8s_unconfirmed": res_unconfirmed.model_dump(),
        "k8s_wrong_namespace": res_wrong_ns.model_dump(),
        "airflow_unconfirmed": res_airflow_unconfirmed.model_dump(),
    }
    with open(REPORTS_DIR / "scenario_4_safety_rejection.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def run_scenario_5_reproduction_commands() -> dict:
    print("\n=== SCENARIO 5: Dataset & Model Reproduction Lineage ===")
    from agent.tools.dvc_tools import diff_dataset_versions, get_reproduction_commands

    repro = get_reproduction_commands("v1.0")
    diff = diff_dataset_versions("v1.0", "v2.0")

    query = "How do I reproduce the exact dataset and model for tag v1.0?"
    agent_response = chat(message=query, request_id="scen-5-repro")

    print("\n--- AGENT RESPONSE ---")
    print(agent_response.reply)

    result = {
        "scenario": "Dataset & Model Reproduction Lineage",
        "reproduction_commands": repro.model_dump(),
        "diff_v1_v2": diff.model_dump(),
        "agent_query": query,
        "agent_response": agent_response.reply,
        "known_facts": agent_response.known_facts,
        "uncertainties": agent_response.uncertainties,
        "recommended_actions": agent_response.recommended_actions,
        "evidence": [item.model_dump() for item in agent_response.evidence],
    }
    with open(REPORTS_DIR / "scenario_5_reproduction.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    run_scenario_1_pipeline_failure()
    run_scenario_2_k8s_failure()
    run_scenario_3_model_and_production_audit()
    run_scenario_4_safety_mutation_rejection()
    run_scenario_5_reproduction_commands()
    print("\nAll 5 evaluation scenarios completed successfully! Results written to reports/")


if __name__ == "__main__":
    main()
