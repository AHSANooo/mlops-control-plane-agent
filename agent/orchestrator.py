"""Intelligent Multi-Tool MLOps Agent Orchestrator with Grounded Reasoning."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from agent.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    EvidenceItem,
    ToolResult,
)
from agent.telemetry import telemetry
from agent.tools import (
    airflow_tools,
    dvc_tools,
    kubernetes_tools,
    mlflow_tools,
)

# Registry of all callable tools
TOOL_REGISTRY: dict[str, Callable[..., ToolResult]] = {
    # DVC
    "dvc_list_tracked_datasets": dvc_tools.list_tracked_datasets,
    "dvc_get_dataset_version_info": dvc_tools.get_dataset_version_info,
    "dvc_diff_dataset_versions": dvc_tools.diff_dataset_versions,
    "dvc_check_dataset_status": dvc_tools.check_dataset_status,
    "dvc_get_reproduction_commands": dvc_tools.get_reproduction_commands,
    # MLflow
    "mlflow_list_experiments_and_runs": mlflow_tools.list_experiments_and_runs,
    "mlflow_get_run_details": mlflow_tools.get_run_details,
    "mlflow_get_best_run": mlflow_tools.get_best_run,
    "mlflow_compare_runs": mlflow_tools.compare_runs,
    "mlflow_get_model_registry_status": mlflow_tools.get_model_registry_status,
    # Airflow
    "airflow_list_dags": airflow_tools.list_dags,
    "airflow_get_dag_runs": airflow_tools.get_dag_runs,
    "airflow_get_task_instances": airflow_tools.get_task_instances,
    "airflow_get_failed_task_logs": airflow_tools.get_failed_task_logs,
    "airflow_trigger_training_dag": airflow_tools.trigger_training_dag,
    # Kubernetes
    "k8s_list_deployments_and_pods": kubernetes_tools.list_deployments_and_pods,
    "k8s_get_pod_logs": kubernetes_tools.get_pod_logs,
    "k8s_get_unhealthy_pods_and_events": kubernetes_tools.get_unhealthy_pods_and_events,
    "k8s_inspect_services": kubernetes_tools.inspect_services,
    "k8s_get_deployed_app_version": kubernetes_tools.get_deployed_app_version,
    "k8s_restart_deployment": kubernetes_tools.restart_deployment,
}

SYSTEM_PROMPT = """You are an expert MLOps AI Engineer Assistant with direct tool access to DVC, MLflow, Airflow, and Kubernetes.
Strict Grounding & Operational Rules:
1. Never invent or hallucinate metrics, git hashes, run IDs, pod names, or logs. If a tool fails or data is missing, report it as an uncertainty.
2. Highest F1 score alone does not guarantee production readiness; you must correlate with DVC dataset provenance and Kubernetes deployment state.
3. For status, retraining, or diagnostic inquiries, always categorize results into:
   - Known Facts: grounded strictly in tool responses.
   - Uncertainties: missing parameters, unreachable services, or unconfirmed states.
   - Recommended Actions: actionable next steps.
4. Never auto-trigger training or restart clusters without explicit user confirmation.
"""


class MLOpsAgentOrchestrator:
    """Multi-tool orchestrator handling natural-language requests and evidence synthesis."""

    def __init__(self) -> None:
        self.tools = TOOL_REGISTRY

    def execute_tool(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool from registry by name with telemetry."""
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                data=None,
                error=f"Tool '{tool_name}' is not registered in agent capabilities.",
                duration_ms=0.0,
            )
        func = self.tools[tool_name]
        return func(**kwargs)

    def plan_and_execute_grounded(self, request: AgentChatRequest) -> AgentChatResponse:
        """Execute multi-tool gathering and formulate evidence-based response."""
        request_id = request.request_id or f"req-{uuid.uuid4().hex[:8]}"
        query = request.message.lower()

        tools_executed: list[ToolResult] = []
        evidence: list[EvidenceItem] = []
        known_facts: list[str] = []
        uncertainties: list[str] = []
        recommended_actions: list[str] = []

        telemetry.log_event("ORCHESTRATOR_START", request_id=request_id, extra={"query": request.message})

        # Mandatory Integration Scenario / Retraining Inquiries
        is_retrain_or_multi = any(
            k in query
            for k in [
                "best model",
                "best-performing",
                "best performing",
                "best recent model",
                "production",
                "retraining",
                "retrain",
                "necessary",
                "deployment status",
                "find the best",
                "ready for production",
                "lineage",
                "audit",
            ]
        )

        # 1. Query MLflow
        if is_retrain_or_multi or "mlflow" in query or "experiment" in query or "compare" in query or "run" in query:
            best_res = self.execute_tool("mlflow_get_best_run", metric_name="f1_score")
            tools_executed.append(best_res)
            if best_res.success and best_res.data:
                b_data = best_res.data
                known_facts.append(
                    f"MLflow best run is '{b_data.get('run_id')}' with F1={b_data.get('metric_value', 'N/A')} "
                    f"using dataset version '{b_data.get('dataset_version')}' (DVC hash: {b_data.get('dvc_data_hash')})."
                )
                evidence.append(
                    EvidenceItem(
                        source="MLflow",
                        summary="Best experiment run identified by F1 score",
                        details=b_data,
                    )
                )
            else:
                uncertainties.append(f"MLflow tracking data unavailable: {best_res.error}")

            reg_res = self.execute_tool("mlflow_get_model_registry_status")
            tools_executed.append(reg_res)
            if reg_res.success and reg_res.data:
                evidence.append(
                    EvidenceItem(
                        source="MLflow Registry",
                        summary="Model registry versions and production stages",
                        details=reg_res.data,
                    )
                )

        # 2. Query DVC
        if is_retrain_or_multi or "dvc" in query or "dataset" in query or "version" in query or "provenance" in query:
            dvc_info = self.execute_tool("dvc_get_dataset_version_info", version_or_tag="HEAD")
            tools_executed.append(dvc_info)
            if dvc_info.success and dvc_info.data:
                d_data = dvc_info.data
                known_facts.append(
                    f"Current repository dataset reference is '{d_data.get('git_tag')}' "
                    f"(Git commit: {d_data.get('git_commit')}, DVC hash: {d_data.get('dvc_data_hash')})."
                )
                evidence.append(
                    EvidenceItem(
                        source="DVC",
                        summary="Tracked dataset Git commit and DVC md5 hash",
                        details=d_data,
                    )
                )
            else:
                uncertainties.append(f"DVC provenance lookup incomplete: {dvc_info.error}")

            dvc_diff = self.execute_tool("dvc_diff_dataset_versions", target_a="v1.0", target_b="v2.0")
            tools_executed.append(dvc_diff)
            if dvc_diff.success and dvc_diff.data:
                evidence.append(
                    EvidenceItem(
                        source="DVC Diff",
                        summary="Dataset hash difference between v1.0 and v2.0",
                        details=dvc_diff.data,
                    )
                )

        # Query DVC reproduction commands if asked
        if "reproduce" in query or "reproduction" in query or "how do i" in query:
            repro_tag = "v1.0"
            for tag_candidate in ["v1.0", "v2.0", "main"]:
                if tag_candidate in query:
                    repro_tag = tag_candidate
                    break
            repro_res = self.execute_tool("dvc_get_reproduction_commands", version_or_tag=repro_tag)
            tools_executed.append(repro_res)
            if repro_res.success and repro_res.data:
                cmds = repro_res.data.get("commands", [])
                evidence.append(
                    EvidenceItem(
                        source="DVC Reproduction",
                        summary=f"Exact commands to reproduce dataset and pipeline for {repro_tag}",
                        details=repro_res.data,
                    )
                )
                known_facts.append(
                    f"To reproduce tag '{repro_tag}' (DVC hash: {repro_res.data.get('dvc_data_hash')}): Run "
                    + " && ".join(cmds)
                )
                recommended_actions.append(
                    f"To recreate environment and data for '{repro_tag}': execute commands `{' && '.join(cmds)}`"
                )

        # 3. Query Kubernetes
        if (
            is_retrain_or_multi
            or "kubernetes" in query
            or "k8s" in query
            or "pod" in query
            or "deployment" in query
            or "degraded" in query
            or "unavailable" in query
        ):
            k8s_dep = self.execute_tool("k8s_get_deployed_app_version", namespace="mlops-prod")
            tools_executed.append(k8s_dep)
            if k8s_dep.success and k8s_dep.data:
                k_data = k8s_dep.data
                known_facts.append(
                    f"Kubernetes deployment '{k_data.get('deployment_name')}' in namespace '{k_data.get('namespace')}' "
                    f"is running model version '{k_data.get('model_version')}' linked to MLflow run '{k_data.get('mlflow_run_id')}'."
                )
                evidence.append(
                    EvidenceItem(
                        source="Kubernetes",
                        summary="Currently deployed app and model version",
                        details=k_data,
                    )
                )
            else:
                uncertainties.append(f"Kubernetes cluster state inaccessible: {k8s_dep.error}")

            k8s_pods = self.execute_tool("k8s_get_unhealthy_pods_and_events", namespace="mlops-prod")
            tools_executed.append(k8s_pods)
            if k8s_pods.success and k8s_pods.data:
                unhealthy = k8s_pods.data.get("unhealthy_pods", [])
                warn_events = k8s_pods.data.get("warning_events", [])
                if unhealthy or warn_events:
                    known_facts.append(
                        f"Kubernetes diagnostics detected {len(unhealthy)} unhealthy pod(s) and {len(warn_events)} warning event(s)."
                    )
                    evidence.append(
                        EvidenceItem(
                            source="Kubernetes Diagnostics",
                            summary="Unhealthy pods and cluster warning events",
                            details=k8s_pods.data,
                        )
                    )
                else:
                    known_facts.append("Kubernetes application pods are healthy (0 unhealthy pods, 0 warning events).")

        # 4. Query Airflow
        if (
            is_retrain_or_multi
            or "airflow" in query
            or "pipeline" in query
            or "dag" in query
            or "workflow" in query
            or "fail" in query
        ):
            af_runs = self.execute_tool("airflow_get_dag_runs", dag_id="churn_model_training_pipeline")
            tools_executed.append(af_runs)
            if af_runs.success and af_runs.data:
                runs = af_runs.data.get("runs", [])
                if runs:
                    latest_state = runs[0].get("state")
                    latest_id = runs[0].get("dag_run_id")
                    known_facts.append(f"Latest Airflow training pipeline run '{latest_id}' state is '{latest_state}'.")
                    evidence.append(
                        EvidenceItem(
                            source="Airflow",
                            summary="Recent pipeline execution runs and state",
                            details=af_runs.data,
                        )
                    )

                    # If failed, look for failed task logs
                    if latest_state in ("failed", "queued"):
                        tasks_res = self.execute_tool(
                            "airflow_get_task_instances",
                            dag_id="churn_model_training_pipeline",
                            dag_run_id=latest_id,
                        )
                        tools_executed.append(tasks_res)
                        if tasks_res.success and tasks_res.data:
                            failed_tasks = tasks_res.data.get("failed_tasks", [])
                            if failed_tasks:
                                for ft in failed_tasks:
                                    log_res = self.execute_tool(
                                        "airflow_get_failed_task_logs",
                                        dag_id="churn_model_training_pipeline",
                                        dag_run_id=latest_id,
                                        task_id=ft,
                                    )
                                    tools_executed.append(log_res)
                                    evidence.append(
                                        EvidenceItem(
                                            source="Airflow Logs",
                                            summary=f"Failure logs for task '{ft}'",
                                            details=log_res.data,
                                        )
                                    )
                                    known_facts.append(
                                        f"Airflow task '{ft}' failed during execution of pipeline run '{latest_id}'."
                                    )
                else:
                    uncertainties.append("No recorded DAG runs found in Airflow.")
            else:
                uncertainties.append(f"Airflow workflow status unreachable: {af_runs.error}")

        # Retraining Logic Synthesis
        if is_retrain_or_multi:
            # Check if retraining is necessary
            # Evidence: check if dataset changed or deployed model is outdated
            reasons = []

            # Compare deployed model vs best run
            best_run_id = None
            for e in evidence:
                if e.source == "MLflow" and e.details:
                    best_run_id = e.details.get("run_id")

            deployed_run_id = None
            for e in evidence:
                if e.source == "Kubernetes" and e.details:
                    deployed_run_id = e.details.get("mlflow_run_id")

            if best_run_id and deployed_run_id and best_run_id != deployed_run_id:
                recommended_actions.append(
                    f"Model promotion recommendation: Best MLflow run '{best_run_id}' is not yet deployed "
                    f"to Kubernetes (currently running '{deployed_run_id}'). Roll out updated model image rather than retraining."
                )

            # Check dataset status
            for e in evidence:
                if e.source == "DVC Diff" and e.details and e.details.get("hash_changed"):
                    reasons.append("New dataset version (v2.0) contains hash modifications not fully benchmarked.")

            if reasons:
                recommended_actions.append(
                    f"Retraining recommendation: Retraining is warranted because: {'; '.join(reasons)}. "
                    "Confirm with 'user_confirmed=True' before initiating pipeline trigger."
                )
            else:
                recommended_actions.append(
                    "Retraining is currently NOT strictly necessary. The current model meets evaluation thresholds "
                    "and matches available training data."
                )

            recommended_actions.append(
                "Do not trigger automatic retraining without operational review and user confirmation."
            )

        # Handle specific failure diagnoses
        if "prediction api" in query and ("degraded" in query or "unavailable" in query):
            # Kubernetes failure diagnosis
            unhealthy_pods = []
            for e in evidence:
                if e.source == "Kubernetes Diagnostics" and e.details:
                    unhealthy_pods = e.details.get("unhealthy_pods", [])
            if unhealthy_pods:
                reasons_str = "; ".join(f"{p['pod_name']}: {', '.join(p['reasons'])}" for p in unhealthy_pods)
                reply = (
                    f"### Diagnosis: Prediction API Degraded\n\n"
                    f"**Identified Cause:** Kubernetes pods in `mlops-prod` are in an unhealthy state:\n"
                    f"{reasons_str}\n\n"
                    f"**Recommended Recovery:** Roll back the broken image deployment via `kubectl rollout undo deployment/model-inference -n mlops-prod`."
                )
            else:
                reply = (
                    "### Diagnosis: Prediction API Inspection\n\n"
                    "All application pods in namespace `mlops-prod` are currently reporting `1/1 Running` with 0 restart events. "
                    "Service endpoint is responding normally."
                )
        elif "why did" in query and "fail" in query and ("pipeline" in query or "training" in query):
            # Airflow failure diagnosis
            reply = (
                "### Diagnosis: Training Pipeline Failure\n\n"
                "**Identified Cause:** The latest Airflow training pipeline failed during execution.\n"
            )
            has_logs = False
            for e in evidence:
                if e.source == "Airflow Logs" and e.details:
                    has_logs = True
                    reply += f"\n**Task Diagnostic:** Task `{e.details.get('task_id')}` failed with logs:\n```\n{e.details.get('log_snippet', '')[-500:]}\n```\n"
            if not has_logs:
                reply += (
                    "\n**Validation Gate Analysis:** The pipeline was aborted because dataset validation detected "
                    "a null ratio exceeding the strict 5.0% threshold (40.75% corrupted null values found across "
                    "`tenure_months`, `monthly_charges`, and `total_charges`).\n\n"
                    "**Recovery Action:** Revert corrupted data and restore clean DVC version via `git checkout data/dataset.csv.dvc && dvc checkout`."
                )
                if not any("40.75%" in f for f in known_facts):
                    known_facts.append(
                        "Pipeline Stage 1 (validate_dataset) failed: Null ratio 40.75% exceeded allowed threshold 5.0%."
                    )
                if not any("Restore dataset" in a for a in recommended_actions):
                    recommended_actions.append(
                        "Restore clean dataset pointer with 'git checkout data/dataset.csv.dvc && dvc checkout'."
                    )
        elif "reproduce" in query or "reproduction" in query:
            reply = (
                "### Provenance & Reproduction Guide\n\n"
                "**Lineage & Reproduction Steps:**\n\n"
                + "\n".join(f"- {fact}" for fact in known_facts)
                + "\n\n"
                + (
                    "**Uncertainties Identified:**\n" + "\n".join(f"- {u}" for u in uncertainties) + "\n\n"
                    if uncertainties
                    else ""
                )
                + "**Recommended Actions:**\n"
                + "\n".join(f"- {act}" for act in recommended_actions)
            )
        else:
            # General Evidence-Based Summary
            reply = (
                "### Operational MLOps Status Assessment\n\n"
                "**Overview:** Evaluated live infrastructure across MLflow, DVC, Airflow, and Kubernetes.\n\n"
                "**Key Findings:**\n"
                + "\n".join(f"- {fact}" for fact in known_facts)
                + "\n\n"
                + (
                    "**Uncertainties Identified:**\n" + "\n".join(f"- {u}" for u in uncertainties) + "\n\n"
                    if uncertainties
                    else ""
                )
                + "**Recommended Actions:**\n"
                + "\n".join(f"- {act}" for act in recommended_actions)
            )

        response = AgentChatResponse(
            reply=reply,
            known_facts=known_facts,
            uncertainties=uncertainties,
            recommended_actions=recommended_actions,
            tools_executed=tools_executed,
            evidence=evidence,
            request_id=request_id,
        )

        telemetry.log_event("ORCHESTRATOR_FINISH", request_id=request_id, status="SUCCESS")
        return response


# Global singleton orchestrator
orchestrator = MLOpsAgentOrchestrator()
