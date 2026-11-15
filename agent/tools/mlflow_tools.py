"""MLflow Experiment Intelligence Tool Module."""

from __future__ import annotations

import time
from typing import Any

from mlflow.tracking import MlflowClient

from agent.config import settings
from agent.schemas import ToolResult
from agent.telemetry import telemetry


def _get_client() -> MlflowClient:
    """Instantiate authenticated or local MlflowClient."""
    return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)


def list_experiments_and_runs(
    experiment_name: str = "customer-churn-prediction",
    max_runs: int = 5,
) -> ToolResult:
    """List recent training runs and evaluation metrics for an experiment."""
    start_time = time.perf_counter()
    tool_name = "mlflow_list_experiments_and_runs"
    try:
        client = _get_client()
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                success=False,
                tool_name=tool_name,
                data=None,
                error=f"Experiment '{experiment_name}' not found on server {settings.mlflow_tracking_uri}",
                duration_ms=duration_ms,
            )

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=max_runs,
            order_by=["attributes.start_time DESC"],
        )

        run_summaries: list[dict[str, Any]] = []
        for r in runs:
            run_summaries.append(
                {
                    "run_id": r.info.run_id,
                    "status": r.info.status,
                    "start_time": r.info.start_time,
                    "metrics": r.data.metrics,
                    "tags": {
                        k: v
                        for k, v in r.data.tags.items()
                        if k in ("git_commit", "dataset_version", "dvc_data_hash", "model_type", "pipeline")
                    },
                    "params": r.data.params,
                }
            )

        duration_ms = (time.perf_counter() - start_time) * 1000
        result = ToolResult(
            success=True,
            tool_name=tool_name,
            data={
                "experiment_id": exp.experiment_id,
                "experiment_name": exp.name,
                "runs_found": len(run_summaries),
                "runs": run_summaries,
            },
            error=None,
            duration_ms=duration_ms,
        )
        telemetry.log_event("TOOL_EXEC", tool_name=tool_name, status="SUCCESS", duration_ms=duration_ms)
        return result
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        telemetry.log_event(
            "TOOL_EXEC", tool_name=tool_name, status="ERROR", duration_ms=duration_ms, extra={"err": str(e)}
        )
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"MLflow connection failure: {e!s}",
            duration_ms=duration_ms,
        )


def get_run_details(run_id: str) -> ToolResult:
    """Retrieve detailed metrics, parameters, tags, and artifacts for an MLflow run."""
    start_time = time.perf_counter()
    tool_name = "mlflow_get_run_details"
    try:
        client = _get_client()
        run = client.get_run(run_id)

        # List artifacts
        try:
            artifacts = client.list_artifacts(run_id)
            artifact_paths = [a.path for a in artifacts]
        except Exception:
            artifact_paths = []

        data = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
            "artifacts": artifact_paths,
            "dataset_version": run.data.tags.get("dataset_version", "unknown"),
            "dvc_data_hash": run.data.tags.get("dvc_data_hash", "unknown"),
            "git_commit": run.data.tags.get("git_commit", "unknown"),
        }
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=True,
            tool_name=tool_name,
            data=data,
            error=None,
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Failed to fetch MLflow run {run_id}: {e!s}",
            duration_ms=duration_ms,
        )


def get_best_run(
    experiment_name: str = "customer-churn-prediction",
    metric_name: str = "f1_score",
) -> ToolResult:
    """Identify the top run according to a specified evaluation metric (e.g. f1_score)."""
    start_time = time.perf_counter()
    tool_name = "mlflow_get_best_run"
    try:
        client = _get_client()
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                success=False,
                tool_name=tool_name,
                data=None,
                error=f"Experiment '{experiment_name}' not found.",
                duration_ms=duration_ms,
            )

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=1,
        )

        if not runs:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                success=False,
                tool_name=tool_name,
                data=None,
                error=f"No runs recorded in experiment '{experiment_name}'.",
                duration_ms=duration_ms,
            )

        best = runs[0]
        data = {
            "run_id": best.info.run_id,
            "experiment_name": experiment_name,
            "metric_evaluated": metric_name,
            "metric_value": best.data.metrics.get(metric_name),
            "all_metrics": best.data.metrics,
            "params": best.data.params,
            "dataset_version": best.data.tags.get("dataset_version", "unknown"),
            "dvc_data_hash": best.data.tags.get("dvc_data_hash", "unknown"),
            "git_commit": best.data.tags.get("git_commit", "unknown"),
        }
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=True,
            tool_name=tool_name,
            data=data,
            error=None,
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Failed to identify best run: {e!s}",
            duration_ms=duration_ms,
        )


def compare_runs(run_ids: list[str]) -> ToolResult:
    """Compare metrics, parameters, and data hashes across multiple runs."""
    start_time = time.perf_counter()
    tool_name = "mlflow_compare_runs"
    try:
        client = _get_client()
        comparisons: list[dict[str, Any]] = []
        for rid in run_ids:
            try:
                r = client.get_run(rid)
                comparisons.append(
                    {
                        "run_id": r.info.run_id,
                        "f1_score": r.data.metrics.get("f1_score"),
                        "roc_auc": r.data.metrics.get("roc_auc"),
                        "precision": r.data.metrics.get("precision"),
                        "recall": r.data.metrics.get("recall"),
                        "dataset_version": r.data.tags.get("dataset_version", "unknown"),
                        "dvc_data_hash": r.data.tags.get("dvc_data_hash", "unknown"),
                        "n_estimators": r.data.params.get("n_estimators"),
                        "max_depth": r.data.params.get("max_depth"),
                    }
                )
            except Exception as read_err:
                comparisons.append({"run_id": rid, "error": str(read_err)})

        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=True,
            tool_name=tool_name,
            data={"runs": comparisons, "total_compared": len(comparisons)},
            error=None,
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Failed to compare runs: {e!s}",
            duration_ms=duration_ms,
        )


def get_model_registry_status(model_name: str = "CustomerChurnClassifier") -> ToolResult:
    """Retrieve registered model versions, production stages, and run links from Model Registry."""
    start_time = time.perf_counter()
    tool_name = "mlflow_get_model_registry_status"
    try:
        client = _get_client()
        versions = client.search_model_versions(f"name='{model_name}'")
        summary: list[dict[str, Any]] = []

        for v in versions:
            summary.append(
                {
                    "model_name": v.name,
                    "version": str(v.version),
                    "current_stage": v.current_stage,
                    "run_id": v.run_id,
                    "status": v.status,
                    "creation_timestamp": v.creation_timestamp,
                }
            )

        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=True,
            tool_name=tool_name,
            data={"registered_name": model_name, "versions": summary, "count": len(summary)},
            error=None,
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Failed to query model registry: {e!s}",
            duration_ms=duration_ms,
        )
