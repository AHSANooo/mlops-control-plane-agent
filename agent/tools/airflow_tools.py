"""Apache Airflow 3 Workflow Agent Tool Module."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import httpx

from agent.config import settings
from agent.schemas import ToolResult
from agent.telemetry import telemetry


def _get_http_client(timeout: float = 6.0) -> httpx.Client:
    """Create an HTTP client with retry transport for resilient Airflow connection."""
    transport = httpx.HTTPTransport(retries=2)
    return httpx.Client(transport=transport, timeout=timeout)


def _get_auth_headers() -> dict[str, str]:
    """Retrieve Bearer JWT auth token from Airflow 3 /auth/token or fallback."""
    if settings.airflow_api_token:
        return {"Authorization": f"Bearer {settings.airflow_api_token}"}

    if not settings.airflow_password:
        return {}

    try:
        # Request JWT token from Airflow 3 simple auth manager
        token_url = f"{settings.airflow_base_url.rstrip('/')}/auth/token"
        payload = {
            "username": settings.airflow_username,
            "password": settings.airflow_password,
        }
        with _get_http_client(timeout=5.0) as client:
            resp = client.post(token_url, json=payload)
            if resp.status_code == 201:
                token = resp.json().get("access_token")
                if token:
                    return {"Authorization": f"Bearer {token}"}
    except Exception as e:
        telemetry.log_event("AUTH_WARN", tool_name="airflow_auth", extra={"warning": str(e)})

    # Fallback to empty header
    return {}


def list_dags() -> ToolResult:
    """List available Airflow workflows and their active/paused status."""
    start_time = time.perf_counter()
    tool_name = "airflow_list_dags"
    try:
        headers = _get_auth_headers()
        base = settings.airflow_base_url.rstrip("/")

        # Try Airflow 3 /api/v2/dags then fallback to /api/v1/dags
        endpoints = [f"{base}/api/v2/dags", f"{base}/api/v1/dags"]
        last_error = ""

        for ep in endpoints:
            try:
                auth = (
                    (settings.airflow_username, settings.airflow_password) if not headers.get("Authorization") else None
                )
                res = httpx.get(ep, headers=headers, auth=auth, timeout=6.0)
                if res.status_code == 200:
                    data = res.json()
                    dags = data.get("dags", [])
                    simplified = [
                        {
                            "dag_id": d.get("dag_id"),
                            "description": d.get("description"),
                            "is_paused": d.get("is_paused"),
                            "tags": [t.get("name") if isinstance(t, dict) else t for t in d.get("tags", [])],
                            "timetable_summary": d.get("timetable_summary"),
                        }
                        for d in dags
                    ]
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    return ToolResult(
                        success=True,
                        tool_name=tool_name,
                        data={"dags": simplified, "count": len(simplified)},
                        error=None,
                        duration_ms=duration_ms,
                    )
                else:
                    last_error = f"HTTP {res.status_code}: {res.text[:200]}"
            except httpx.RequestError as req_err:
                last_error = str(req_err)

        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Airflow service unavailable: {last_error}",
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Failed to list DAGs: {e!s}",
            duration_ms=duration_ms,
        )


def get_dag_runs(
    dag_id: str = "churn_model_training_pipeline",
    limit: int = 5,
) -> ToolResult:
    """Retrieve execution history, run states, and timestamps for a DAG."""
    start_time = time.perf_counter()
    tool_name = "airflow_get_dag_runs"
    try:
        headers = _get_auth_headers()
        base = settings.airflow_base_url.rstrip("/")
        url = f"{base}/api/v2/dags/{dag_id}/dagRuns"

        auth = (settings.airflow_username, settings.airflow_password) if not headers.get("Authorization") else None
        res = httpx.get(
            url,
            headers=headers,
            auth=auth,
            params={"limit": limit, "order_by": "-start_date"},
            timeout=6.0,
        )

        if res.status_code == 200:
            data = res.json()
            runs = data.get("dag_runs", [])
            simplified = [
                {
                    "dag_run_id": r.get("dag_run_id"),
                    "state": r.get("state"),
                    "start_date": r.get("start_date"),
                    "end_date": r.get("end_date"),
                    "run_type": r.get("run_type"),
                }
                for r in runs
            ]
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                success=True,
                tool_name=tool_name,
                data={"dag_id": dag_id, "runs": simplified, "count": len(simplified)},
                error=None,
                duration_ms=duration_ms,
            )

        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Airflow API error {res.status_code}: {res.text[:200]}",
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Failed to fetch DAG runs: {e!s}",
            duration_ms=duration_ms,
        )


def get_task_instances(dag_id: str, dag_run_id: str) -> ToolResult:
    """Retrieve individual task status (success/failed/upstream_failed) for a specific DAG run."""
    start_time = time.perf_counter()
    tool_name = "airflow_get_task_instances"
    try:
        headers = _get_auth_headers()
        base = settings.airflow_base_url.rstrip("/")
        url = f"{base}/api/v2/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances"

        auth = (settings.airflow_username, settings.airflow_password) if not headers.get("Authorization") else None
        res = httpx.get(url, headers=headers, auth=auth, timeout=6.0)

        if res.status_code == 200:
            data = res.json()
            tasks = data.get("task_instances", [])
            summary = [
                {
                    "task_id": t.get("task_id"),
                    "state": t.get("state"),
                    "try_number": t.get("try_number"),
                    "start_date": t.get("start_date"),
                    "end_date": t.get("end_date"),
                    "duration": t.get("duration"),
                }
                for t in tasks
            ]
            failed_tasks = [t["task_id"] for t in summary if t["state"] == "failed"]
            upstream_failed_tasks = [t["task_id"] for t in summary if t["state"] == "upstream_failed"]
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                success=True,
                tool_name=tool_name,
                data={
                    "dag_id": dag_id,
                    "dag_run_id": dag_run_id,
                    "tasks": summary,
                    "failed_tasks": failed_tasks,
                    "upstream_failed_tasks": upstream_failed_tasks,
                },
                error=None,
                duration_ms=duration_ms,
            )

        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Airflow API error {res.status_code}: {res.text[:200]}",
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Failed to fetch task instances: {e!s}",
            duration_ms=duration_ms,
        )


def get_failed_task_logs(
    dag_id: str,
    dag_run_id: str,
    task_id: str,
    try_number: int = 1,
) -> ToolResult:
    """Retrieve execution logs and stack trace diagnostics for a failed task, trying specified try_number with fallback."""
    start_time = time.perf_counter()
    tool_name = "airflow_get_failed_task_logs"
    try:
        headers = _get_auth_headers()
        base = settings.airflow_base_url.rstrip("/")

        urls_to_try = [
            f"{base}/api/v2/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/{try_number}",
        ]
        if try_number != 1:
            urls_to_try.append(f"{base}/api/v2/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/1")

        auth = (
            (settings.airflow_username, settings.airflow_password)
            if not headers.get("Authorization") and settings.airflow_password
            else None
        )

        last_error = ""
        with _get_http_client(timeout=6.0) as client:
            for url in urls_to_try:
                try:
                    res = client.get(url, headers=headers, auth=auth)
                    if res.status_code == 200:
                        content = res.text
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        return ToolResult(
                            success=True,
                            tool_name=tool_name,
                            data={
                                "dag_id": dag_id,
                                "dag_run_id": dag_run_id,
                                "task_id": task_id,
                                "try_number": try_number,
                                "log_snippet": content[-3000:],  # Return tail of the log
                            },
                            error=None,
                            duration_ms=duration_ms,
                        )
                    last_error = f"HTTP {res.status_code}: {res.text[:200]}"
                except Exception as req_err:
                    last_error = str(req_err)

        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Could not retrieve task logs: {last_error}",
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Failed to retrieve task logs: {e!s}",
            duration_ms=duration_ms,
        )


def trigger_training_dag(
    dag_id: str = "churn_model_training_pipeline",
    user_confirmed: bool = False,
    conf: dict[str, Any] | None = None,
) -> ToolResult:
    """Trigger an approved training workflow in Airflow. Requires explicit user confirmation."""
    start_time = time.perf_counter()
    tool_name = "airflow_trigger_training_dag"

    # Rule 4: Mandatory user confirmation check
    if not user_confirmed:
        duration_ms = (time.perf_counter() - start_time) * 1000
        err_msg = (
            "SAFETY REJECTION: Workflow execution rejected. Airflow DAG triggers mutate system state "
            "and require explicit user confirmation (user_confirmed=True)."
        )
        telemetry.log_event("SAFETY_REJECTION", tool_name=tool_name, status="SECURITY_REJECTION")
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=err_msg,
            duration_ms=duration_ms,
        )

    try:
        headers = _get_auth_headers()
        base = settings.airflow_base_url.rstrip("/")
        url = f"{base}/api/v2/dags/{dag_id}/dagRuns"

        auth = (settings.airflow_username, settings.airflow_password) if not headers.get("Authorization") else None
        body: dict[str, Any] = {
            "logical_date": datetime.now(UTC).isoformat(),
        }
        if conf:
            body["conf"] = conf

        res = httpx.post(url, headers=headers, auth=auth, json=body, timeout=8.0)
        duration_ms = (time.perf_counter() - start_time) * 1000

        if res.status_code in (200, 201):
            data = res.json()
            return ToolResult(
                success=True,
                tool_name=tool_name,
                data={
                    "dag_id": dag_id,
                    "dag_run_id": data.get("dag_run_id"),
                    "state": data.get("state"),
                    "logical_date": data.get("logical_date"),
                },
                error=None,
                duration_ms=duration_ms,
            )

        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Failed to trigger DAG (HTTP {res.status_code}): {res.text[:200]}",
            duration_ms=duration_ms,
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Airflow trigger failed: {e!s}",
            duration_ms=duration_ms,
        )
