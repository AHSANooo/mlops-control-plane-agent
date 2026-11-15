"""Tests for Airflow workflow tools and safety guards."""

from __future__ import annotations

from unittest.mock import patch

import httpx

from agent.schemas import ToolResult
from agent.tools import airflow_tools


def test_trigger_dag_rejected_without_user_confirmation() -> None:
    """Guardrail Rule 4: Airflow DAG triggers must strictly require user_confirmed=True."""
    res = airflow_tools.trigger_training_dag(
        dag_id="churn_model_training_pipeline",
        user_confirmed=False,
    )
    assert isinstance(res, ToolResult)
    assert res.success is False
    assert "SAFETY REJECTION" in res.error
    assert "user_confirmed=True" in res.error


def test_airflow_api_timeout_handling() -> None:
    """Tool must catch network timeout exceptions without raising raw unhandled errors to the orchestrator."""
    with patch("httpx.get") as mock_get:
        mock_get.side_effect = httpx.TimeoutException("Read timeout after 6000ms")
        res = airflow_tools.list_dags()
        assert isinstance(res, ToolResult)
        assert res.success is False
        assert res.data is None
        assert "timeout" in res.error.lower()


def test_get_task_instances_separates_failed_and_upstream_failed() -> None:
    """Verify get_task_instances distinguishes between failed tasks and downstream upstream_failed tasks."""
    mock_payload = {
        "task_instances": [
            {
                "task_id": "validate_dataset",
                "state": "failed",
                "try_number": 1,
                "start_date": "2026-11-15T16:00:00Z",
                "end_date": "2026-11-15T16:00:05Z",
                "duration": 5.0,
            },
            {
                "task_id": "preprocess_data",
                "state": "upstream_failed",
                "try_number": 0,
                "start_date": None,
                "end_date": None,
                "duration": None,
            },
            {
                "task_id": "train_model",
                "state": "upstream_failed",
                "try_number": 0,
                "start_date": None,
                "end_date": None,
                "duration": None,
            },
            {
                "task_id": "load_data",
                "state": "success",
                "try_number": 1,
                "start_date": "2026-11-15T15:59:50Z",
                "end_date": "2026-11-15T15:59:59Z",
                "duration": 9.0,
            },
        ]
    }
    with patch("httpx.get") as mock_get:
        mock_resp = httpx.Response(200, json=mock_payload, request=httpx.Request("GET", "http://test"))
        mock_get.return_value = mock_resp
        res = airflow_tools.get_task_instances(dag_id="churn_model_training_pipeline", dag_run_id="run_123")
        assert res.success is True
        assert res.data is not None
        assert res.data["failed_tasks"] == ["validate_dataset"]
        assert res.data["upstream_failed_tasks"] == ["preprocess_data", "train_model"]
        assert len(res.data["tasks"]) == 4
        assert res.data["tasks"][0]["try_number"] == 1
        assert res.data["tasks"][1]["try_number"] == 0
