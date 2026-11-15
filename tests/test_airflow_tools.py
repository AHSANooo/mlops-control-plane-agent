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
