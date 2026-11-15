"""Tests for MLflow intelligence tools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.schemas import ToolResult
from agent.tools import mlflow_tools


def test_get_best_run_live_or_mock() -> None:
    res = mlflow_tools.get_best_run(experiment_name="customer-churn-prediction", metric_name="f1_score")
    assert isinstance(res, ToolResult)
    if res.success:
        assert "run_id" in res.data
        assert "metric_value" in res.data
        assert res.data["metric_evaluated"] == "f1_score"


def test_compare_runs() -> None:
    res = mlflow_tools.compare_runs(["run-1", "run-2"])
    assert isinstance(res, ToolResult)
    assert res.success is True
    assert "runs" in res.data


def test_mocked_connection_error_prevents_hallucination() -> None:
    """Verify that when MLflow server is disconnected, the tool fails cleanly without inventing metrics."""
    with patch("agent.tools.mlflow_tools._get_client") as mock_client:
        mock_instance = MagicMock()
        mock_instance.get_experiment_by_name.side_effect = ConnectionError("Connection refused to MLflow")
        mock_client.return_value = mock_instance

        res = mlflow_tools.get_best_run(experiment_name="customer-churn-prediction")
        assert isinstance(res, ToolResult)
        assert res.success is False
        assert res.data is None  # Never invents hallucinated data
        assert "Connection refused to MLflow" in str(res.error)
