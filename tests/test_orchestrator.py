"""Integration tests for MLOps Agent Orchestrator and FastAPI Gateway."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from agent.orchestrator import MLOpsAgentOrchestrator
from agent.schemas import AgentChatRequest, AgentChatResponse
from api.main import app

client = TestClient(app)


def test_api_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["registered_tools_count"] >= 15


def test_retraining_inquiry_does_not_call_trigger_dag() -> None:
    """Mandatory integration rule: Retraining must not be initiated automatically."""
    orchestrator = MLOpsAgentOrchestrator()

    with patch.object(orchestrator, "execute_tool", wraps=orchestrator.execute_tool) as mock_exec:
        req = AgentChatRequest(
            message="Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.",
            user_confirmed=False,
        )
        res = orchestrator.plan_and_execute_grounded(req)

        assert isinstance(res, AgentChatResponse)
        # Ensure trigger_training_dag was NEVER called
        called_tools = [c.args[0] for c in mock_exec.call_args_list]
        assert "airflow_trigger_training_dag" not in called_tools
        assert "k8s_restart_deployment" not in called_tools

        # Ensure response categorizes Known Facts and Recommended Actions
        assert len(res.known_facts) > 0
        assert len(res.recommended_actions) > 0
        assert len(res.evidence) > 0


def test_api_chat_endpoint_e2e() -> None:
    payload = {
        "message": "Find the best model from recent experiments and check Kubernetes deployment.",
        "user_confirmed": False,
    }
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "known_facts" in data
    assert "recommended_actions" in data
    assert "request_id" in data


def test_orchestrator_does_not_hallucinate_when_tools_unavailable() -> None:
    """Anti-hallucination test: when MLflow, DVC, Airflow, and K8s fail, the agent never invents operational facts."""
    from agent.schemas import ToolResult

    orchestrator = MLOpsAgentOrchestrator()

    def mock_failing_tool(tool_name: str, **kwargs: Any) -> ToolResult:
        return ToolResult(
            success=False,
            tool_name=tool_name,
            data=None,
            error=f"Service {tool_name} connection timeout / offline",
            duration_ms=5.0,
        )

    with patch.object(orchestrator, "execute_tool", side_effect=mock_failing_tool):
        req = AgentChatRequest(
            message="Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.",
            user_confirmed=False,
        )
        res = orchestrator.plan_and_execute_grounded(req)

        assert isinstance(res, AgentChatResponse)
        # Strict anti-hallucination check: zero invented known facts
        assert len(res.known_facts) == 0
        # All unavailable tools must be flagged in uncertainties
        assert len(res.uncertainties) >= 3
        assert any("MLflow" in u for u in res.uncertainties)
        assert any("Airflow" in u for u in res.uncertainties)
        assert any("Kubernetes" in u for u in res.uncertainties)
        assert "unreachable" in res.reply.lower() or "unavailable" in res.reply.lower()


def test_api_rejects_invalid_empty_request() -> None:
    """Input validation test: empty message payload must return 422 Unprocessable Entity."""
    response = client.post("/api/v1/chat", json={"message": ""})
    assert response.status_code == 422
