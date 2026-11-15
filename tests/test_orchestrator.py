"""Integration tests for MLOps Agent Orchestrator and FastAPI Gateway."""

from __future__ import annotations

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
