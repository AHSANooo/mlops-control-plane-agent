"""Tests for Kubernetes tools and security admission boundaries."""

from __future__ import annotations

from agent.schemas import ToolResult
from agent.tools import kubernetes_tools


def test_k8s_rejects_unauthorized_namespace() -> None:
    """Guardrail Rule 4: Rejects unauthorized namespaces outside allowlist (e.g. kube-system)."""
    res = kubernetes_tools.list_deployments_and_pods(namespace="kube-system")
    assert isinstance(res, ToolResult)
    assert res.success is False
    assert "SAFETY REJECTION" in res.error
    assert "forbidden" in res.error.lower()


def test_k8s_restart_rejects_unconfirmed_mutation() -> None:
    """Guardrail Rule 4: Restart operations must reject when user_confirmed is False."""
    res = kubernetes_tools.restart_deployment(
        deployment_name="model-inference",
        namespace="mlops-prod",
        user_confirmed=False,
    )
    assert isinstance(res, ToolResult)
    assert res.success is False
    assert "SAFETY REJECTION" in res.error
    assert "user_confirmed=True" in res.error


def test_k8s_restart_rejects_non_prod_namespace_even_if_confirmed() -> None:
    """Guardrail Rule 4: Mutations are strictly confined to mlops-prod."""
    res = kubernetes_tools.restart_deployment(
        deployment_name="model-inference",
        namespace="default",
        user_confirmed=True,
    )
    assert isinstance(res, ToolResult)
    assert res.success is False
    assert "SAFETY REJECTION" in res.error
    assert "strictly confined to 'mlops-prod'" in res.error


def test_k8s_get_deployed_app_version() -> None:
    res = kubernetes_tools.get_deployed_app_version(namespace="mlops-prod")
    assert isinstance(res, ToolResult)
    if res.success:
        assert res.data["namespace"] == "mlops-prod"
        assert "model-inference" in res.data["deployment_name"]
