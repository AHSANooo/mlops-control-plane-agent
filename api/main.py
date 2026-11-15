"""FastAPI Application Gateway exposing MLOps Agent conversational endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agent.config import settings
from agent.orchestrator import orchestrator
from agent.schemas import AgentChatRequest, AgentChatResponse
from agent.telemetry import telemetry

app = FastAPI(
    title="MLOps AI Agent Gateway",
    description="Production conversational interface for DVC, MLflow, Airflow, and Kubernetes operations",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    """Agent health status and available tools inventory."""
    return {
        "status": "healthy",
        "service": "mlops-ai-agent",
        "registered_tools_count": len(orchestrator.tools),
        "registered_tools": list(orchestrator.tools.keys()),
        "mlflow_tracking_uri": settings.mlflow_tracking_uri,
        "airflow_base_url": settings.airflow_base_url,
        "k8s_namespace": settings.k8s_default_namespace,
    }


@app.post("/api/v1/chat", response_model=AgentChatResponse)
def chat_endpoint(request: AgentChatRequest) -> AgentChatResponse:
    """Conversational endpoint executing multi-tool gathering and evidence-based synthesis."""
    try:
        response = orchestrator.chat(request)
        return response
    except Exception as e:
        telemetry.log_event("API_ERROR", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Agent orchestration failed: {e!s}") from e


def main() -> None:
    import uvicorn

    uvicorn.run("api.main:app", host=settings.app_host, port=settings.app_port, reload=False)


if __name__ == "__main__":
    main()
