"""Main agent entrypoint module."""

from __future__ import annotations

from agent.config import settings
from agent.orchestrator import MLOpsAgentOrchestrator, orchestrator
from agent.schemas import AgentChatRequest, AgentChatResponse


def chat(
    message: str,
    user_confirmed: bool = False,
    request_id: str | None = None,
) -> AgentChatResponse:
    """Direct programmatic interface for conversational MLOps agent."""
    req = AgentChatRequest(
        message=message,
        user_confirmed=user_confirmed,
        request_id=request_id,
    )
    return orchestrator.chat(req)


__all__ = ["MLOpsAgentOrchestrator", "chat", "orchestrator", "settings"]
