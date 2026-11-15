"""Structured JSON telemetry logging and secret redaction module."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

# Regex patterns matching sensitive information
SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|secret|password|bearer\s+[a-z0-9_\-\.]+)=([^\s&]+)"),
    re.compile(r"(?i)(authorization:\s*bearer\s+)([a-zA-Z0-9_\-\.]+)"),
    re.compile(r"ey[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}"),  # JWT
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),  # OpenAI/API key
]


def redact_secrets(text: str) -> str:
    """Mask sensitive tokens, passwords, and API keys from log strings."""
    if not isinstance(text, str):
        return text

    redacted = text
    # Mask JWTs
    redacted = SECRET_PATTERNS[2].sub("[REDACTED_JWT]", redacted)
    # Mask OpenAI-style keys
    redacted = SECRET_PATTERNS[3].sub("[REDACTED_KEY]", redacted)
    # Mask Authorization header
    redacted = SECRET_PATTERNS[1].sub(r"\1[REDACTED_AUTH]", redacted)
    # Mask key=value or json secrets
    redacted = SECRET_PATTERNS[0].sub(r"\1=[REDACTED]", redacted)

    return redacted


def sanitize_dict(data: Any) -> Any:
    """Recursively scrub sensitive keys from telemetry dictionaries."""
    if isinstance(data, dict):
        sanitized: dict[str, Any] = {}
        for k, v in data.items():
            k_lower = str(k).lower()
            if any(secret_word in k_lower for secret_word in ["password", "secret", "token", "key", "authorization"]):
                sanitized[k] = "[REDACTED]"
            elif isinstance(v, (dict, list)):
                sanitized[k] = sanitize_dict(v)
            elif isinstance(v, str):
                sanitized[k] = redact_secrets(v)
            else:
                sanitized[k] = v
        return sanitized
    elif isinstance(data, list):
        return [sanitize_dict(item) for item in data]
    elif isinstance(data, str):
        return redact_secrets(data)
    return data


class JsonStructuredLogger:
    """Structured JSON logger with automatic sanitization for MLOps agent actions."""

    def __init__(self, logger_name: str = "mlops_agent") -> None:
        self.logger = logging.getLogger(logger_name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_event(
        self,
        event_type: str,
        request_id: str | None = None,
        tool_name: str | None = None,
        status: str = "SUCCESS",
        duration_ms: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Emit a structured JSON log entry."""
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "request_id": request_id or "unassigned",
            "status": status,
            "duration_ms": round(duration_ms, 2),
        }
        if tool_name:
            payload["tool_name"] = tool_name
        if extra:
            payload["data"] = sanitize_dict(extra)

        json_line = json.dumps(payload, default=str)
        if status in ("ERROR", "FAILED", "SECURITY_REJECTION"):
            self.logger.error(json_line)
        elif status == "WARNING":
            self.logger.warning(json_line)
        else:
            self.logger.info(json_line)

        return payload


# Global telemetry logger instance
telemetry = JsonStructuredLogger()
