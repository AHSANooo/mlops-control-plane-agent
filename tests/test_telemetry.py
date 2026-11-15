"""Tests for structured telemetry logging and regex secret redaction."""

from __future__ import annotations

from agent.telemetry import JsonStructuredLogger, redact_secrets, sanitize_dict


def test_redact_secrets_masks_jwt() -> None:
    sample_jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    text = f"User token is {sample_jwt}"
    redacted = redact_secrets(text)
    assert sample_jwt not in redacted
    assert "[REDACTED_JWT]" in redacted


def test_redact_secrets_masks_api_keys() -> None:
    key_str = "sk-1234567890abcdef1234567890abcdef"
    text = f"Using API key {key_str}"
    redacted = redact_secrets(text)
    assert key_str not in redacted
    assert "[REDACTED_KEY]" in redacted


def test_sanitize_dict_removes_passwords() -> None:
    payload = {
        "username": "admin",
        "password": "supersecretpassword123",
        "nested": {"api_token": "token-xyz-12345", "metric": 0.85},
    }
    cleaned = sanitize_dict(payload)
    assert cleaned["password"] == "[REDACTED]"
    assert cleaned["nested"]["api_token"] == "[REDACTED]"
    assert cleaned["nested"]["metric"] == 0.85


def test_json_structured_logger_emits_valid_json() -> None:
    logger = JsonStructuredLogger("test_logger")
    evt = logger.log_event(
        event_type="TEST_EVENT",
        request_id="req-123",
        tool_name="test_tool",
        status="SUCCESS",
        duration_ms=42.5,
        extra={"param": "value"},
    )
    assert evt["event_type"] == "TEST_EVENT"
    assert evt["request_id"] == "req-123"
    assert evt["tool_name"] == "test_tool"
    assert evt["duration_ms"] == 42.5
