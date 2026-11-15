"""Pytest test suite configuration and environment setup."""

from __future__ import annotations

import os

# Ensure network calls to offline services fail fast without hanging
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "2"
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1"
os.environ["GIT_TERMINAL_PROMPT"] = "0"
os.environ["DVC_NO_ANALYTICS"] = "1"
