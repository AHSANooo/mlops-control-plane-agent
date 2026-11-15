"""Configuration module for MLOps AI Agent using Pydantic Settings."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Global configuration settings for MLOps agent services and security boundaries."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MLflow Tracking
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        alias="MLFLOW_TRACKING_URI",
    )

    # Apache Airflow 3
    airflow_base_url: str = Field(
        default="http://localhost:8080",
        alias="AIRFLOW_BASE_URL",
    )
    airflow_username: str = Field(
        default="admin",
        alias="AIRFLOW_USERNAME",
    )
    airflow_password: str = Field(
        default="",
        alias="AIRFLOW_PASSWORD",
    )
    airflow_api_token: str | None = Field(
        default=None,
        alias="AIRFLOW_API_TOKEN",
    )

    # Kubernetes Guardrails
    k8s_namespace_allowlist: list[str] = Field(
        default_factory=lambda: ["mlops-prod"],
    )
    k8s_default_namespace: str = Field(
        default="mlops-prod",
        alias="KUBERNETES_NAMESPACE",
    )
    kubeconfig_path: str | None = Field(
        default=None,
        alias="KUBECONFIG",
    )

    # DVC Workspace
    dvc_repo_dir: str = Field(
        default=".",
        alias="DVC_REPO_DIR",
    )

    # LLM Provider Configuration
    openai_api_key: str | None = Field(
        default=None,
        alias="OPENAI_API_KEY",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        alias="OPENAI_MODEL",
    )
    gemini_api_key: str | None = Field(
        default=None,
        alias="GEMINI_API_KEY",
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        alias="GEMINI_MODEL",
    )

    # Application & Telemetry
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8001, alias="APP_PORT")


# Global singleton instance
settings = AgentSettings()
