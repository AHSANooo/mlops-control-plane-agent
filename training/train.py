"""Model training, MLflow experiment logging, and quality gate module."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Any
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import yaml


def load_config(config_path: str | Path = "params.yaml") -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
        return config


def get_git_commit() -> str:
    """Retrieve short Git commit hash without shell=True."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_git_tag() -> str:
    """Retrieve exact Git tag if available without shell=True."""
    try:
        proc = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass

    # Fallback to latest reachable tag
    try:
        proc = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass
    return "v1.0"


def get_dvc_hash(dvc_file_path: str | Path = "data/dataset.csv.dvc") -> str:
    """Extract md5 hash from tracked DVC pointer file."""
    path = Path(dvc_file_path)
    if not path.exists():
        return "uncommitted-dvc-hash"
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            outs = data.get("outs", [])
            if outs and isinstance(outs, list) and "md5" in outs[0]:
                return str(outs[0]["md5"])
    except Exception:
        pass
    return "unknown-hash"


def train_model(config_path: str | Path = "params.yaml") -> dict[str, Any]:
    """Train classifier, log metrics/artifacts to MLflow, and return evaluation results."""
    config = load_config(config_path)
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    mlflow_cfg = config.get("mlflow", {})

    train_path = Path(dataset_cfg.get("processed_train_path", "data/train.parquet"))
    test_path = Path(dataset_cfg.get("processed_test_path", "data/test.parquet"))
    target_col = dataset_cfg.get("target_column", "churn")

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {train_path} or {test_path}. Run preprocessing first."
        )

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    x_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    x_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    n_estimators = int(model_cfg.get("n_estimators", 100))
    max_depth = int(model_cfg.get("max_depth", 6))
    random_state = int(model_cfg.get("random_state", 42))

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    clf.fit(x_train, y_train)

    # Evaluation
    y_pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)[:, 1]

    metrics = {
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    # Provenance tags
    git_commit = get_git_commit()
    dataset_version = get_git_tag()
    dvc_hash = get_dvc_hash("data/dataset.csv.dvc")

    tags = {
        "git_commit": git_commit,
        "dataset_version": dataset_version,
        "dvc_data_hash": dvc_hash,
        "model_type": model_cfg.get("type", "RandomForestClassifier"),
    }

    tracking_uri = mlflow_cfg.get("tracking_uri", "http://localhost:5000")
    experiment_name = mlflow_cfg.get("experiment_name", "customer-churn-prediction")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Log parameters
        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": random_state,
                "train_samples": len(train_df),
                "test_samples": len(test_df),
            }
        )

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log tags
        mlflow.set_tags(tags)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            registered_model_name=None,  # Registration handled by quality gate
        )

    print(f"MLflow Run completed: run_id={run_id}, metrics={metrics}, tags={tags}")

    return {
        "run_id": run_id,
        "metrics": metrics,
        "tags": tags,
        "tracking_uri": tracking_uri,
    }


def apply_quality_gate(
    run_id: str,
    f1: float,
    config_path: str | Path = "params.yaml",
) -> dict[str, Any]:
    """Validate model quality against minimum thresholds before registering in Model Registry.

    Raises:
        ValueError: If f1_score is below min_f1 threshold.
    """
    config = load_config(config_path)
    gate_cfg = config.get("quality_gate", {})
    min_f1 = float(gate_cfg.get("min_f1", 0.70))
    registered_name = config.get("mlflow", {}).get(
        "registered_model_name", "CustomerChurnClassifier"
    )
    tracking_uri = config.get("mlflow", {}).get("tracking_uri", "http://localhost:5000")

    if f1 < min_f1:
        raise ValueError(
            f"Quality Gate FAILED: F1 score ({f1:.4f}) is below minimum threshold ({min_f1:.4f}). "
            f"Model registration blocked for run_id={run_id}."
        )

    # Register model in MLflow Registry
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"runs:/{run_id}/model"
    reg_result = mlflow.register_model(model_uri=model_uri, name=registered_name)

    summary = {
        "status": "PASSED",
        "registered_model_name": registered_name,
        "model_version": str(reg_result.version),
        "run_id": run_id,
        "f1_score": round(f1, 4),
        "min_f1_threshold": min_f1,
    }
    print(f"Quality gate PASSED: {summary}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate churn model.")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    train_res = train_model(args.config)
    f1 = train_res["metrics"]["f1_score"]
    run_id = train_res["run_id"]
    apply_quality_gate(run_id=run_id, f1=f1, config_path=args.config)


if __name__ == "__main__":
    main()
