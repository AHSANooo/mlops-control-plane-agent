"""Airflow 3 TaskFlow Pipeline for Churn Model Training, MLflow Logging, and Quality Gate."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from airflow.decorators import dag, task

DEFAULT_ARGS = {
    "owner": "mlops-platform",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(seconds=10),
}


@dag(
    dag_id="churn_model_training_pipeline",
    default_args=DEFAULT_ARGS,
    description="5-Stage MLOps Training Pipeline with Validation, Preprocessing, MLflow, and Quality Gate",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "churn", "production"],
)
def churn_training_pipeline() -> None:
    """5-stage TaskFlow pipeline enforcing data validation and quality gates."""

    @task(task_id="validate_dataset")
    def task_validate_dataset() -> dict[str, Any]:
        """Validate raw dataset schema, presence, and maximum null threshold (<5%)."""
        import pandas as pd

        # Find params.yaml and data path across local and container environments
        base_dir = Path(os.environ.get("AIRFLOW_HOME", "/opt/airflow"))
        candidates = [
            Path("params.yaml"),
            base_dir / "params.yaml",
            Path("/opt/airflow/params.yaml"),
        ]
        config_path = next((p for p in candidates if p.exists()), Path("params.yaml"))

        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        dataset_cfg = cfg.get("dataset", {})
        raw_path = Path(dataset_cfg.get("raw_path", "data/dataset.csv"))
        if not raw_path.is_absolute() and not raw_path.exists():
            raw_path = base_dir / raw_path

        if not raw_path.exists():
            raise FileNotFoundError(f"Dataset not found at {raw_path}")

        df = pd.read_csv(raw_path)
        required_cols = dataset_cfg.get("required_columns", [])
        max_null_ratio = float(dataset_cfg.get("max_null_ratio", 0.05))

        # Check required columns
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset validation failed: missing columns {missing}")

        # Check null ratios
        null_ratios = df[required_cols].isna().mean()
        max_null = float(null_ratios.max())
        if max_null > max_null_ratio:
            offending = null_ratios[null_ratios > max_null_ratio].to_dict()
            raise ValueError(
                f"Dataset validation failed: Maximum null ratio {max_null:.2%} "
                f"exceeds limit {max_null_ratio:.2%}. Offending columns: {offending}"
            )

        print(f"Validation successful on {len(df)} rows. Max null ratio: {max_null:.2%}")
        return {
            "status": "PASSED",
            "rows": len(df),
            "max_null_ratio": max_null,
            "raw_path": str(raw_path),
            "config_path": str(config_path),
        }

    @task(task_id="preprocess_data")
    def task_preprocess_data(val_result: dict[str, Any]) -> dict[str, Any]:
        """Split data into train and test sets and perform one-hot encoding."""
        import pandas as pd
        from sklearn.model_selection import train_test_split

        config_path = Path(val_result["config_path"])
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        base_dir = Path(os.environ.get("AIRFLOW_HOME", "/opt/airflow"))
        raw_path = Path(val_result["raw_path"])
        df = pd.read_csv(raw_path)

        categorical_cols = ["contract_type", "internet_service", "tech_support"]
        encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        if "customer_id" in encoded.columns:
            encoded = encoded.drop(columns=["customer_id"])

        bool_cols = encoded.select_dtypes(include="bool").columns
        encoded[bool_cols] = encoded[bool_cols].astype(int)

        target_col = cfg.get("dataset", {}).get("target_column", "churn")
        test_size = float(cfg.get("dataset", {}).get("test_size", 0.2))

        train_df, test_df = train_test_split(
            encoded,
            test_size=test_size,
            random_state=42,
            stratify=encoded[target_col],
        )

        train_path = Path(cfg.get("dataset", {}).get("processed_train_path", "data/train.parquet"))
        test_path = Path(cfg.get("dataset", {}).get("processed_test_path", "data/test.parquet"))
        if not train_path.is_absolute():
            train_path = base_dir / train_path
        if not test_path.is_absolute():
            test_path = base_dir / test_path

        train_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.parent.mkdir(parents=True, exist_ok=True)

        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)

        print(f"Preprocessing completed: train={len(train_df)} rows, test={len(test_df)} rows")
        return {
            "status": "PASSED",
            "train_path": str(train_path),
            "test_path": str(test_path),
            "config_path": str(config_path),
            "features_count": int(train_df.shape[1] - 1),
        }

    @task(task_id="train_model")
    def task_train_model(prep_result: dict[str, Any]) -> dict[str, Any]:
        """Fit RandomForest model on preprocessed training dataset."""
        import joblib
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier

        config_path = Path(prep_result["config_path"])
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        train_path = Path(prep_result["train_path"])
        train_df = pd.read_parquet(train_path)

        target_col = cfg.get("dataset", {}).get("target_column", "churn")
        x_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]

        model_cfg = cfg.get("model", {})
        n_estimators = int(model_cfg.get("n_estimators", 100))
        max_depth = int(model_cfg.get("max_depth", 6))
        random_state = int(model_cfg.get("random_state", 42))

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        clf.fit(x_train, y_train)

        model_dir = Path("/tmp/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = model_dir / "trained_model.joblib"
        joblib.dump(clf, model_save_path)

        return {
            "status": "PASSED",
            "model_path": str(model_save_path),
            "test_path": prep_result["test_path"],
            "config_path": str(config_path),
            "n_estimators": n_estimators,
            "max_depth": max_depth,
        }

    @task(task_id="evaluate_and_log_mlflow")
    def task_evaluate_and_log_mlflow(train_result: dict[str, Any]) -> dict[str, Any]:
        """Evaluate model against test split and log parameters, metrics, tags to MLflow."""
        import subprocess

        import joblib
        import mlflow.sklearn
        import pandas as pd
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

        import mlflow

        config_path = Path(train_result["config_path"])
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        clf = joblib.load(train_result["model_path"])
        test_df = pd.read_parquet(train_result["test_path"])

        target_col = cfg.get("dataset", {}).get("target_column", "churn")
        x_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        y_pred = clf.predict(x_test)
        y_prob = clf.predict_proba(x_test)[:, 1]

        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        roc = float(roc_auc_score(y_test, y_prob))

        # Git commit
        git_commit = "unknown"
        try:
            res = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
                shell=False,
            )
            if res.returncode == 0:
                git_commit = res.stdout.strip()
        except Exception:
            pass

        # Git tag
        dataset_version = "v1.0"
        try:
            res = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True,
                text=True,
                check=False,
                shell=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                dataset_version = res.stdout.strip()
        except Exception:
            pass

        # DVC hash
        dvc_hash = "unknown"
        for p in [Path("data/dataset.csv.dvc"), Path("/opt/airflow/data/dataset.csv.dvc")]:
            if p.exists():
                try:
                    with open(p, encoding="utf-8") as df_p:
                        dvc_meta = yaml.safe_load(df_p)
                        dvc_hash = str(dvc_meta.get("outs", [{}])[0].get("md5", "unknown"))
                        break
                except Exception:
                    pass

        # MLflow URI: check container vs host
        tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI",
            cfg.get("mlflow", {}).get("tracking_uri", "http://mlflow:5000"),
        )
        experiment_name = cfg.get("mlflow", {}).get("experiment_name", "customer-churn-prediction")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_params(
                {
                    "n_estimators": train_result["n_estimators"],
                    "max_depth": train_result["max_depth"],
                    "test_rows": len(test_df),
                }
            )
            mlflow.log_metrics(
                {
                    "f1_score": f1,
                    "precision": prec,
                    "recall": rec,
                    "roc_auc": roc,
                }
            )
            mlflow.set_tags(
                {
                    "git_commit": git_commit,
                    "dataset_version": dataset_version,
                    "dvc_data_hash": dvc_hash,
                    "pipeline": "airflow-3-taskflow",
                }
            )
            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

        print(f"Logged MLflow run {run_id}: F1={f1:.4f}, AUC={roc:.4f}")
        return {
            "status": "PASSED",
            "run_id": run_id,
            "f1_score": f1,
            "roc_auc": roc,
            "tracking_uri": tracking_uri,
            "config_path": str(config_path),
        }

    @task(task_id="apply_model_quality_gate")
    def task_apply_model_quality_gate(eval_result: dict[str, Any]) -> dict[str, Any]:
        """Verify model satisfies quality gate criteria before Model Registry promotion."""
        import mlflow

        config_path = Path(eval_result["config_path"])
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        min_f1 = float(cfg.get("quality_gate", {}).get("min_f1", 0.70))
        f1_score_val = float(eval_result["f1_score"])
        run_id = eval_result["run_id"]
        tracking_uri = eval_result["tracking_uri"]

        if f1_score_val < min_f1:
            raise ValueError(
                f"Quality Gate FAILED: F1 score {f1_score_val:.4f} < minimum threshold {min_f1:.4f}. "
                f"Model registration blocked for run {run_id}."
            )

        registered_name = cfg.get("mlflow", {}).get("registered_model_name", "CustomerChurnClassifier")
        mlflow.set_tracking_uri(tracking_uri)
        model_uri = f"runs:/{run_id}/model"
        reg = mlflow.register_model(model_uri=model_uri, name=registered_name)

        print(f"Quality gate PASSED: Model {registered_name} registered as version {reg.version}")
        return {
            "status": "PASSED",
            "registered_model": registered_name,
            "version": str(reg.version),
            "f1_score": f1_score_val,
            "min_f1": min_f1,
            "run_id": run_id,
        }

    # Task execution order
    val_out = task_validate_dataset()
    prep_out = task_preprocess_data(val_out)
    train_out = task_train_model(prep_out)
    eval_out = task_evaluate_and_log_mlflow(train_out)
    task_apply_model_quality_gate(eval_out)


training_dag = churn_training_pipeline()
