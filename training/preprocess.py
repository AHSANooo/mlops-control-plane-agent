"""Data preprocessing step for MLOps training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(config_path: str | Path = "params.yaml") -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
        return config


def preprocess_data(config_path: str | Path = "params.yaml") -> dict[str, Any]:
    """Preprocess raw churn dataset and split into train/test splits."""
    config = load_config(config_path)
    dataset_cfg = config.get("dataset", {})
    raw_path = Path(dataset_cfg.get("raw_path", "data/dataset.csv"))
    train_path = Path(dataset_cfg.get("processed_train_path", "data/train.parquet"))
    test_path = Path(dataset_cfg.get("processed_test_path", "data/test.parquet"))
    target_col = dataset_cfg.get("target_column", "churn")
    test_size = float(dataset_cfg.get("test_size", 0.2))
    random_state = int(dataset_cfg.get("random_state", 42))

    df = pd.read_csv(raw_path)

    # Encode categorical columns
    categorical_cols = ["contract_type", "internet_service", "tech_support"]
    encoded_df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Drop identifiers
    if "customer_id" in encoded_df.columns:
        encoded_df = encoded_df.drop(columns=["customer_id"])

    # Ensure all boolean dummy columns are integer
    bool_cols = encoded_df.select_dtypes(include="bool").columns
    encoded_df[bool_cols] = encoded_df[bool_cols].astype(int)

    train_df, test_df = train_test_split(
        encoded_df,
        test_size=test_size,
        random_state=random_state,
        stratify=encoded_df[target_col],
    )

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    summary = {
        "status": "PASSED",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "features_count": int(train_df.shape[1] - 1),
        "train_path": str(train_path),
        "test_path": str(test_path),
    }
    print(f"Data preprocessing completed: {summary}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess dataset into train/test splits.")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    preprocess_data(args.config)


if __name__ == "__main__":
    main()
