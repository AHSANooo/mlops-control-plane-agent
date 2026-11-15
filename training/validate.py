"""Dataset validation step for MLOps training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import pandas as pd
import yaml


def load_config(config_path: str | Path = "params.yaml") -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
        return config


def validate_dataset(config_path: str | Path = "params.yaml") -> dict[str, Any]:
    """Validate raw dataset against schema constraints and null thresholds.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If required columns are missing or null ratio exceeds tolerance.
    """
    config = load_config(config_path)
    dataset_cfg = config.get("dataset", {})
    raw_path = Path(dataset_cfg.get("raw_path", "data/dataset.csv"))
    required_cols = dataset_cfg.get("required_columns", [])
    max_null_ratio = float(dataset_cfg.get("max_null_ratio", 0.05))

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at {raw_path.resolve()}")

    df = pd.read_csv(raw_path)

    # 1. Check required columns
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Validation failed: Dataset is missing required columns: {missing_cols}"
        )

    # 2. Check null ratios
    null_ratios = df[required_cols].isna().mean()
    max_detected_null = float(null_ratios.max())
    offending_cols = null_ratios[null_ratios > max_null_ratio].to_dict()

    if max_detected_null > max_null_ratio:
        raise ValueError(
            f"Validation failed: Maximum column null ratio {max_detected_null:.2%} "
            f"exceeds allowed threshold of {max_null_ratio:.2%}. Offending columns: {offending_cols}"
        )

    # 3. Check minimum row count
    if len(df) < 100:
        raise ValueError(f"Validation failed: Dataset row count {len(df)} is below minimum 100.")

    summary = {
        "status": "PASSED",
        "total_rows": int(len(df)),
        "columns_verified": len(required_cols),
        "max_null_ratio": round(max_detected_null, 4),
        "dataset_path": str(raw_path),
    }
    print(f"Data validation PASSED: {summary}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw dataset.")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()
    validate_dataset(args.config)


if __name__ == "__main__":
    main()
