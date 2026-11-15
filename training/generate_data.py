"""Synthetic tabular dataset generator for MLOps churn prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_churn_dataset(
    n_rows: int = 2000,
    seed: int = 42,
    corrupt_null_ratio: float = 0.0,
) -> pd.DataFrame:
    """Generate synthetic customer churn dataset.

    Args:
        n_rows: Number of records to generate.
        seed: Random seed for reproducibility.
        corrupt_null_ratio: Fraction of missing values to inject into feature columns.
    """
    rng = np.random.default_rng(seed)

    customer_ids = [f"CUST-{idx:05d}" for idx in range(1, n_rows + 1)]
    tenure_months = rng.integers(1, 72, size=n_rows)
    monthly_charges = np.round(rng.uniform(18.0, 120.0, size=n_rows), 2)
    contract_choices = ["month-to-month", "one-year", "two-year"]
    contract_p = [0.55, 0.25, 0.20]
    contract_type = rng.choice(contract_choices, size=n_rows, p=contract_p)

    internet_choices = ["dsl", "fiber-optic", "no"]
    internet_p = [0.4, 0.45, 0.15]
    internet_service = rng.choice(internet_choices, size=n_rows, p=internet_p)

    tech_support_choices = ["yes", "no"]
    tech_support = rng.choice(tech_support_choices, size=n_rows, p=[0.3, 0.7])

    num_support_tickets = rng.poisson(lam=1.5, size=n_rows)
    total_charges = np.round(tenure_months * monthly_charges * rng.uniform(0.9, 1.05, size=n_rows), 2)

    # Churn probability based on realistic business rules
    logit = (
        -1.5
        - 0.04 * tenure_months
        + 0.02 * monthly_charges
        + 0.4 * (contract_type == "month-to-month")
        - 0.8 * (contract_type == "two-year")
        + 0.5 * (internet_service == "fiber-optic")
        - 0.4 * (tech_support == "yes")
        + 0.35 * num_support_tickets
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.uniform(0, 1, size=n_rows) < prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "num_support_tickets": num_support_tickets,
            "churn": churn,
        }
    )

    # Optional injection of nulls for failure simulation
    if corrupt_null_ratio > 0.0:
        corrupt_cols = ["tenure_months", "monthly_charges", "total_charges"]
        for col in corrupt_cols:
            mask = rng.uniform(0, 1, size=n_rows) < corrupt_null_ratio
            df.loc[mask, col] = np.nan

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic churn dataset.")
    parser.add_argument("--rows", type=int, default=2000, help="Number of rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="data/dataset.csv",
        help="Target output CSV filepath",
    )
    parser.add_argument(
        "--corrupt-nulls",
        type=float,
        default=0.0,
        help="Fraction of null values to inject (e.g. 0.4 for 40% failure injection)",
    )

    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_churn_dataset(
        n_rows=args.rows,
        seed=args.seed,
        corrupt_null_ratio=args.corrupt_nulls,
    )
    df.to_csv(output_path, index=False)
    print(
        f"Generated {len(df)} rows with seed {args.seed} -> {output_path} "
        f"(Churn balance: {df['churn'].mean():.2%}, Null ratio: {df.isna().mean().max():.2%})"
    )


if __name__ == "__main__":
    main()
