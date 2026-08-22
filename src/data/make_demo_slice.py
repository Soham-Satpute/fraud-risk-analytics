"""
src/data/make_demo_slice.py
---------------------------
Extracts a lightweight, curated held-out test transaction slice (~1,500 rows)
from the temporal evaluation split (TransactionDT > 12,192,854).

Purpose:
  - Powers the simulated real-time inference feed in the FastAPI backend & Next.js frontend.
  - Generates seed data for the PostgreSQL `demo_replay` table without exceeding free-tier limits (< 2MB).
  - Maintains realistic temporal sequencing and representative fraud cases.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Paths
_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
MERGED_PARQUET = PROCESSED_DIR / "train_merged.parquet"
OUTPUT_PARQUET = PROCESSED_DIR / "demo_replay_slice.parquet"
OUTPUT_JSON = PROCESSED_DIR / "demo_replay_slice.json"

# Temporal test split cutoff (80th percentile from Week 1 audit)
TEMPORAL_CUTOFF_DT = 12192854


def create_demo_slice(
    merged_parquet_path: Path = MERGED_PARQUET,
    total_samples: int = 1500,
    fraud_target_ratio: float = 0.15,  # Slightly enriched (15%) for rich interactive demo experience
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Extract a curated sample of held-out test transactions.

    Parameters:
      merged_parquet_path: Path to the full merged parquet dataset.
      total_samples: Total number of transactions in the slice (default 1,500).
      fraud_target_ratio: Target fraud proportion to ensure sufficient positive examples in replay.
      random_state: Seed for deterministic sampling.

    Returns:
      pd.DataFrame: Curated slice sorted chronologically by TransactionDT.
    """
    if not merged_parquet_path.exists():
        raise FileNotFoundError(f"Merged dataset not found at {merged_parquet_path}")

    print(f"Loading held-out temporal partition from {merged_parquet_path}...")
    df = pd.read_parquet(merged_parquet_path)

    # Filter strictly for held-out temporal test set
    test_mask = df["TransactionDT"] > TEMPORAL_CUTOFF_DT
    test_df = df[test_mask].copy()

    total_test = len(test_df)
    print(f"Held-out temporal test rows: {total_test:,}")

    # Separate fraud and legit
    fraud_pool = test_df[test_df["isFraud"] == 1]
    legit_pool = test_df[test_df["isFraud"] == 0]

    n_fraud = int(total_samples * fraud_target_ratio)
    n_legit = total_samples - n_fraud

    n_fraud = min(n_fraud, len(fraud_pool))
    n_legit = min(n_legit, len(legit_pool))

    print(f"Sampling {n_fraud:,} fraud cases and {n_legit:,} legitimate cases...")
    sampled_fraud = fraud_pool.sample(n=n_fraud, random_state=random_state)
    sampled_legit = legit_pool.sample(n=n_legit, random_state=random_state)

    # Combine and sort chronologically by TransactionDT
    demo_df = pd.concat([sampled_fraud, sampled_legit], ignore_index=True)
    demo_df = demo_df.sort_values("TransactionDT").reset_index(drop=True)

    return demo_df


def save_demo_slice(demo_df: pd.DataFrame) -> None:
    """Persist slice to parquet and compact JSON format."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save parquet
    demo_df.to_parquet(OUTPUT_PARQUET, index=False)
    parquet_size_kb = OUTPUT_PARQUET.stat().st_size / 1024.0

    # 2. Select key preview columns for JSON seed format
    summary_cols = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card1",
        "card4",
        "card6",
        "P_emaildomain",
        "C1",
        "D1",
        "isFraud",
    ]
    available_cols = [c for c in summary_cols if c in demo_df.columns]
    summary_records = demo_df[available_cols].to_dict(orient="records")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)
    json_size_kb = OUTPUT_JSON.stat().st_size / 1024.0

    print("\n" + "=" * 60)
    print(" DEMO REPLAY SLICE GENERATED")
    print("=" * 60)
    print(f" Total Rows:       {len(demo_df):,}")
    print(f" Fraud Rate:       {(demo_df['isFraud'] == 1).mean():.2%}")
    print(f" Time Range (DT):  {demo_df['TransactionDT'].min():,} – {demo_df['TransactionDT'].max():,}")
    print(f" Parquet Size:     {parquet_size_kb:.1f} KB -> {OUTPUT_PARQUET}")
    print(f" JSON Seed Size:   {json_size_kb:.1f} KB -> {OUTPUT_JSON}")
    print("=" * 60 + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Generate curated held-out demo slice.")
    parser.add_argument("--samples", type=int, default=1500, help="Total sample count (default: 1500)")
    parser.add_argument("--fraud-ratio", type=float, default=0.15, help="Target fraud ratio (default: 0.15)")
    args = parser.parse_args()

    demo_df = create_demo_slice(total_samples=args.samples, fraud_target_ratio=args.fraud_ratio)
    save_demo_slice(demo_df)


if __name__ == "__main__":
    main()
