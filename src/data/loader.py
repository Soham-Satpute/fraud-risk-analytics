"""
src/data/loader.py
------------------
Memory-optimized data loading pipeline for IEEE-CIS Fraud Detection dataset.

Responsibilities:
  - Load raw CSVs with dtype downcasting to reduce memory footprint (~2.5GB → ~700MB).
  - Left-join train_transaction with train_identity on TransactionID.
  - Persist merged dataset as columnar parquet in data/processed/ for fast downstream reads.
  - Provide a fast-path loader that reads from parquet if it already exists.

Raw data stays LOCAL (data/raw/) and is NEVER pushed to Postgres or committed to Git.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-level paths (resolved relative to this file's location)
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent          # src/
_PROJECT_ROOT = _SRC_DIR.parent                            # fraud-risk-analytics/
RAW_DIR = _PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"

TRAIN_TRANSACTION_CSV = RAW_DIR / "train_transaction.csv"
TRAIN_IDENTITY_CSV = RAW_DIR / "train_identity.csv"
MERGED_PARQUET = PROCESSED_DIR / "train_merged.parquet"


# ---------------------------------------------------------------------------
# Dtype maps for memory-efficient loading
# ---------------------------------------------------------------------------
# Specifying dtypes on read avoids pandas inferring float64 for everything.

_TRANSACTION_DTYPES: dict[str, str] = {
    "TransactionID": "int32",
    "isFraud": "int8",
    "TransactionDT": "int32",
    "TransactionAmt": "float32",
    "ProductCD": "category",
    "card1": "int16",
    "card2": "float32",
    "card3": "float32",
    "card4": "category",
    "card5": "float32",
    "card6": "category",
    "addr1": "float32",
    "addr2": "float32",
    "dist1": "float32",
    "dist2": "float32",
    "P_emaildomain": "category",
    "R_emaildomain": "category",
}

# C-features: counting columns (C1–C14)
_TRANSACTION_DTYPES.update({f"C{i}": "float32" for i in range(1, 15)})

# D-features: time-delta columns (D1–D15)
_TRANSACTION_DTYPES.update({f"D{i}": "float32" for i in range(1, 16)})

# M-features: match flags (M1–M9) — stored as category (True/False/NaN)
_TRANSACTION_DTYPES.update({f"M{i}": "category" for i in range(1, 10)})

# V-features: Vesta-engineered features (V1–V339)
_TRANSACTION_DTYPES.update({f"V{i}": "float32" for i in range(1, 340)})

_IDENTITY_DTYPES: dict[str, str] = {
    "TransactionID": "int32",
    "id_01": "float32",
    "id_02": "float32",
    "id_03": "float32",
    "id_04": "float32",
    "id_05": "float32",
    "id_06": "float32",
    "id_07": "float32",
    "id_08": "float32",
    "id_09": "float32",
    "id_10": "float32",
    "id_11": "float32",
    "id_12": "category",
    "id_13": "float32",
    "id_14": "float32",
    "id_15": "category",
    "id_16": "category",
    "id_17": "float32",
    "id_18": "float32",
    "id_19": "float32",
    "id_20": "float32",
    "id_21": "float32",
    "id_22": "float32",
    "id_23": "category",
    "id_24": "float32",
    "id_25": "float32",
    "id_26": "float32",
    "id_27": "category",
    "id_28": "category",
    "id_29": "category",
    "id_30": "category",
    "id_31": "category",
    "id_32": "float32",
    "id_33": "category",
    "id_34": "category",
    "id_35": "category",
    "id_36": "category",
    "id_37": "category",
    "id_38": "category",
    "DeviceType": "category",
    "DeviceInfo": "category",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_transactions(path: Path = TRAIN_TRANSACTION_CSV) -> pd.DataFrame:
    """
    Load raw train_transaction.csv with memory-optimized dtypes.

    Returns a DataFrame with TransactionID as a plain column (not index),
    preserving the original row order.
    """
    logger.info("Loading transaction CSV: %s", path)
    # Only apply dtypes that are actually in the file; ignore missing columns silently.
    available_cols = pd.read_csv(path, nrows=0).columns.tolist()
    dtypes = {k: v for k, v in _TRANSACTION_DTYPES.items() if k in available_cols}

    df = pd.read_csv(path, dtype=dtypes, low_memory=False)
    logger.info(
        "Transactions loaded — shape: %s, memory: %.1f MB",
        df.shape,
        df.memory_usage(deep=True).sum() / 1e6,
    )
    return df


def load_raw_identity(path: Path = TRAIN_IDENTITY_CSV) -> pd.DataFrame:
    """
    Load raw train_identity.csv with memory-optimized dtypes.
    """
    logger.info("Loading identity CSV: %s", path)
    available_cols = pd.read_csv(path, nrows=0).columns.tolist()
    dtypes = {k: v for k, v in _IDENTITY_DTYPES.items() if k in available_cols}

    df = pd.read_csv(path, dtype=dtypes, low_memory=False)
    logger.info(
        "Identity loaded — shape: %s, memory: %.1f MB",
        df.shape,
        df.memory_usage(deep=True).sum() / 1e6,
    )
    return df


def merge_train(
    transactions: Optional[pd.DataFrame] = None,
    identity: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Left-join transactions with identity on TransactionID.

    Transactions that have no corresponding identity record retain NaN
    for all identity columns (expected — only ~23% of train rows have identity).

    Args:
        transactions: Pre-loaded transaction DataFrame. Loaded from CSV if None.
        identity:     Pre-loaded identity DataFrame. Loaded from CSV if None.

    Returns:
        Merged DataFrame.
    """
    if transactions is None:
        transactions = load_raw_transactions()
    if identity is None:
        identity = load_raw_identity()

    logger.info("Merging transactions (%d rows) with identity (%d rows)…", len(transactions), len(identity))
    merged = transactions.merge(identity, on="TransactionID", how="left")
    logger.info(
        "Merged — shape: %s, memory: %.1f MB",
        merged.shape,
        merged.memory_usage(deep=True).sum() / 1e6,
    )
    return merged


def build_parquet_cache(overwrite: bool = False) -> pd.DataFrame:
    """
    Build (or load) the merged parquet cache in data/processed/.

    This is the fast-path for all downstream analysis:
      - First call: reads raw CSVs, merges, writes parquet (~10-15 min on first run).
      - Subsequent calls: reads parquet directly (~10-15 sec).

    Args:
        overwrite: Force rebuild even if parquet already exists.

    Returns:
        Merged training DataFrame.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if MERGED_PARQUET.exists() and not overwrite:
        logger.info("Parquet cache found — loading from %s", MERGED_PARQUET)
        df = pd.read_parquet(MERGED_PARQUET)
        logger.info(
            "Loaded from cache — shape: %s, memory: %.1f MB",
            df.shape,
            df.memory_usage(deep=True).sum() / 1e6,
        )
        return df

    logger.info("Building parquet cache (first run — this will take a few minutes)…")
    df = merge_train()
    df.to_parquet(MERGED_PARQUET, index=False, engine="pyarrow", compression="snappy")
    logger.info("Parquet cache written to %s  (%.1f MB)", MERGED_PARQUET, MERGED_PARQUET.stat().st_size / 1e6)
    return df


def memory_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a per-column memory usage report, sorted descending.

    Useful for identifying which dtypes are still consuming excess memory.
    """
    mem = df.memory_usage(deep=True).reset_index()
    mem.columns = ["column", "bytes"]
    mem["MB"] = (mem["bytes"] / 1e6).round(3)
    mem["dtype"] = mem["column"].map(lambda c: str(df[c].dtype) if c in df.columns else "index")
    return mem.sort_values("bytes", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI entry-point: python -m src.data.loader
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    df = build_parquet_cache()
    print("\n=== Memory Report (top 20 columns) ===")
    print(memory_report(df).head(20).to_string(index=False))
    print(f"\nTotal rows   : {len(df):,}")
    print(f"Total columns: {df.shape[1]:,}")
    print(f"Fraud rate   : {df['isFraud'].mean():.4%}")
    print(f"Identity join: {df['DeviceType'].notna().mean():.1%} of rows have identity data")
