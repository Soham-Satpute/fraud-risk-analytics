"""
src/features/identity.py
------------------------
Approximate client entity reconstruction for IEEE-CIS Fraud Detection data.

IMPORTANT CAVEAT (must be stated wherever this output is used):
  The entity identifiers produced here are APPROXIMATE, CORRELATION-BASED
  PROXIES — not confirmed ground-truth client keys. The IEEE-CIS dataset
  contains no unique customer ID. We infer probable client groupings from
  correlated card/address/email attributes. A single real customer may map
  to multiple proxy keys (e.g., address change, new card). A single proxy
  key may occasionally merge distinct customers with coincidentally identical
  attributes. All downstream analysis must acknowledge this limitation.

Methodology:
  - Group transactions by a composite key of card and address attributes.
  - Use a stable hash of the composite key as a surrogate "client_id".
  - Report entity distribution and overlap across temporal/random splits.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Proxy identity construction
# ---------------------------------------------------------------------------

# Columns used to construct the approximate client proxy.
# Ordered by specificity: card attributes first (most unique),
# then geographic context, then email domains.
PROXY_COLS = [
    "card1",   # card ID component — most discriminative
    "card2",   # card BIN / issuer detail
    "card3",   # card type detail
    "card5",   # card product subtype
    "addr1",   # billing zip/region
    "addr2",   # billing country code
    "P_emaildomain",  # purchaser email domain
]


def build_entity_proxy(
    df: pd.DataFrame,
    cols: list[str] = PROXY_COLS,
    proxy_col: str = "client_proxy_id",
) -> pd.DataFrame:
    """
    Add an approximate client proxy identifier column to the DataFrame.

    The proxy is a stable MD5 hash of the concatenated (str) values of
    the specified columns. NaN values are represented as the string "NA"
    so they contribute to grouping (a card with consistent NaN addr is
    still a consistent group).

    Args:
        df:         Input DataFrame (train or test transactions).
        cols:       Columns to include in the composite key.
        proxy_col:  Name of the new column to add.

    Returns:
        DataFrame with proxy_col added (original DataFrame is NOT mutated;
        a copy with only the proxy column appended is returned — use
        pd.concat or df.assign externally if needed).
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning("Proxy cols not found in DataFrame, skipping: %s", missing)
        cols = [c for c in cols if c in df.columns]

    if not cols:
        raise ValueError("No valid proxy columns found in DataFrame.")

    logger.info("Building entity proxy from columns: %s", cols)

    # Build composite string key per row.
    # Category columns in pandas 3.x reject fillna with a value not in their
    # category list. Convert to object dtype first so all columns accept "NA".
    key_series = (
        df[cols]
        .astype(object)        # strips category dtype, allows any fill value
        .fillna("NA")
        .astype(str)
        .apply(lambda row: "|".join(row.values), axis=1)
    )

    # Stable hash → integer proxy ID (first 8 hex chars → int)
    proxy_ids = key_series.apply(
        lambda s: int(hashlib.md5(s.encode()).hexdigest()[:8], 16)
    )

    result = df.copy()
    result[proxy_col] = proxy_ids.astype("int64")
    n_unique = result[proxy_col].nunique()
    logger.info(
        "Entity proxy built — %d unique proxy IDs across %d transactions "
        "(avg %.1f transactions per proxy)",
        n_unique,
        len(result),
        len(result) / n_unique,
    )
    return result


# ---------------------------------------------------------------------------
# Split construction & overlap analysis
# ---------------------------------------------------------------------------

SplitStrategy = Literal["temporal", "random", "grouped"]


def make_temporal_split(
    df: pd.DataFrame,
    dt_col: str = "TransactionDT",
    train_frac: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split: first `train_frac` of the time span → train,
    remainder → test.

    Preserves original row order within each split. Does NOT shuffle.

    Args:
        df:         Merged DataFrame with TransactionDT column.
        dt_col:     Temporal column (delta seconds from dataset origin).
        train_frac: Fraction of the time span to allocate to train.

    Returns:
        (train_df, test_df)
    """
    threshold = df[dt_col].quantile(train_frac)
    train = df[df[dt_col] <= threshold].copy()
    test = df[df[dt_col] > threshold].copy()
    logger.info(
        "Temporal split — threshold DT=%.0f | train: %d rows | test: %d rows",
        threshold, len(train), len(test),
    )
    return train, test


def make_random_split(
    df: pd.DataFrame,
    train_frac: float = 0.80,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Random 80/20 split (stratified on isFraud to preserve fraud rate).

    Args:
        df:           Input DataFrame.
        train_frac:   Fraction of rows for training.
        random_state: Fixed seed for reproducibility.

    Returns:
        (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        df,
        train_size=train_frac,
        stratify=df["isFraud"],
        random_state=random_state,
    )
    logger.info("Random split — train: %d rows | test: %d rows", len(train), len(test))
    return train.copy(), test.copy()


def make_grouped_split(
    df: pd.DataFrame,
    proxy_col: str = "client_proxy_id",
    n_splits: int = 5,
    test_fold: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    GroupKFold split ensuring that all transactions from a given proxy
    client appear in either train OR test — not both.

    This simulates the hardest evaluation scenario: the model must
    generalize to clients it has never seen.

    Args:
        df:           DataFrame with proxy_col populated.
        proxy_col:    Column containing entity proxy IDs.
        n_splits:     Number of GroupKFold splits (test ≈ 1/n_splits of data).
        test_fold:    Which fold to use as the held-out test set.
        random_state: Not used by GroupKFold directly, but kept for API consistency.

    Returns:
        (train_df, test_df)
    """
    if proxy_col not in df.columns:
        raise ValueError(f"proxy_col '{proxy_col}' not found. Run build_entity_proxy first.")

    gkf = GroupKFold(n_splits=n_splits)
    groups = df[proxy_col].values
    X_dummy = np.zeros(len(df))

    splits = list(gkf.split(X_dummy, groups=groups))
    train_idx, test_idx = splits[test_fold]

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()
    logger.info(
        "Grouped split — train: %d rows | test: %d rows",
        len(train), len(test),
    )
    return train, test


# ---------------------------------------------------------------------------
# Overlap metrics
# ---------------------------------------------------------------------------

def compute_entity_overlap(
    train: pd.DataFrame,
    test: pd.DataFrame,
    proxy_col: str = "client_proxy_id",
    label_col: str = "isFraud",
) -> dict:
    """
    Quantify how many test-set entity proxies also appear in the training set,
    and whether overlapping entities have a different fraud rate than new ones.

    Args:
        train:     Training split DataFrame (with proxy_col).
        test:      Test split DataFrame (with proxy_col).
        proxy_col: Entity proxy column name.
        label_col: Fraud label column (0/1).

    Returns:
        Dictionary of overlap metrics with descriptive keys.
    """
    train_entities = set(train[proxy_col].unique())
    test_entities = set(test[proxy_col].unique())
    overlap = train_entities & test_entities

    n_test_entities = len(test_entities)
    n_overlap = len(overlap)
    pct_overlap = n_overlap / n_test_entities * 100 if n_test_entities > 0 else 0.0

    # Fraud rate: recurring entities vs new entities in test set
    test_recurring = test[test[proxy_col].isin(overlap)]
    test_new = test[~test[proxy_col].isin(overlap)]

    fraud_rate_recurring = test_recurring[label_col].mean() if len(test_recurring) > 0 else float("nan")
    fraud_rate_new = test_new[label_col].mean() if len(test_new) > 0 else float("nan")
    fraud_rate_overall = test[label_col].mean()

    metrics = {
        "train_unique_entities": len(train_entities),
        "test_unique_entities": n_test_entities,
        "overlapping_entities": n_overlap,
        "overlap_pct": round(pct_overlap, 2),
        "test_rows_recurring_entity": len(test_recurring),
        "test_rows_new_entity": len(test_new),
        "fraud_rate_overall_test": round(float(fraud_rate_overall), 4),
        "fraud_rate_recurring_entities": round(float(fraud_rate_recurring), 4),
        "fraud_rate_new_entities": round(float(fraud_rate_new), 4),
    }

    logger.info(
        "Overlap: %d / %d test entities seen in train (%.1f%%)",
        n_overlap, n_test_entities, pct_overlap,
    )
    logger.info(
        "Fraud rate — overall: %.3f | recurring: %.3f | new: %.3f",
        fraud_rate_overall, fraud_rate_recurring, fraud_rate_new,
    )
    return metrics


def summarise_entity_distribution(
    df: pd.DataFrame,
    proxy_col: str = "client_proxy_id",
    label_col: str = "isFraud",
) -> pd.DataFrame:
    """
    Per-entity summary: transaction count, fraud count, fraud rate.
    Useful for understanding entity-level fraud concentration.

    Returns a DataFrame indexed by proxy_col, sorted by transaction count desc.
    """
    summary = (
        df.groupby(proxy_col)
        .agg(
            n_transactions=(label_col, "count"),
            n_fraud=(label_col, "sum"),
        )
        .assign(fraud_rate=lambda x: x["n_fraud"] / x["n_transactions"])
        .sort_values("n_transactions", ascending=False)
        .reset_index()
    )
    return summary
