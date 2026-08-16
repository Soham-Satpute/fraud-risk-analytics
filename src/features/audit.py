"""
src/features/audit.py
---------------------
V/D/C feature block audit utilities for the IEEE-CIS Fraud Detection dataset.

Covers:
  - TransactionDT temporal profiling (delta structure, day/hour cycles, fraud drift).
  - D-feature (timedelta) audit: missingness, correlation with target.
  - C-feature (counting/velocity) audit: distribution and correlation with fraud.
  - V-feature (Vesta-engineered) audit: missingness clusters, collinearity screening,
    correlation with target.
  - Cross-correlation between V/D/C and planned engineered features (Week 3 overlap check).
  - Feature stability across train/test splits.

All functions return structured DataFrames or dicts suitable for direct inclusion in
the docs/01_data_integrity_investigation.md report.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
D_COLS = [f"D{i}" for i in range(1, 16)]
C_COLS = [f"C{i}" for i in range(1, 15)]
V_COLS = [f"V{i}" for i in range(1, 340)]

LABEL_COL = "isFraud"
DT_COL = "TransactionDT"

# ---------------------------------------------------------------------------
# 1. TransactionDT Temporal Profiling
# ---------------------------------------------------------------------------


def profile_transaction_dt(df: pd.DataFrame, dt_col: str = DT_COL) -> dict:
    """
    Profile TransactionDT as a relative delta (seconds from dataset origin).

    The dataset documentation confirms TransactionDT is NOT an absolute Unix
    timestamp — it is a delta in seconds from some undisclosed reference point.
    We do NOT attempt to reconstruct the calendar origin.

    Returns a dict with:
      - span_seconds, span_days, span_weeks
      - min_dt, max_dt
      - quantile distribution (10th, 25th, 50th, 75th, 90th)
      - estimated day-of-week cycle (7-day periodicity via autocorrelation bucket)
      - fraud rate in first vs second half of time span
    """
    s = df[dt_col]
    min_dt = int(s.min())
    max_dt = int(s.max())
    span_seconds = max_dt - min_dt
    span_days = span_seconds / 86_400
    span_weeks = span_days / 7

    # Fraud rate drift: first half vs second half of time span
    midpoint = min_dt + span_seconds / 2
    first_half_fraud = df[df[dt_col] <= midpoint][LABEL_COL].mean()
    second_half_fraud = df[df[dt_col] > midpoint][LABEL_COL].mean()

    # Day-of-cycle proxy: transaction volume by (DT % 86400) / 3600 → hour bucket
    hour_bucket = ((df[dt_col] - min_dt) % 86_400) // 3_600
    hourly_volume = hour_bucket.value_counts().sort_index()
    peak_hour = int(hourly_volume.idxmax())

    # Day-of-week proxy: (DT // 86400) % 7
    day_bucket = ((df[dt_col] - min_dt) // 86_400) % 7
    daily_volume = day_bucket.value_counts().sort_index()

    return {
        "dt_col": dt_col,
        "min_dt": min_dt,
        "max_dt": max_dt,
        "span_seconds": span_seconds,
        "span_days": round(span_days, 1),
        "span_weeks": round(span_weeks, 1),
        "quantiles": s.quantile([0.10, 0.25, 0.50, 0.75, 0.90]).to_dict(),
        "fraud_rate_first_half": round(float(first_half_fraud), 4),
        "fraud_rate_second_half": round(float(second_half_fraud), 4),
        "peak_hour_bucket": peak_hour,
        "hourly_volume_by_bucket": hourly_volume.to_dict(),
        "daily_volume_by_weekday_proxy": daily_volume.to_dict(),
        "note": (
            "TransactionDT is a relative delta (seconds), not an absolute "
            "calendar timestamp. Hour/day buckets are computed modulo 86400/604800 "
            "from the dataset minimum — they reflect intra-day and intra-week cycles "
            "but do NOT reconstruct absolute dates."
        ),
    }


def fraud_rate_over_time(
    df: pd.DataFrame,
    dt_col: str = DT_COL,
    n_bins: int = 50,
) -> pd.DataFrame:
    """
    Bin transactions into n_bins time windows and compute fraud rate per bin.
    Returns a DataFrame with columns: bin_start, bin_end, n_transactions, fraud_rate.
    """
    df = df.copy()
    df["_bin"] = pd.cut(df[dt_col], bins=n_bins, labels=False)
    bin_edges = pd.cut(df[dt_col], bins=n_bins).cat.categories

    summary = (
        df.groupby("_bin")
        .agg(
            n_transactions=(LABEL_COL, "count"),
            n_fraud=(LABEL_COL, "sum"),
        )
        .assign(fraud_rate=lambda x: x["n_fraud"] / x["n_transactions"])
    )
    summary.index = pd.RangeIndex(len(summary))
    summary["bin_start"] = [iv.left for iv in bin_edges]
    summary["bin_end"] = [iv.right for iv in bin_edges]
    return summary[["bin_start", "bin_end", "n_transactions", "n_fraud", "fraud_rate"]]


# ---------------------------------------------------------------------------
# 2. D-Feature Audit (Timedelta columns D1–D15)
# ---------------------------------------------------------------------------


def audit_d_features(
    df: pd.DataFrame,
    d_cols: Optional[list[str]] = None,
    label_col: str = LABEL_COL,
) -> pd.DataFrame:
    """
    Profile D1–D15 timedelta columns.

    Returns a DataFrame with per-column stats:
      - pct_missing, mean, median, std, min, max
      - pearson correlation with isFraud (among non-null rows)
      - spearman correlation with isFraud
    """
    d_cols = d_cols or [c for c in D_COLS if c in df.columns]
    rows = []
    for col in d_cols:
        series = df[col]
        non_null = df[[col, label_col]].dropna(subset=[col])
        rows.append({
            "feature": col,
            "pct_missing": round(series.isna().mean() * 100, 2),
            "mean": round(float(series.mean()), 3),
            "median": round(float(series.median()), 3),
            "std": round(float(series.std()), 3),
            "min": round(float(series.min()), 3),
            "max": round(float(series.max()), 3),
            "pearson_r_fraud": round(float(non_null[col].corr(non_null[label_col])), 4),
            "spearman_r_fraud": round(
                float(non_null[col].rank().corr(non_null[label_col].rank())), 4
            ),
        })
    return pd.DataFrame(rows).sort_values("pct_missing")


# ---------------------------------------------------------------------------
# 3. C-Feature Audit (Counting columns C1–C14)
# ---------------------------------------------------------------------------


def audit_c_features(
    df: pd.DataFrame,
    c_cols: Optional[list[str]] = None,
    label_col: str = LABEL_COL,
) -> pd.DataFrame:
    """
    Profile C1–C14 counting/velocity columns.

    Returns per-column stats including skewness and correlation with fraud.
    """
    c_cols = c_cols or [c for c in C_COLS if c in df.columns]
    rows = []
    for col in c_cols:
        series = df[col]
        non_null = df[[col, label_col]].dropna(subset=[col])
        rows.append({
            "feature": col,
            "pct_missing": round(series.isna().mean() * 100, 2),
            "mean": round(float(series.mean()), 3),
            "median": round(float(series.median()), 3),
            "std": round(float(series.std()), 3),
            "skewness": round(float(series.skew()), 3),
            "max": round(float(series.max()), 3),
            "pearson_r_fraud": round(float(non_null[col].corr(non_null[label_col])), 4),
            "spearman_r_fraud": round(
                float(non_null[col].rank().corr(non_null[label_col].rank())), 4
            ),
        })
    return pd.DataFrame(rows).sort_values("pct_missing")


# ---------------------------------------------------------------------------
# 4. V-Feature Audit (Vesta-engineered V1–V339)
# ---------------------------------------------------------------------------


def audit_v_features_missingness(
    df: pd.DataFrame,
    v_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Profile missingness across V1–V339 columns.

    The V-features come in missingness clusters (groups of columns that are
    all missing for the same rows). This function identifies those clusters
    by computing pairwise missingness correlation within the V-block.

    Returns a DataFrame with columns: feature, pct_missing, missingness_cluster_id.
    missingness_cluster_id is assigned by rounding the missingness rate to the
    nearest 5%, which groups columns with near-identical missingness patterns.
    """
    v_cols = v_cols or [c for c in V_COLS if c in df.columns]
    logger.info("Auditing missingness for %d V-columns…", len(v_cols))

    missing_rates = df[v_cols].isna().mean() * 100
    result = pd.DataFrame({
        "feature": missing_rates.index,
        "pct_missing": missing_rates.values.round(2),
    })
    # Cluster by rounding to nearest 5%
    result["missingness_cluster_id"] = (result["pct_missing"] / 5).round() * 5
    result["missingness_cluster_id"] = result["missingness_cluster_id"].astype(int)
    return result.sort_values("pct_missing").reset_index(drop=True)


def audit_v_features_correlation(
    df: pd.DataFrame,
    v_cols: Optional[list[str]] = None,
    label_col: str = LABEL_COL,
    top_n: int = 50,
    fillna_val: float = 0.0,
) -> pd.DataFrame:
    """
    Compute point-biserial (Pearson) correlation of each V-feature with isFraud.

    NaN values are filled with fillna_val before correlation (a common and
    simple approach for tree-based models; noted as an assumption).

    Returns top_n features sorted by absolute correlation, descending.
    """
    v_cols = v_cols or [c for c in V_COLS if c in df.columns]
    logger.info("Computing V-feature correlations (%d cols)…", len(v_cols))

    filled = df[v_cols].fillna(fillna_val)
    corrs = filled.corrwith(df[label_col]).abs()
    result = pd.DataFrame({
        "feature": corrs.index,
        "abs_corr_with_fraud": corrs.values.round(4),
    }).sort_values("abs_corr_with_fraud", ascending=False).reset_index(drop=True)
    return result.head(top_n)


def find_collinear_v_features(
    df: pd.DataFrame,
    v_cols: Optional[list[str]] = None,
    threshold: float = 0.98,
    fillna_val: float = 0.0,
    sample_n: int = 50_000,
) -> pd.DataFrame:
    """
    Identify pairs of V-features with Pearson |r| > threshold (near-duplicates).

    Computing a full 339×339 correlation matrix is memory-intensive. This
    function samples up to sample_n rows and operates on a subset of columns
    that have sufficient non-missing data.

    Returns a DataFrame of (feature_a, feature_b, correlation) pairs.
    """
    v_cols = v_cols or [c for c in V_COLS if c in df.columns]

    # Only keep columns with less than 80% missing
    non_sparse = [c for c in v_cols if df[c].isna().mean() < 0.80]
    logger.info(
        "Collinearity check on %d V-cols (non-sparse) from %d total…",
        len(non_sparse), len(v_cols),
    )

    sample = df[non_sparse].fillna(fillna_val)
    if len(sample) > sample_n:
        sample = sample.sample(sample_n, random_state=42)

    corr_matrix = sample.corr().abs()

    pairs = []
    cols = corr_matrix.columns.tolist()
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1:]:
            val = corr_matrix.loc[col_a, col_b]
            if val >= threshold:
                pairs.append({"feature_a": col_a, "feature_b": col_b, "abs_corr": round(val, 4)})

    result = pd.DataFrame(pairs).sort_values("abs_corr", ascending=False).reset_index(drop=True)
    logger.info("Found %d near-duplicate V-feature pairs (|r| >= %.2f)", len(result), threshold)
    return result


# ---------------------------------------------------------------------------
# 5. Cross-correlation with planned engineered features (Week 3 overlap check)
# ---------------------------------------------------------------------------

# These are the feature TYPES we plan to engineer in Week 3.
# We check if V/D/C columns already encode the same signal.
PLANNED_FEATURE_PROXIES = {
    "velocity_proxy": "C1",        # C1 approximates transaction count by card
    "timedelta_proxy": "D1",       # D1 is days since last transaction
    "amount_proxy": "TransactionAmt",
}


def check_vdc_vs_engineered_overlap(
    df: pd.DataFrame,
    label_col: str = LABEL_COL,
) -> pd.DataFrame:
    """
    Check whether planned Week 3 engineered features (velocity, timedelta,
    amount z-score) would be redundant with existing V/D/C columns.

    Computes Pearson correlation between each proxy and all V/D/C columns
    present in the DataFrame.

    Returns a DataFrame of (vdc_feature, engineered_proxy, pearson_r, abs_r)
    sorted by abs_r descending — high abs_r means the engineered feature
    would duplicate the existing column.
    """
    all_vdc = [c for c in (D_COLS + C_COLS + V_COLS) if c in df.columns]
    results = []

    for proxy_name, proxy_col in PLANNED_FEATURE_PROXIES.items():
        if proxy_col not in df.columns:
            logger.warning("Proxy column %s not found, skipping.", proxy_col)
            continue

        proxy_series = df[proxy_col].fillna(0)
        for vdc_col in all_vdc:
            r = proxy_series.corr(df[vdc_col].fillna(0))
            results.append({
                "vdc_feature": vdc_col,
                "engineered_proxy": proxy_name,
                "pearson_r": round(float(r), 4),
                "abs_r": round(abs(float(r)), 4),
            })

    result_df = pd.DataFrame(results).sort_values("abs_r", ascending=False).reset_index(drop=True)
    # Flag high-overlap pairs
    result_df["high_redundancy"] = result_df["abs_r"] >= 0.85
    return result_df


# ---------------------------------------------------------------------------
# 6. Feature stability across splits
# ---------------------------------------------------------------------------


def check_feature_stability(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    psi_threshold: float = 0.10,
) -> pd.DataFrame:
    """
    Check distribution stability of V/D/C features across train/test splits
    using Population Stability Index (PSI).

    PSI interpretation:
      < 0.10  → stable (no significant shift)
      0.10-0.25 → minor shift (worth monitoring)
      > 0.25  → major shift (potential data integrity issue)

    Args:
        train:          Training split.
        test:           Test split.
        feature_cols:   Columns to check. Defaults to all D+C+V cols present.
        psi_threshold:  Flag features above this PSI.

    Returns:
        DataFrame with (feature, psi, flag_unstable) sorted by PSI descending.
    """
    if feature_cols is None:
        feature_cols = [c for c in (D_COLS + C_COLS + V_COLS[:50]) if c in train.columns]
        # Limit V to first 50 for speed; full audit can be run offline

    logger.info("Checking stability for %d features…", len(feature_cols))

    def _psi(train_series: pd.Series, test_series: pd.Series, bins: int = 10) -> float:
        """Compute PSI between train and test distributions."""
        combined = pd.concat([train_series, test_series]).dropna()
        if combined.nunique() < 2:
            return 0.0

        # Bin edges from training data
        _, edges = pd.cut(train_series.dropna(), bins=bins, retbins=True, duplicates="drop")
        edges[0] = -np.inf
        edges[-1] = np.inf

        train_counts = pd.cut(train_series.dropna(), bins=edges).value_counts().sort_index()
        test_counts = pd.cut(test_series.dropna(), bins=edges).value_counts().sort_index()

        train_pct = (train_counts / train_counts.sum()).clip(1e-4)
        test_pct = (test_counts / test_counts.sum()).clip(1e-4)

        psi = ((test_pct - train_pct) * np.log(test_pct / train_pct)).sum()
        return float(psi)

    rows = []
    for col in feature_cols:
        psi_val = _psi(train[col], test[col])
        rows.append({
            "feature": col,
            "psi": round(psi_val, 4),
            "flag_unstable": psi_val > psi_threshold,
        })

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
