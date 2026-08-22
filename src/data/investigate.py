"""
src/data/investigate.py
-----------------------
Week 1 investigation runner: orchestrates all data integrity analysis steps
and writes structured results to JSON files in data/processed/ for use by
the docs/ report and notebook.

Run with:
    python -m src.data.investigate

Outputs (all in data/processed/):
    investigation_dt_profile.json      — TransactionDT temporal profiling
    investigation_entity_summary.csv   — Entity distribution per proxy
    investigation_overlap_temporal.json  — Temporal split overlap metrics
    investigation_overlap_random.json    — Random split overlap metrics
    investigation_overlap_grouped.json   — Grouped split overlap metrics
    investigation_d_audit.csv          — D-feature audit table
    investigation_c_audit.csv          — C-feature audit table
    investigation_v_missingness.csv    — V-feature missingness clusters
    investigation_v_correlations.csv   — V-feature correlations with fraud (top-50)
    investigation_v_collinear.csv      — Near-duplicate V-feature pairs
    investigation_vdc_overlap.csv      — V/D/C vs planned engineered feature overlap
    investigation_stability_temporal.csv — Feature stability (temporal split)
    investigation_fraud_over_time.csv  — Fraud rate binned over time
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.data.loader import build_parquet_cache, PROCESSED_DIR
from src.features.identity import (
    build_entity_proxy,
    compute_entity_overlap,
    make_temporal_split,
    make_random_split,
    make_grouped_split,
    summarise_entity_distribution,
)
from src.features.audit import (
    profile_transaction_dt,
    fraud_rate_over_time,
    audit_d_features,
    audit_c_features,
    audit_v_features_missingness,
    audit_v_features_correlation,
    find_collinear_v_features,
    check_vdc_vs_engineered_overlap,
    check_feature_stability,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _save_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved: %s", path.name)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    logger.info("Saved: %s  (%d rows)", path.name, len(df))


def run_investigation() -> None:
    """Execute all Week 1 investigation steps end to end."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Load data (from parquet cache if available)
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1 — Load data")
    logger.info("=" * 60)
    df = build_parquet_cache()

    logger.info(
        "Dataset: %d rows × %d columns | fraud rate: %.4f%%",
        len(df), df.shape[1], df["isFraud"].mean() * 100,
    )
    logger.info(
        "Identity join coverage: %.1f%% of rows have identity data",
        df["DeviceType"].notna().mean() * 100,
    )

    # -----------------------------------------------------------------------
    # Step 2: TransactionDT temporal profiling
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2 — TransactionDT temporal profiling")
    logger.info("=" * 60)
    dt_profile = profile_transaction_dt(df)
    _save_json(dt_profile, PROCESSED_DIR / "investigation_dt_profile.json")

    fraud_time = fraud_rate_over_time(df, n_bins=50)
    _save_csv(fraud_time, PROCESSED_DIR / "investigation_fraud_over_time.csv")

    logger.info(
        "TransactionDT: range [%d, %d] | span: %.1f days (%.1f weeks)",
        dt_profile["min_dt"], dt_profile["max_dt"],
        dt_profile["span_days"], dt_profile["span_weeks"],
    )
    logger.info(
        "Fraud rate drift: first half %.4f | second half %.4f",
        dt_profile["fraud_rate_first_half"], dt_profile["fraud_rate_second_half"],
    )

    # -----------------------------------------------------------------------
    # Step 3: Approximate entity proxy construction
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3 — Entity proxy construction & split overlap analysis")
    logger.info("=" * 60)
    df = build_entity_proxy(df)

    entity_summary = summarise_entity_distribution(df)
    _save_csv(entity_summary, PROCESSED_DIR / "investigation_entity_summary.csv")

    n_unique_entities = df["client_proxy_id"].nunique()
    avg_txn_per_entity = len(df) / n_unique_entities
    multi_txn_entities = (entity_summary["n_transactions"] > 1).sum()
    logger.info(
        "Entity proxy: %d unique proxies | avg %.1f txns/entity | %d entities with >1 txn",
        n_unique_entities, avg_txn_per_entity, multi_txn_entities,
    )

    # 3a. Temporal split
    train_t, test_t = make_temporal_split(df)
    overlap_temporal = compute_entity_overlap(train_t, test_t)
    overlap_temporal["split_strategy"] = "temporal_80_20"
    _save_json(overlap_temporal, PROCESSED_DIR / "investigation_overlap_temporal.json")

    # 3b. Random split
    train_r, test_r = make_random_split(df)
    overlap_random = compute_entity_overlap(train_r, test_r)
    overlap_random["split_strategy"] = "random_80_20_stratified"
    _save_json(overlap_random, PROCESSED_DIR / "investigation_overlap_random.json")

    # 3c. Grouped split
    train_g, test_g = make_grouped_split(df)
    overlap_grouped = compute_entity_overlap(train_g, test_g)
    overlap_grouped["split_strategy"] = "grouped_entity_kfold"
    _save_json(overlap_grouped, PROCESSED_DIR / "investigation_overlap_grouped.json")

    logger.info(
        "Overlap summary — temporal: %.1f%% | random: %.1f%% | grouped: %.1f%%",
        overlap_temporal["overlap_pct"],
        overlap_random["overlap_pct"],
        overlap_grouped["overlap_pct"],
    )

    # -----------------------------------------------------------------------
    # Step 4: V/D/C feature block audits
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4 — V/D/C feature block audits")
    logger.info("=" * 60)

    d_audit = audit_d_features(df)
    _save_csv(d_audit, PROCESSED_DIR / "investigation_d_audit.csv")

    c_audit = audit_c_features(df)
    _save_csv(c_audit, PROCESSED_DIR / "investigation_c_audit.csv")

    v_missing = audit_v_features_missingness(df)
    _save_csv(v_missing, PROCESSED_DIR / "investigation_v_missingness.csv")

    v_corr = audit_v_features_correlation(df, top_n=50)
    _save_csv(v_corr, PROCESSED_DIR / "investigation_v_correlations.csv")

    v_collinear = find_collinear_v_features(df, threshold=0.98)
    _save_csv(v_collinear, PROCESSED_DIR / "investigation_v_collinear.csv")

    n_clusters = v_missing["missingness_cluster_id"].nunique()
    logger.info(
        "V-features: %d present | %d missingness clusters | %d near-duplicate pairs (|r|≥0.98)",
        len([c for c in df.columns if c.startswith("V")]),
        n_clusters,
        len(v_collinear),
    )

    # -----------------------------------------------------------------------
    # Step 5: Cross-correlation with planned engineered features
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5 — Cross-correlation: V/D/C vs planned engineered features")
    logger.info("=" * 60)
    vdc_overlap = check_vdc_vs_engineered_overlap(df)
    _save_csv(vdc_overlap, PROCESSED_DIR / "investigation_vdc_overlap.csv")

    high_redundancy = vdc_overlap[vdc_overlap["high_redundancy"]]
    logger.info(
        "%d V/D/C features are highly correlated (|r|≥0.85) with planned engineered features",
        len(high_redundancy),
    )
    if not high_redundancy.empty:
        logger.info("Top redundant pairs:\n%s", high_redundancy.head(10).to_string(index=False))

    # -----------------------------------------------------------------------
    # Step 6: Feature stability under temporal split
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 6 — Feature stability (temporal split)")
    logger.info("=" * 60)
    stability = check_feature_stability(train_t, test_t)
    _save_csv(stability, PROCESSED_DIR / "investigation_stability_temporal.csv")

    n_unstable = stability["flag_unstable"].sum()
    logger.info(
        "%d / %d features flagged as unstable (PSI > 0.10) under temporal split",
        n_unstable, len(stability),
    )

    # -----------------------------------------------------------------------
    # Final summary for report
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("INVESTIGATION COMPLETE — Summary")
    logger.info("=" * 60)

    summary = {
        "total_rows": len(df),
        "total_columns": df.shape[1],
        "fraud_rate_pct": round(df["isFraud"].mean() * 100, 4),
        "identity_join_coverage_pct": round(df["DeviceType"].notna().mean() * 100, 1),
        "transactionDT_span_days": dt_profile["span_days"],
        "transactionDT_span_weeks": dt_profile["span_weeks"],
        "fraud_rate_first_half": dt_profile["fraud_rate_first_half"],
        "fraud_rate_second_half": dt_profile["fraud_rate_second_half"],
        "unique_entity_proxies": int(n_unique_entities),
        "avg_transactions_per_proxy": round(avg_txn_per_entity, 1),
        "entity_overlap_temporal_pct": overlap_temporal["overlap_pct"],
        "entity_overlap_random_pct": overlap_random["overlap_pct"],
        "entity_overlap_grouped_pct": overlap_grouped["overlap_pct"],
        "fraud_rate_recurring_entities_temporal": overlap_temporal["fraud_rate_recurring_entities"],
        "fraud_rate_new_entities_temporal": overlap_temporal["fraud_rate_new_entities"],
        "v_missingness_clusters": int(n_clusters),
        "v_near_duplicate_pairs": len(v_collinear),
        "vdc_high_redundancy_with_engineered": int(len(high_redundancy)),
        "features_unstable_psi_temporal": int(n_unstable),
    }
    _save_json(summary, PROCESSED_DIR / "investigation_summary.json")

    print("\n" + "=" * 60)
    print("INVESTIGATION SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<45} {v}")
    print("\nAll outputs saved to:", PROCESSED_DIR)


if __name__ == "__main__":
    run_investigation()
