"""
src/features/build_features.py
------------------------------
Execution script to generate leakage-free feature sets for training and evaluation.

Workflow:
  1. Load data/processed/train_merged.parquet (590,540 rows x 434 cols).
  2. Partition temporally at TransactionDT <= 12,192,854 (Train: 472,432 | Test: 118,108).
  3. Fit FraudFeaturePipeline on Train partition ONLY.
  4. Transform Train and Test partitions independently.
  5. Audit correlations of engineered features against target (isFraud) and against C1/D1.
  6. Persist:
     - data/processed/train_features.parquet
     - data/processed/test_features.parquet
     - models/feature_pipeline.joblib
     - data/processed/feature_metadata.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Project-level paths
_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.engineer import FraudFeaturePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = _PROJECT_ROOT / "data" / "processed"
MODELS_DIR = _PROJECT_ROOT / "models"

MERGED_PARQUET = DATA_DIR / "train_merged.parquet"
TRAIN_FEATURES_PARQUET = DATA_DIR / "train_features.parquet"
TEST_FEATURES_PARQUET = DATA_DIR / "test_features.parquet"
PIPELINE_JOBLIB = MODELS_DIR / "feature_pipeline.joblib"
FEATURE_METADATA_JSON = DATA_DIR / "feature_metadata.json"

# Locked temporal split threshold (80th percentile from Week 1 audit)
TEMPORAL_SPLIT_CUTOFF_DT: int = 12192854


def run_feature_build() -> None:
    start_time = time.time()
    logger.info("Starting Week 3 Feature Engineering Build...")

    if not MERGED_PARQUET.exists():
        logger.error("Dataset not found at %s. Please run Week 1 loader first.", MERGED_PARQUET)
        sys.exit(1)

    # 1. Load full merged dataset
    logger.info("Loading merged dataset from %s...", MERGED_PARQUET)
    df = pd.read_parquet(MERGED_PARQUET)
    total_rows, total_cols = df.shape
    logger.info("Loaded %d rows x %d columns.", total_rows, total_cols)

    # 2. Enforce temporal partition
    train_mask = df["TransactionDT"] <= TEMPORAL_SPLIT_CUTOFF_DT
    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df = df[~train_mask].copy().reset_index(drop=True)

    logger.info(
        "Temporal Split applied at DT <= %d: Train=%d rows (%.1f%%) | Test=%d rows (%.1f%%)",
        TEMPORAL_SPLIT_CUTOFF_DT,
        len(train_df),
        (len(train_df) / total_rows) * 100,
        len(test_df),
        (len(test_df) / total_rows) * 100,
    )

    # 3. Fit pipeline strictly on train partition
    pipeline = FraudFeaturePipeline()
    pipeline.fit(train_df)

    # 4. Transform train and test independently
    logger.info("Transforming train partition...")
    train_feat_df = pipeline.transform(train_df)

    logger.info("Transforming test partition with frozen train lookups...")
    test_feat_df = pipeline.transform(test_df)

    engineered_cols = pipeline.engineered_feature_names
    logger.info("Successfully engineered %d new features: %s", len(engineered_cols), engineered_cols)

    # 5. Cross-Correlation and Redundancy Audit
    logger.info("Auditing correlations of newly engineered features...")
    audit_results: dict[str, dict[str, float]] = {}

    for feat in engineered_cols:
        if feat in train_feat_df.columns:
            feat_series = train_feat_df[feat].astype(float)
            target_series = train_feat_df["isFraud"].astype(float)
            c1_series = train_feat_df["C1"].astype(float)
            d1_series = train_feat_df["D1"].astype(float)

            # Pearson correlation with target
            corr_target = float(feat_series.corr(target_series))
            # Correlation with C1 (velocity) and D1 (time-since-last)
            corr_c1 = float(feat_series.corr(c1_series)) if not feat_series.isna().all() else 0.0
            corr_d1 = float(feat_series.corr(d1_series)) if not feat_series.isna().all() else 0.0

            audit_results[feat] = {
                "corr_with_isFraud": round(corr_target, 4) if not np.isnan(corr_target) else 0.0,
                "corr_with_C1": round(corr_c1, 4) if not np.isnan(corr_c1) else 0.0,
                "corr_with_D1": round(corr_d1, 4) if not np.isnan(corr_d1) else 0.0,
            }

    # 6. Save parquets and artifacts
    logger.info("Saving transformed datasets to %s...", DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_feat_df.to_parquet(TRAIN_FEATURES_PARQUET, index=False)
    test_feat_df.to_parquet(TEST_FEATURES_PARQUET, index=False)
    pipeline.save(PIPELINE_JOBLIB)

    train_size_mb = TRAIN_FEATURES_PARQUET.stat().st_size / (1024 * 1024)
    test_size_mb = TEST_FEATURES_PARQUET.stat().st_size / (1024 * 1024)

    # 7. Write metadata
    metadata = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_source_rows": total_rows,
        "temporal_split_cutoff_dt": TEMPORAL_SPLIT_CUTOFF_DT,
        "train_rows": len(train_feat_df),
        "test_rows": len(test_feat_df),
        "train_fraud_rate": float((train_feat_df["isFraud"] == 1).mean()),
        "test_fraud_rate": float((test_feat_df["isFraud"] == 1).mean()),
        "total_feature_count": len(train_feat_df.columns),
        "engineered_feature_count": len(engineered_cols),
        "engineered_features": engineered_cols,
        "correlation_audit": audit_results,
        "artifacts": {
            "train_features_parquet": str(TRAIN_FEATURES_PARQUET.name),
            "train_size_mb": round(train_size_mb, 2),
            "test_features_parquet": str(TEST_FEATURES_PARQUET.name),
            "test_size_mb": round(test_size_mb, 2),
            "pipeline_joblib": str(PIPELINE_JOBLIB.name),
        },
    }

    with open(FEATURE_METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print("  WEEK 3 FEATURE ENGINEERING SUMMARY")
    print("=" * 75)
    print(f" Train Rows:           {len(train_feat_df):,} ({train_size_mb:.1f} MB)")
    print(f" Test Rows:            {len(test_feat_df):,} ({test_size_mb:.1f} MB)")
    print(f" Total Features:       {len(train_feat_df.columns)} ({len(engineered_cols)} newly engineered)")
    print(f" Pipeline Artifact:    {PIPELINE_JOBLIB}")
    print(f" Metadata JSON:        {FEATURE_METADATA_JSON}")
    print(f" Elapsed Build Time:   {elapsed:.1f}s")
    print("-" * 75)
    print(f" {'Feature':<28} {'Corr(isFraud)':<15} {'Corr(C1)':<12} {'Corr(D1)':<12}")
    print("-" * 75)
    for feat, metrics in audit_results.items():
        print(f" {feat:<28} {metrics['corr_with_isFraud']:<15} {metrics['corr_with_C1']:<12} {metrics['corr_with_D1']:<12}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_feature_build()
