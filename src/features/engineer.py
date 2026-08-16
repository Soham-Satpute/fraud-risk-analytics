"""
src/features/engineer.py
------------------------
Leakage-free Feature Engineering Pipeline for IEEE-CIS Fraud Detection.

Strict Principles (enforcing Week 1 audit findings):
  1. DO NOT rebuild C1 (transaction velocity) or D1 (time-since-last-transaction),
     as they are identical (|r|=1.000) to planned hand-built features.
  2. Build high-value, non-redundant additive features:
     - Log-transformed transaction amount (compresses right-tail skewness)
     - Amount Z-scores, differences, and ratios by card and card+region groups
     - Amount Z-scores by email domain
     - Cyclical intraday (24h) and intraweek (7d) temporal encodings (sin/cos)
     - Frequency/count encodings for high-cardinality categorical entities
     - Payer vs. Recipient email domain consistency and missingness flags
  3. Strict Leakage Prevention:
     - All aggregation lookups and frequency dictionaries are fit STRICTLY
       on the training partition (TransactionDT <= 12,192,854).
     - Test and live inference batches are transformed using frozen training lookups.
     - Unseen categories in test receive deterministic fallbacks (e.g. freq=0, zscore=0).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Base reference timestamp (Day 1 offset in seconds)
REFERENCE_DT_ORIGIN: int = 86400

# High-cardinality & categorical columns for frequency encoding
FREQUENCY_ENCODE_COLUMNS: list[str] = [
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "ProductCD",
    "P_emaildomain",
    "R_emaildomain",
]


class FraudFeaturePipeline:
    """
    Scikit-learn style transformer for fraud feature engineering.
    Guarantees zero temporal leakage across train/test splits.
    """

    def __init__(self, dt_origin: int = REFERENCE_DT_ORIGIN) -> None:
        self.dt_origin = dt_origin
        self.is_fitted: bool = False

        # Fitted statistics containers (populated during fit())
        self.freq_lookups: dict[str, dict[Any, float]] = {}
        self.card1_amt_stats: dict[Any, tuple[float, float]] = {}       # card1 -> (mean, std)
        self.card1_addr1_amt_stats: dict[tuple[Any, Any], tuple[float, float]] = {} # (card1, addr1) -> (mean, std)
        self.email_amt_stats: dict[Any, tuple[float, float]] = {}       # P_emaildomain -> (mean, std)
        self.global_amt_mean: float = 0.0
        self.global_amt_std: float = 1.0

        # Feature column tracking
        self.engineered_feature_names: list[str] = []

    def fit(self, train_df: pd.DataFrame) -> FraudFeaturePipeline:
        """
        Fit aggregation lookups and frequency encodings on the training partition only.
        """
        logger.info("Fitting FraudFeaturePipeline on %d training rows...", len(train_df))

        # 1. Global TransactionAmt baseline
        amt_series = train_df["TransactionAmt"].astype(float)
        self.global_amt_mean = float(amt_series.mean())
        self.global_amt_std = float(amt_series.std(ddof=0))
        if self.global_amt_std == 0.0 or np.isnan(self.global_amt_std):
            self.global_amt_std = 1.0

        # 2. Frequency encodings (normalized to [0, 1] proportion)
        total_rows = float(len(train_df))
        self.freq_lookups = {}
        for col in FREQUENCY_ENCODE_COLUMNS:
            if col in train_df.columns:
                counts = train_df[col].value_counts(dropna=True)
                freq_dict = (counts / total_rows).to_dict()
                self.freq_lookups[col] = freq_dict

        # 3. Card1 Amount Statistics (mean, std)
        if "card1" in train_df.columns and "TransactionAmt" in train_df.columns:
            card1_grp = train_df.groupby("card1")["TransactionAmt"].agg(["mean", "std"])
            self.card1_amt_stats = {
                k: (float(m), float(s) if not np.isnan(s) and s > 0 else 0.0)
                for k, (m, s) in card1_grp.iterrows()
            }

        # 4. Card1 + Addr1 Amount Statistics (mean, std)
        if "card1" in train_df.columns and "addr1" in train_df.columns and "TransactionAmt" in train_df.columns:
            # Filter non-null addr1
            valid_ca = train_df.dropna(subset=["addr1"])
            ca_grp = valid_ca.groupby(["card1", "addr1"])["TransactionAmt"].agg(["mean", "std"])
            self.card1_addr1_amt_stats = {
                (k1, k2): (float(m), float(s) if not np.isnan(s) and s > 0 else 0.0)
                for (k1, k2), (m, s) in ca_grp.iterrows()
            }

        # 5. P_emaildomain Amount Statistics (mean, std)
        if "P_emaildomain" in train_df.columns and "TransactionAmt" in train_df.columns:
            valid_email = train_df.dropna(subset=["P_emaildomain"])
            email_grp = valid_email.groupby("P_emaildomain")["TransactionAmt"].agg(["mean", "std"])
            self.email_amt_stats = {
                k: (float(m), float(s) if not np.isnan(s) and s > 0 else 0.0)
                for k, (m, s) in email_grp.iterrows()
            }

        self.is_fitted = True
        logger.info("FraudFeaturePipeline fitted successfully.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations using frozen training statistics.
        Returns a new DataFrame containing original + newly engineered features.
        """
        if not self.is_fitted:
            raise RuntimeError("FraudFeaturePipeline must be fit on training data before calling transform().")

        out_df = df.copy()
        new_cols: dict[str, Any] = {}

        # -------------------------------------------------------------------
        # 1. Amount Transformations & Skewness Reduction
        # -------------------------------------------------------------------
        raw_amt = out_df["TransactionAmt"].astype(float).values
        # Clip at 0 to guard against erroneous negative amounts
        safe_amt = np.maximum(raw_amt, 0.0)
        new_cols["log_TransactionAmt"] = np.log1p(safe_amt).astype(np.float32)

        # -------------------------------------------------------------------
        # 2. Amount Z-Scores and Relative Deviations (Card1 level)
        # -------------------------------------------------------------------
        if "card1" in out_df.columns:
            card1_vals = out_df["card1"].values
            card_means = np.zeros(len(out_df), dtype=np.float32)
            card_stds = np.zeros(len(out_df), dtype=np.float32)

            for i, c1 in enumerate(card1_vals):
                if c1 in self.card1_amt_stats:
                    m, s = self.card1_amt_stats[c1]
                    card_means[i] = m
                    card_stds[i] = s
                else:
                    card_means[i] = self.global_amt_mean
                    card_stds[i] = self.global_amt_std

            # Difference from card mean
            amt_diff = raw_amt - card_means
            new_cols["amt_diff_mean_card1"] = amt_diff.astype(np.float32)

            # Ratio to card mean
            amt_ratio = raw_amt / np.maximum(card_means, 1e-3)
            new_cols["amt_ratio_mean_card1"] = amt_ratio.astype(np.float32)

            # Z-score (standard deviations from mean)
            # If std is 0 (e.g. single transaction in train), use fallback global std or 0 z-score
            std_mask = card_stds > 1e-4
            z_score = np.zeros_like(raw_amt, dtype=np.float32)
            z_score[std_mask] = (amt_diff[std_mask] / card_stds[std_mask]).astype(np.float32)
            # Clip extreme outlier z-scores to [-10, 10] for numerical stability
            new_cols["amt_zscore_card1"] = np.clip(z_score, -10.0, 10.0)

        # -------------------------------------------------------------------
        # 3. Localized Amount Z-Scores (Card1 + Addr1 interaction)
        # -------------------------------------------------------------------
        if "card1" in out_df.columns and "addr1" in out_df.columns:
            c1_vals = out_df["card1"].values
            addr_vals = out_df["addr1"].values
            ca_zscores = np.zeros(len(out_df), dtype=np.float32)

            for i in range(len(out_df)):
                c1 = c1_vals[i]
                addr = addr_vals[i]
                if pd.notna(addr):
                    pair = (c1, addr)
                    if pair in self.card1_addr1_amt_stats:
                        m, s = self.card1_addr1_amt_stats[pair]
                        if s > 1e-4:
                            ca_zscores[i] = (raw_amt[i] - m) / s
                        else:
                            ca_zscores[i] = 0.0
                    elif c1 in self.card1_amt_stats:
                        m, s = self.card1_amt_stats[c1]
                        ca_zscores[i] = (raw_amt[i] - m) / s if s > 1e-4 else 0.0

            new_cols["amt_zscore_card1_addr1"] = np.clip(ca_zscores, -10.0, 10.0)

        # -------------------------------------------------------------------
        # 4. Email Domain Amount Z-Scores
        # -------------------------------------------------------------------
        if "P_emaildomain" in out_df.columns:
            email_vals = out_df["P_emaildomain"].values
            email_zscores = np.zeros(len(out_df), dtype=np.float32)

            for i, email in enumerate(email_vals):
                if pd.notna(email) and email in self.email_amt_stats:
                    m, s = self.email_amt_stats[email]
                    if s > 1e-4:
                        email_zscores[i] = (raw_amt[i] - m) / s

            new_cols["amt_zscore_email"] = np.clip(email_zscores, -10.0, 10.0)

        # -------------------------------------------------------------------
        # 5. Cyclical Temporal Features (Diurnal 24h & Intraweek 7d)
        # -------------------------------------------------------------------
        if "TransactionDT" in out_df.columns:
            dt_vals = out_df["TransactionDT"].astype(int).values
            rel_seconds = np.maximum(dt_vals - self.dt_origin, 0)

            # Intraday hour: [0, 23]
            hour_of_day = (rel_seconds % 86400) // 3600
            new_cols["hour_of_day"] = hour_of_day.astype(np.int8)
            new_cols["hour_sin"] = np.sin(2.0 * np.pi * hour_of_day / 24.0).astype(np.float32)
            new_cols["hour_cos"] = np.cos(2.0 * np.pi * hour_of_day / 24.0).astype(np.float32)

            # Intraweek day of week: [0, 6]
            day_of_week = (rel_seconds // 86400) % 7
            new_cols["day_of_week"] = day_of_week.astype(np.int8)
            new_cols["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0).astype(np.float32)
            new_cols["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0).astype(np.float32)

        # -------------------------------------------------------------------
        # 6. Frequency / Count Encodings
        # -------------------------------------------------------------------
        for col, lookup in self.freq_lookups.items():
            if col in out_df.columns:
                series = out_df[col]
                mapped = series.map(lookup).fillna(0.0).astype(np.float32)
                new_cols[f"freq_{col}"] = mapped.values

        # -------------------------------------------------------------------
        # 7. Email Consistency & Missingness Flags
        # -------------------------------------------------------------------
        if "P_emaildomain" in out_df.columns:
            new_cols["null_P_email"] = out_df["P_emaildomain"].isna().astype(np.int8).values

        if "R_emaildomain" in out_df.columns:
            new_cols["null_R_email"] = out_df["R_emaildomain"].isna().astype(np.int8).values

        if "P_emaildomain" in out_df.columns and "R_emaildomain" in out_df.columns:
            p_email = out_df["P_emaildomain"].astype(str)
            r_email = out_df["R_emaildomain"].astype(str)
            both_present = out_df["P_emaildomain"].notna() & out_df["R_emaildomain"].notna()
            match_mask = both_present & (p_email == r_email)
            new_cols["email_match_flag"] = match_mask.astype(np.int8).values

        # -------------------------------------------------------------------
        # Attach newly engineered columns
        # -------------------------------------------------------------------
        for col_name, arr in new_cols.items():
            out_df[col_name] = arr

        self.engineered_feature_names = list(new_cols.keys())
        return out_df

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on train and transform train in a single call."""
        return self.fit(train_df).transform(train_df)

    def save(self, file_path: str | Path) -> None:
        """Serialize fitted pipeline to disk."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)
        logger.info("FraudFeaturePipeline saved to %s", file_path)

    @classmethod
    def load(cls, file_path: str | Path) -> FraudFeaturePipeline:
        """Load a serialized fitted pipeline from disk."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Pipeline artifact not found at {file_path}")
        pipeline = joblib.load(file_path)
        logger.info("FraudFeaturePipeline loaded from %s", file_path)
        return pipeline
