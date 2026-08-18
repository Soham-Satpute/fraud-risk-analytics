"""
src/explainability/shap_explainer.py
-----------------------------------
TreeSHAP Feature Attribution Engine for Champion LightGBM Model.

Supports:
  1. Interactive single-transaction inference-time explanation.
  2. Vectorized batch explanation over Pandas DataFrames.
  3. Seamless integration with ReasonCodeEngine for structured payload generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from src.explainability.reason_codes import (
    BusinessDecisionPolicy,
    ReasonCodeEngine,
    TransactionExplanationPayload,
)

logger = logging.getLogger(__name__)

_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "champion_model.joblib"
DEFAULT_PIPELINE_PATH = _PROJECT_ROOT / "models" / "feature_pipeline.joblib"

METADATA_COLUMNS: set[str] = {"TransactionID", "TransactionDT", "isFraud"}


class FraudSHAPExplainer:
    """
    High-performance TreeSHAP feature attribution engine for LightGBM fraud models.
    """

    def __init__(
        self,
        model: lgb.LGBMClassifier | None = None,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        pipeline_path: Path | str = DEFAULT_PIPELINE_PATH,
        policy: BusinessDecisionPolicy | None = None,
    ) -> None:
        """
        Initialize explainer with trained model, feature pipeline, and decision policy.
        """
        self.model_path = Path(model_path)
        self.pipeline_path = Path(pipeline_path)

        if model is not None:
            self.model = model
        elif self.model_path.exists():
            logger.info("Loading Champion model from %s", self.model_path)
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        # Feature pipeline for transforming raw input transactions
        self.pipeline: Any = None
        if self.pipeline_path.exists():
            logger.info("Loading Feature Pipeline from %s", self.pipeline_path)
            self.pipeline = joblib.load(self.pipeline_path)

        # Booster reference
        self.booster = getattr(self.model, "booster_", self.model)

        # Initialize TreeExplainer
        logger.info("Initializing TreeExplainer on Champion Booster...")
        self.explainer = shap.TreeExplainer(self.booster)

        # Base value in log-odds / margin space
        expected_val = self.explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            self.base_value = float(expected_val[1] if len(expected_val) > 1 else expected_val[0])
        elif expected_val is not None:
            self.base_value = float(expected_val)
        else:
            self.base_value = 0.0

        self.reason_engine = ReasonCodeEngine(policy=policy)
        self.feature_names = self._get_feature_names()

    def _get_feature_names(self) -> list[str]:
        """Extract training feature names from model booster."""
        if hasattr(self.model, "feature_name_"):
            return list(self.model.feature_name_)
        elif hasattr(self.booster, "feature_name"):
            return list(self.booster.feature_name())
        return []

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw input transaction records and prepare categorical encodings matching booster.
        """
        # If input has fewer columns than model features, pass through pipeline
        if self.pipeline is not None and len(df.columns) < 450:
            df_proc = self.pipeline.transform(df)
        else:
            df_proc = df.copy()

        # Build feature DataFrame with correct column ordering
        if self.feature_names:
            # Build missing column dictionary for fast non-fragmented concatenation
            missing_cols = [col for col in self.feature_names if col not in df_proc.columns]
            if missing_cols:
                missing_df = pd.DataFrame(np.nan, index=df_proc.index, columns=missing_cols)
                df_proc = pd.concat([df_proc, missing_df], axis=1)

            X = df_proc[self.feature_names].copy()
        else:
            feature_cols = [c for c in df_proc.columns if c not in METADATA_COLUMNS]
            X = df_proc[feature_cols].copy()

        # Align categorical features with booster categories
        if hasattr(self.booster, "pandas_categorical") and self.booster.pandas_categorical:
            cat_cols = [
                c for c in self.feature_names
                if c.startswith(("ProductCD", "card", "addr", "M", "id_"))
                or c in ("P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo")
            ]
            if len(cat_cols) == len(self.booster.pandas_categorical):
                for col, cats in zip(cat_cols, self.booster.pandas_categorical):
                    X[col] = pd.Categorical(X[col].astype(str), categories=cats)
        else:
            for col in X.columns:
                if X[col].dtype.name in ("category", "object") or col.startswith(("ProductCD", "card", "addr", "M", "id_")):
                    X[col] = pd.Categorical(X[col].astype(str))

        return X

    def explain_transaction(
        self,
        raw_record: dict[str, Any] | pd.Series | pd.DataFrame,
        top_k: int = 5,
    ) -> TransactionExplanationPayload:
        """
        Interactive inference-time explanation for a single transaction.

        Parameters:
            raw_record: Raw transaction attributes dictionary or Pandas series.
            top_k: Number of top risk and mitigating factors to return.

        Returns:
            TransactionExplanationPayload: Fully structured explanation payload.
        """
        if isinstance(raw_record, dict):
            df = pd.DataFrame([raw_record])
        elif isinstance(raw_record, pd.Series):
            df = pd.DataFrame([raw_record.to_dict()])
        else:
            df = raw_record.copy()

        transaction_id = int(df["TransactionID"].iloc[0]) if "TransactionID" in df.columns else None

        # Prepare feature matrix
        X = self._prepare_features(df)

        # Predict probability
        prob = float(self.model.predict_proba(X)[0, 1])

        # Compute SHAP values for single instance
        shap_vals = self.explainer.shap_values(X)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]  # Positive class SHAP
        elif len(shap_vals.shape) == 3:
            sv = shap_vals[0, :, 1]
        else:
            sv = shap_vals[0]

        feat_names = list(X.columns)
        feat_values = X.iloc[0].values

        # Context summary attributes for analyst UI
        context_keys = ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "card1", "card4", "card6", "P_emaildomain", "R_emaildomain"]
        raw_context = {k: (float(df[k].iloc[0]) if isinstance(df[k].iloc[0], (np.floating, float)) else str(df[k].iloc[0])) for k in context_keys if k in df.columns}

        return self.reason_engine.build_explanation_payload(
            transaction_id=transaction_id,
            probability=prob,
            base_value_log_odds=self.base_value,
            feature_names=feat_names,
            feature_values=feat_values,
            shap_values=sv,
            raw_context=raw_context,
            top_k=top_k,
        )

    def explain_batch(
        self,
        df: pd.DataFrame,
        top_k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray, list[TransactionExplanationPayload]]:
        """
        Vectorized batch SHAP explanation over multiple transactions.

        Parameters:
            df: Input transactions DataFrame.
            top_k: Number of top factors per transaction.

        Returns:
            tuple: (probabilities_array, shap_values_matrix, list_of_payloads)
        """
        logger.info("Preparing feature matrix for batch of %d transactions...", len(df))
        X = self._prepare_features(df)

        logger.info("Computing batch model predictions...")
        probabilities = self.model.predict_proba(X)[:, 1]

        logger.info("Computing TreeSHAP values for %d instances...", len(X))
        raw_shap = self.explainer.shap_values(X)
        if isinstance(raw_shap, list):
            shap_matrix = raw_shap[1]
        elif len(raw_shap.shape) == 3:
            shap_matrix = raw_shap[:, :, 1]
        else:
            shap_matrix = raw_shap

        feat_names = list(X.columns)
        payloads: list[TransactionExplanationPayload] = []

        context_keys = ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "card1", "card4", "card6", "P_emaildomain", "R_emaildomain"]

        logger.info("Extracting structured reason codes for %d transactions...", len(df))
        for i in range(len(df)):
            tx_id = int(df["TransactionID"].iloc[i]) if "TransactionID" in df.columns else None
            prob_i = float(probabilities[i])
            sv_i = shap_matrix[i]
            val_i = X.iloc[i].values

            raw_context = {
                k: (float(df[k].iloc[i]) if isinstance(df[k].iloc[i], (np.floating, float)) else str(df[k].iloc[i]))
                for k in context_keys
                if k in df.columns
            }

            payload = self.reason_engine.build_explanation_payload(
                transaction_id=tx_id,
                probability=prob_i,
                base_value_log_odds=self.base_value,
                feature_names=feat_names,
                feature_values=val_i,
                shap_values=sv_i,
                raw_context=raw_context,
                top_k=top_k,
            )
            payloads.append(payload)

        return probabilities, shap_matrix, payloads

    def compute_global_feature_importance(
        self,
        df: pd.DataFrame,
        max_samples: int = 5000,
    ) -> pd.DataFrame:
        """
        Compute global mean absolute SHAP feature importances.
        """
        sample_df = df.sample(n=min(len(df), max_samples), random_state=42)
        X = self._prepare_features(sample_df)

        shap_vals = self.explainer.shap_values(X)
        if isinstance(shap_vals, list):
            sv = shap_vals[1]
        elif len(shap_vals.shape) == 3:
            sv = shap_vals[:, :, 1]
        else:
            sv = shap_vals

        mean_abs_shap = np.abs(sv).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": X.columns,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        return importance_df
