"""
src/models/evaluation.py
------------------------
Evaluation metrics and statistical confidence interval engine for fraud models.

Core Capabilities:
  - PR-AUC (Precision-Recall Area Under Curve — primary metric under heavy class imbalance)
  - ROC-AUC
  - Recall at Fixed False Positive Rates (FPR = 1%, FPR = 5%)
  - 1,000-sample Bootstrapped 95% Confidence Intervals on held-out test data
  - Operating threshold sweep for cost curves and review volume estimation
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
) -> dict[str, float]:
    """
    Compute standard rank and probability metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    # Avoid crash if only 1 class in sample
    if len(np.unique(y_true)) < 2:
        return {
            "pr_auc": 0.0,
            "roc_auc": 0.5,
            "log_loss": float("nan"),
            "brier_score": float("nan"),
        }

    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    ll = float(log_loss(y_true, np.clip(y_prob, 1e-7, 1.0 - 1e-7)))
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "pr_auc": round(pr_auc, 5),
        "roc_auc": round(roc_auc, 5),
        "log_loss": round(ll, 5),
        "brier_score": round(brier, 5),
    }


def calculate_recall_at_fixed_fpr(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    target_fpr: float = 0.01,
) -> dict[str, float]:
    """
    Determine the operating threshold where False Positive Rate <= target_fpr,
    and return the achieved recall (fraud capture rate) and precision at that threshold.

    Parameters:
      y_true: Ground truth binary labels {0, 1}
      y_prob: Predicted risk probabilities [0.0, 1.0]
      target_fpr: Desired FPR limit (e.g. 0.01 for 1% FPR, 0.05 for 5% FPR)

    Returns:
      dict with operating_threshold, achieved_fpr, recall, precision, review_rate
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    fprs, tprs, thresholds = roc_curve(y_true, y_prob)

    # Find the index of the highest threshold where FPR <= target_fpr
    valid_indices = np.where(fprs <= target_fpr)[0]
    if len(valid_indices) == 0:
        idx = 0
    else:
        idx = valid_indices[-1]  # Highest FPR that is still <= target_fpr

    achieved_fpr = float(fprs[idx])
    recall = float(tprs[idx])
    threshold = float(thresholds[idx]) if idx < len(thresholds) else 1.0

    # Calculate precision at this operating threshold
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    review_rate = float((tp + fp) / len(y_true))

    return {
        "target_fpr": target_fpr,
        "operating_threshold": round(threshold, 5),
        "achieved_fpr": round(achieved_fpr, 5),
        "recall": round(recall, 5),
        "precision": round(precision, 5),
        "review_rate": round(review_rate, 5),
    }


def bootstrap_metric_confidence_intervals(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Generate non-parametric bootstrap confidence intervals on held-out test predictions
    across 1,000 resamples to provide empirical estimates of uncertainty.

    Evaluates:
      - PR-AUC
      - ROC-AUC
      - Recall @ 1% FPR
      - Recall @ 5% FPR
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n_samples = len(y_true)

    # Point estimates on full dataset
    base_metrics = compute_classification_metrics(y_true, y_prob)
    base_rec1 = calculate_recall_at_fixed_fpr(y_true, y_prob, target_fpr=0.01)["recall"]
    base_rec5 = calculate_recall_at_fixed_fpr(y_true, y_prob, target_fpr=0.05)["recall"]

    rng = np.random.default_rng(random_state)
    boot_pr_auc = np.zeros(n_bootstraps, dtype=np.float32)
    boot_roc_auc = np.zeros(n_bootstraps, dtype=np.float32)
    boot_rec1 = np.zeros(n_bootstraps, dtype=np.float32)
    boot_rec5 = np.zeros(n_bootstraps, dtype=np.float32)

    logger.info("Running %d bootstrap resamples for 95%% confidence intervals...", n_bootstraps)

    for i in range(n_bootstraps):
        indices = rng.integers(0, n_samples, size=n_samples)
        b_true = y_true[indices]
        b_prob = y_prob[indices]

        # Guard against zero positive or zero negative samples in resample
        if len(np.unique(b_true)) < 2:
            boot_pr_auc[i] = base_metrics["pr_auc"]
            boot_roc_auc[i] = base_metrics["roc_auc"]
            boot_rec1[i] = base_rec1
            boot_rec5[i] = base_rec5
            continue

        boot_pr_auc[i] = average_precision_score(b_true, b_prob)
        boot_roc_auc[i] = roc_auc_score(b_true, b_prob)
        boot_rec1[i] = calculate_recall_at_fixed_fpr(b_true, b_prob, target_fpr=0.01)["recall"]
        boot_rec5[i] = calculate_recall_at_fixed_fpr(b_true, b_prob, target_fpr=0.05)["recall"]

    alpha_low = ((1.0 - ci) / 2.0) * 100.0
    alpha_high = (1.0 - (1.0 - ci) / 2.0) * 100.0

    def summarize_metric(point: float, boot_arr: np.ndarray) -> dict[str, float]:
        low = float(np.percentile(boot_arr, alpha_low))
        high = float(np.percentile(boot_arr, alpha_high))
        std_err = float(np.std(boot_arr))
        return {
            "point_estimate": round(point, 5),
            "ci_95_low": round(low, 5),
            "ci_95_high": round(high, 5),
            "std_error": round(std_err, 5),
        }

    return {
        "pr_auc": summarize_metric(base_metrics["pr_auc"], boot_pr_auc),
        "roc_auc": summarize_metric(base_metrics["roc_auc"], boot_roc_auc),
        "recall_at_1pct_fpr": summarize_metric(base_rec1, boot_rec1),
        "recall_at_5pct_fpr": summarize_metric(base_rec5, boot_rec5),
    }


def generate_threshold_sweep(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_thresholds: int = 50,
) -> list[dict[str, float]]:
    """
    Generate fine-grained operating curve data across thresholds from 0.01 to 0.99.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    total_tx = len(y_true)
    total_fraud = int((y_true == 1).sum())
    total_legit = int((y_true == 0).sum())

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    sweep_data = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        recall = tp / total_fraud if total_fraud > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        fpr = fp / total_legit if total_legit > 0 else 0.0
        review_rate = (tp + fp) / total_tx

        sweep_data.append({
            "threshold": round(float(t), 4),
            "recall": round(float(recall), 4),
            "precision": round(float(precision), 4),
            "fpr": round(float(fpr), 4),
            "review_rate": round(float(review_rate), 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
        })

    return sweep_data
