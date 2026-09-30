"""Robustness experiments (Phase 7): threshold sensitivity, feature noise,
missingness, and distribution shift. Each function returns a metrics dict
so results can be assembled into the robustness table.
"""
from __future__ import annotations

from typing import Any, Callable
import numpy as np
import pandas as pd

from audit_project.evaluate import compute_overall_metrics
from audit_project.fairness import disparity_summary


def add_gaussian_noise(X: pd.DataFrame, columns: list[str], noise_level: float, seed: int) -> pd.DataFrame:
    """Add noise scaled to each column's std. noise_level=0 returns X unchanged."""
    if noise_level == 0.0:
        return X.copy()
    rng = np.random.default_rng(seed)
    X_noisy = X.copy()
    for col in columns:
        if col in X_noisy.columns and pd.api.types.is_numeric_dtype(X_noisy[col]):
            std = X_noisy[col].std()
            if std > 0:
                X_noisy[col] = X_noisy[col] + rng.normal(0, noise_level * std, size=len(X))
    return X_noisy


def mask_missing(X: pd.DataFrame, columns: list[str], missing_rate: float, seed: int) -> pd.DataFrame:
    """Randomly set a fraction of values in `columns` to NaN."""
    if missing_rate == 0.0:
        return X.copy()
    rng = np.random.default_rng(seed)
    X_masked = X.copy()
    for col in columns:
        if col in X_masked.columns:
            mask = rng.random(len(X_masked)) < missing_rate
            X_masked.loc[mask, col] = np.nan
    return X_masked


def evaluate_at_thresholds(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    sensitive_features: pd.DataFrame | pd.Series,
    thresholds: list[float],
) -> dict[float, dict[str, Any]]:
    """Run evaluate/fairness metrics at each decision threshold."""
    y_proba = model.predict_proba(X_test)[:, 1]
    results = {}
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        overall = compute_overall_metrics(y_test, y_pred, y_proba)
        disparities = disparity_summary(y_test, y_pred, sensitive_features)
        results[t] = {
            **overall,
            **disparities,
            "threshold": t,
        }
    return results


def evaluate_noise_robustness(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    sensitive_features: pd.DataFrame | pd.Series,
    numeric_cols: list[str],
    noise_levels: list[float],
    seed: int,
) -> dict[float, dict[str, Any]]:
    """Evaluate performance degradation under Gaussian feature noise."""
    results = {}
    for nl in noise_levels:
        X_noisy = add_gaussian_noise(X_test, numeric_cols, nl, seed=seed)
        y_proba = model.predict_proba(X_noisy)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        overall = compute_overall_metrics(y_test, y_pred, y_proba)
        disparities = disparity_summary(y_test, y_pred, sensitive_features)
        results[nl] = {
            **overall,
            **disparities,
            "noise_level": nl,
        }
    return results


def evaluate_missingness_robustness(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    sensitive_features: pd.DataFrame | pd.Series,
    columns: list[str],
    missingness_rates: list[float],
    seed: int,
) -> dict[float, dict[str, Any]]:
    """Evaluate performance degradation under simulated feature missingness."""
    results = {}
    for mr in missingness_rates:
        X_masked = mask_missing(X_test, columns, mr, seed=seed)
        y_proba = model.predict_proba(X_masked)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        overall = compute_overall_metrics(y_test, y_pred, y_proba)
        disparities = disparity_summary(y_test, y_pred, sensitive_features)
        results[mr] = {
            **overall,
            **disparities,
            "missingness_rate": mr,
        }
    return results


def evaluate_distribution_shift(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    sensitive_features: pd.DataFrame | pd.Series,
    shift_condition: Callable[[pd.DataFrame], pd.Series],
    cohort_labels: tuple[str, str] = ("Cohort A", "Cohort B"),
) -> dict[str, dict[str, Any]]:
    """Evaluate model performance on two subpopulations defined by shift_condition."""
    mask = shift_condition(X_test)
    results = {}
    for label, sub_mask in [(cohort_labels[0], mask), (cohort_labels[1], ~mask)]:
        if sub_mask.sum() == 0:
            continue
        X_sub = X_test.loc[sub_mask]
        y_sub = y_test.iloc[sub_mask.values] if hasattr(y_test, "iloc") else y_test[sub_mask]
        sens_sub = sensitive_features.iloc[sub_mask.values] if hasattr(sensitive_features, "iloc") else sensitive_features[sub_mask]
        
        y_proba = model.predict_proba(X_sub)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        overall = compute_overall_metrics(y_sub, y_pred, y_proba)
        disparities = disparity_summary(y_sub, y_pred, sens_sub)
        results[label] = {
            **overall,
            **disparities,
            "count": int(sub_mask.sum()),
        }
    return results
