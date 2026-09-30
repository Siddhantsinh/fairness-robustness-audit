"""Group-disaggregated fairness metrics (Phase 6). Built on Fairlearn's
MetricFrame so per-group values and cross-group disparities come from one
place.
"""
from __future__ import annotations

import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def group_metric_frame(y_true, y_pred, sensitive_features) -> MetricFrame:
    metrics = {
        "count": lambda yt, yp: len(yt),
        "selection_rate": selection_rate,
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }
    return MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )


def disparity_summary(y_true, y_pred, sensitive_features) -> dict:
    return {
        "demographic_parity_difference": demographic_parity_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        ),
        "equalized_odds_difference": equalized_odds_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        ),
    }


def group_table(y_true, y_pred, sensitive_features) -> pd.DataFrame:
    """Per-group metric table, ready to drop into the fairness report."""
    frame = group_metric_frame(y_true, y_pred, sensitive_features)
    return frame.by_group
