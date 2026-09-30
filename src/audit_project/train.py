"""Baseline model training (Phase 4). Two required baselines: logistic
regression and a tree-based model."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def make_split(df: pd.DataFrame, target_col: str, test_size: float, seed: int):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def build_model_pipeline(preprocessor: ColumnTransformer, model_name: str, seed: int) -> Pipeline:
    if model_name == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=seed)
    elif model_name == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=seed, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])


def fit(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    pipeline.fit(X_train, y_train)
    return pipeline
