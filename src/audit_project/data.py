"""Loading and validation for the audit dataset.

Rules this module should enforce (Phase 3):
- Fail loudly and clearly when required columns are missing or invalid.
- Never use the test set to make preprocessing decisions.
- Never silently drop a group.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


class DataValidationError(ValueError):
    """Raised when the loaded dataset does not match the expected schema."""


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw dataset from disk.

    TODO: point this at your chosen dataset (local file or a fetch
    helper, e.g. fairlearn.datasets or an OpenML loader).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for "
            "download instructions."
        )
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame, config: dict) -> None:
    """Raise DataValidationError if required columns are missing/invalid."""
    required = [config["target"]["column"], *config["groups"]["attributes"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    target_col = config["target"]["column"]
    if df[target_col].isna().any():
        raise DataValidationError(f"Target column '{target_col}' has missing values.")

    for group_col in config["groups"]["attributes"]:
        if df[group_col].nunique(dropna=False) < 2:
            raise DataValidationError(
                f"Group column '{group_col}' has fewer than 2 distinct values."
            )


def load_and_validate(csv_path: str | Path, config_path: str | Path) -> tuple[pd.DataFrame, dict]:
    config = load_config(config_path)
    df = load_raw_data(csv_path)
    validate_schema(df, config)
    return df, config
