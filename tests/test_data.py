import pandas as pd
import pytest

from audit_project.data import DataValidationError, validate_schema

MINIMAL_CONFIG = {
    "target": {"column": "target"},
    "groups": {"attributes": ["group"]},
}


def test_validate_schema_passes_on_valid_data():
    df = pd.DataFrame({
        "target": [0, 1, 0, 1],
        "group": ["A", "B", "A", "B"],
        "feature1": [1.0, 2.0, 3.0, 4.0],
    })
    validate_schema(df, MINIMAL_CONFIG)  # should not raise


def test_validate_schema_fails_on_missing_column():
    df = pd.DataFrame({"target": [0, 1]})
    with pytest.raises(DataValidationError):
        validate_schema(df, MINIMAL_CONFIG)


def test_validate_schema_fails_on_missing_target_values():
    df = pd.DataFrame({
        "target": [0, None],
        "group": ["A", "B"],
    })
    with pytest.raises(DataValidationError):
        validate_schema(df, MINIMAL_CONFIG)


def test_validate_schema_fails_on_single_valued_group():
    df = pd.DataFrame({
        "target": [0, 1],
        "group": ["A", "A"],
    })
    with pytest.raises(DataValidationError):
        validate_schema(df, MINIMAL_CONFIG)
