import numpy as np
import pandas as pd
import pytest

from audit_project.evaluate import apply_threshold, compute_overall_metrics
from audit_project.fairness import disparity_summary, group_table
from audit_project.features import build_preprocessor, infer_column_types
from audit_project.robustness import add_gaussian_noise, mask_missing
from audit_project.train import build_model_pipeline, fit, make_split


@pytest.fixture
def sample_dataset():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, size=n),
        "education_num": np.random.randint(1, 16, size=n),
        "hours_per_week": np.random.choice([20, 40, 50, 60], size=n),
        "workclass": np.random.choice(["Private", "Self-emp", "Gov"], size=n),
        "sex": np.random.choice(["Male", "Female"], size=n),
        "race": np.random.choice(["White", "Black", "Asian"], size=n),
        "income": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
    })
    return df


def test_infer_column_types(sample_dataset):
    num_cols, cat_cols = infer_column_types(sample_dataset, exclude=["income"])
    assert "age" in num_cols
    assert "workclass" in cat_cols
    assert "income" not in num_cols and "income" not in cat_cols


def test_preprocessor_and_model_fitting(sample_dataset):
    num_cols, cat_cols = infer_column_types(sample_dataset, exclude=["income"])
    preprocessor = build_preprocessor(num_cols, cat_cols)
    pipeline = build_model_pipeline(preprocessor, model_name="logistic_regression", seed=42)

    X_train, X_test, y_train, y_test = make_split(sample_dataset, target_col="income", test_size=0.25, seed=42)
    fitted_pipe = fit(pipeline, X_train, y_train)

    y_proba = fitted_pipe.predict_proba(X_test)[:, 1]
    assert len(y_proba) == len(X_test)
    assert (y_proba >= 0.0).all() and (y_proba <= 1.0).all()


def test_fairness_disparity_computation():
    y_true = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    sensitive = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])

    disparities = disparity_summary(y_true, y_pred, sensitive_features=sensitive)
    assert "demographic_parity_difference" in disparities
    assert "equalized_odds_difference" in disparities
    assert 0.0 <= disparities["demographic_parity_difference"] <= 1.0


def test_robustness_perturbations(sample_dataset):
    num_cols = ["age", "education_num"]
    noisy_df = add_gaussian_noise(sample_dataset, num_cols, noise_level=0.1, seed=42)
    assert noisy_df.shape == sample_dataset.shape
    assert not np.allclose(noisy_df["age"], sample_dataset["age"])

    masked_df = mask_missing(sample_dataset, num_cols, missing_rate=0.2, seed=42)
    assert masked_df["age"].isna().sum() > 0
