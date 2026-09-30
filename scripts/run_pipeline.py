#!/usr/bin/env python
"""End-to-end pipeline entry point.

Usage:
    python scripts/run_pipeline.py --config configs/config.yaml --data data/raw.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Ensure 'src' is in pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audit_project.data import load_and_validate
from audit_project.evaluate import compute_overall_metrics
from audit_project.fairness import disparity_summary, group_table
from audit_project.features import build_preprocessor, infer_column_types
from audit_project.robustness import (
    evaluate_at_thresholds,
    evaluate_distribution_shift,
    evaluate_missingness_robustness,
    evaluate_noise_robustness,
)
from audit_project.train import build_model_pipeline, fit, make_split


def run_audit(config_path: str, data_path: str) -> None:
    print(f"\n=======================================================")
    print(f"Fairness and Robustness Audit Pipeline")
    print(f"Config: {config_path} | Data: {data_path}")
    print(f"=======================================================\n")

    # 1. Load and Validate
    df, config = load_and_validate(data_path, config_path)
    target_col = config["target"]["column"]
    pos_label = config["target"]["positive_label"]
    group_attrs = config["groups"]["attributes"]
    seed = config["split"]["random_seed"]
    test_size = config["split"]["test_size"]
    
    print(f"[Phase 3] Data loaded: {len(df):,} rows, {len(df.columns)} columns.")

    # 2. Encode target to binary 0/1
    df = df.copy()
    df[target_col] = (df[target_col] == pos_label).astype(int)
    print(f"Target '{target_col}' encoded: positive label '{pos_label}' -> 1 (base rate: {df[target_col].mean():.2%})")

    # 3. Features & Train/Test Split
    exclude = [target_col]
    numeric_cols, categorical_cols = infer_column_types(df, exclude=exclude)
    print(f"Inferred {len(numeric_cols)} numeric and {len(categorical_cols)} categorical predictors.")

    X_train, X_test, y_train, y_test = make_split(df, target_col=target_col, test_size=test_size, seed=seed)
    print(f"Split data into Train ({len(X_train):,}) and Test ({len(X_test):,}) with seed={seed}.\n")

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    models_to_test = ["logistic_regression", "random_forest"]
    overall_results = []
    fairness_results = []
    robustness_records = []

    for model_name in models_to_test:
        print(f"-------------------------------------------------------")
        print(f"Training Model: {model_name.upper()}")
        print(f"-------------------------------------------------------")
        
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        pipeline = build_model_pipeline(preprocessor, model_name=model_name, seed=seed)
        pipeline = fit(pipeline, X_train, y_train)

        # Predictions on Test set
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= config["decision_threshold"]).astype(int)

        # Phase 4: Overall Evaluation
        metrics = compute_overall_metrics(y_test, y_pred, y_proba)
        metrics["model"] = model_name
        overall_results.append(metrics)

        print("\n--- Overall Performance Metrics ---")
        for k in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]:
            print(f"  {k:18s}: {metrics.get(k, 0.0):.4f}")

        # Phase 6: Fairness Audit per group attribute
        print("\n--- Fairness Disparity Audit ---")
        for attr in group_attrs:
            sens_train = X_train[attr]
            sens_test = X_test[attr]
            
            disparities = disparity_summary(y_test, y_pred, sensitive_features=sens_test)
            disparities["model"] = model_name
            disparities["attribute"] = attr
            fairness_results.append(disparities)

            print(f"  Attribute: [{attr}]")
            print(f"    Demographic Parity Diff : {disparities['demographic_parity_difference']:.4f}")
            print(f"    Equalized Odds Diff     : {disparities['equalized_odds_difference']:.4f}")

            # Display per-group detail table
            grp_tbl = group_table(y_test, y_pred, sensitive_features=sens_test)
            formatted_tbl = grp_tbl[["count", "selection_rate", "accuracy", "recall", "false_positive_rate", "false_negative_rate"]]
            print(f"\n{formatted_tbl.to_string()}\n")

        # Phase 7: Robustness Audit
        print("--- Robustness Experiments ---")
        # Experiment A: Thresholds
        thresholds = config["robustness"]["thresholds_to_test"]
        sens_primary = X_test[group_attrs[0]]
        thresh_res = evaluate_at_thresholds(pipeline, X_test, y_test, sensitive_features=sens_primary, thresholds=thresholds)
        for t, res in thresh_res.items():
            robustness_records.append({
                "model": model_name,
                "experiment": "threshold",
                "condition": f"threshold={t:.2f}",
                "f1": res["f1"],
                "accuracy": res["accuracy"],
                "dp_diff": res["demographic_parity_difference"],
                "eo_diff": res["equalized_odds_difference"],
            })

        # Experiment B: Noise
        noise_levels = config["robustness"]["noise_levels"]
        noise_res = evaluate_noise_robustness(
            pipeline, X_test, y_test, sensitive_features=sens_primary,
            numeric_cols=numeric_cols, noise_levels=noise_levels, seed=seed
        )
        for nl, res in noise_res.items():
            robustness_records.append({
                "model": model_name,
                "experiment": "noise",
                "condition": f"noise_level={nl:.2f}",
                "f1": res["f1"],
                "accuracy": res["accuracy"],
                "dp_diff": res["demographic_parity_difference"],
                "eo_diff": res["equalized_odds_difference"],
            })

        # Experiment C: Missingness
        miss_rates = config["robustness"]["missingness_rates"]
        miss_res = evaluate_missingness_robustness(
            pipeline, X_test, y_test, sensitive_features=sens_primary,
            columns=numeric_cols[:3], missingness_rates=miss_rates, seed=seed
        )
        for mr, res in miss_res.items():
            robustness_records.append({
                "model": model_name,
                "experiment": "missingness",
                "condition": f"missingness={mr:.2f}",
                "f1": res["f1"],
                "accuracy": res["accuracy"],
                "dp_diff": res["demographic_parity_difference"],
                "eo_diff": res["equalized_odds_difference"],
            })

        # Experiment D: Distribution shift (younger <= 35 vs older > 35)
        shift_res = evaluate_distribution_shift(
            pipeline, X_test, y_test, sensitive_features=sens_primary,
            shift_condition=lambda df_: df_["age"] <= 35,
            cohort_labels=("age <= 35", "age > 35"),
        )
        for cohort, res in shift_res.items():
            robustness_records.append({
                "model": model_name,
                "experiment": "distribution_shift",
                "condition": f"cohort: {cohort}",
                "f1": res["f1"],
                "accuracy": res["accuracy"],
                "dp_diff": res["demographic_parity_difference"],
                "eo_diff": res["equalized_odds_difference"],
            })
        print("  Threshold, Noise, Missingness, and Subpopulation Shift audits completed.\n")

    # Save artifacts to reports/
    df_overall = pd.DataFrame(overall_results).set_index("model")
    df_fairness = pd.DataFrame(fairness_results)
    df_robustness = pd.DataFrame(robustness_records)

    df_overall.to_csv(reports_dir / "baseline_performance.csv")
    df_fairness.to_csv(reports_dir / "fairness_disparities.csv", index=False)
    df_robustness.to_csv(reports_dir / "robustness_summary.csv", index=False)

    print("=======================================================")
    print("AUDIT SUMMARY TABLES")
    print("=======================================================")
    print("\n--- Model Baseline Performance ---")
    print(df_overall[["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]].to_string())

    print("\n--- Fairness Disparities (Test Set) ---")
    print(df_fairness[["model", "attribute", "demographic_parity_difference", "equalized_odds_difference"]].to_string(index=False))

    print(f"\nArtifacts saved to {reports_dir.resolve()}/")
    print("  - baseline_performance.csv")
    print("  - fairness_disparities.csv")
    print("  - robustness_summary.csv")
    print("Pipeline execution complete!\n")


def main():
    parser = argparse.ArgumentParser(description="Run Fairness & Robustness Audit Pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--data", default="data/raw.csv", help="Path to data/raw.csv")
    args = parser.parse_args()

    run_audit(config_path=args.config, data_path=args.data)


if __name__ == "__main__":
    main()
