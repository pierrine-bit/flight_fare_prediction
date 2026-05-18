"""
Flight Fare Prediction — src package
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Exposes the high-level public API for each pipeline stage so callers
(main.py, retrain.py, DAG tasks, notebooks) can import directly from `src`
without knowing the internal module layout.

Typical usage
─────────────
    from src import load_data, clean_pipeline, engineer_features
    from src import train_all_models, run_evaluation
    from src.logger import setup_logging

Pipeline stages
───────────────
    Stage 1  load_data, inspect_data                         ← data_loader.py
    Stage 2  clean_pipeline, audit_missing_values            ← data_cleaner.py
    Stage 3  engineer_features, build_preprocessor,
             split_and_transform                             ← feature_engineering.py
    Stage 4  run_eda                                         ← eda.py
    Stage 5  train_all_models, build_cv_report               ← models.py
    Stage 6  run_evaluation                                  ← evaluation.py

EDA dashboards (eda.py)
───────────────────────
    plot_fare_overview_dashboard  → eda_fare_overview.png
    plot_airline_dashboard        → eda_airline_analysis.png
    plot_route_dashboard          → eda_route_analysis.png
    plot_correlation_heatmap      → eda_correlations.png

Evaluation plots (evaluation.py)
─────────────────────────────────
    plot_cv_comparison            → eval_cv_comparison.png
    plot_best_model_dashboard     → eval_best_model_diagnostics.png
    plot_learning_curve           → eval_learning_curve.png
    plot_ridge_vs_lasso           → eval_ridge_vs_lasso.png
    plot_predicted_vs_actual      → predicted_vs_actual_<model>.png
    plot_residuals                → residuals_<model>.png
"""

from .data_loader import load_data, inspect_data
from .data_cleaner import clean_pipeline, audit_missing_values, validate_schema
from .feature_engineering import engineer_features, build_preprocessor, split_and_transform, SplitData
from .eda import (
    run_eda,
    plot_fare_overview_dashboard,
    plot_airline_dashboard,
    plot_route_dashboard,
    plot_correlation_heatmap,
)
from .models import train_all_models, build_cv_report
from .evaluation import (
    run_evaluation,
    build_comparison_table,
    plot_cv_comparison,
    plot_cv_fold_scores,        # alias kept for notebook compatibility
    plot_best_model_dashboard,
    plot_predicted_vs_actual,
    plot_residuals,
    plot_learning_curve,
    plot_ridge_vs_lasso,
)

__all__ = [
    # Stage 1 — Loading
    "load_data",
    "inspect_data",
    # Stage 2 — Cleaning
    "clean_pipeline",
    "validate_schema",
    "audit_missing_values",
    # Stage 3 — Feature engineering
    "engineer_features",
    "build_preprocessor",
    "split_and_transform",
    "SplitData",
    # Stage 4 — EDA dashboards
    "run_eda",
    "plot_fare_overview_dashboard",
    "plot_airline_dashboard",
    "plot_route_dashboard",
    "plot_correlation_heatmap",
    # Stage 5 — Training
    "train_all_models",
    "build_cv_report",
    # Stage 6 — Evaluation
    "run_evaluation",
    "build_comparison_table",
    "plot_cv_comparison",
    "plot_cv_fold_scores",
    "plot_best_model_dashboard",
    "plot_predicted_vs_actual",
    "plot_residuals",
    "plot_learning_curve",
    "plot_ridge_vs_lasso",
]
