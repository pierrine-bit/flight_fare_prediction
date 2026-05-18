
import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import OUTPUTS_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_STATE, CV_FOLDS, GRID_CV_FOLDS, N_ITER
from src.logger import setup_logging
from src.data_loader import load_data, inspect_data
from src.data_cleaner import clean_pipeline, audit_missing_values
from src.feature_engineering import engineer_features, build_preprocessor, split_and_transform
from src.eda import run_eda
from src.models import train_all_models, build_cv_report
from src.evaluation import run_evaluation

logger = logging.getLogger(__name__)


def run_pipeline(data_path: Path, skip_eda: bool = False) -> dict:
    """
    Execute every pipeline stage in order and return the evaluation summary.
    """
    for d in [OUTPUTS_DIR, MODELS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Load ─────────────────────────────────────────────────────────
    logger.info("\n%s\nSTAGE 1 — DATA LOADING\n%s", "=" * 65, "=" * 65)
    df_raw = load_data(data_path)
    inspect_data(df_raw)

    # ── Stage 2: Clean ────────────────────────────────────────────────────────
    df_clean = clean_pipeline(df_raw)
    audit_missing_values(df_clean)   # post-clean zero-null confirmation

    # ── Stage 3: Feature engineering ─────────────────────────────────────────
    df_feat = engineer_features(df_clean)
    split   = split_and_transform(df_feat, build_preprocessor())

    # ── Stage 4: EDA ─────────────────────────────────────────────────────────
    if not skip_eda:
        run_eda(df_feat, OUTPUTS_DIR)
    else:
        logger.info("EDA skipped (--skip-eda flag set)")

    # ── Stage 5: Model training ───────────────────────────────────────────────
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    logger.info("CV config: %d-fold shared | GridSearchCV: %d-fold | RandomizedSearchCV: n_iter=%d",
                CV_FOLDS, GRID_CV_FOLDS, N_ITER)

    best_models, results = train_all_models(
        split.X_train_proc, split.y_train,
        split.X_test_proc,  split.y_test,
        kf, log_transformed=split.log_transformed,
    )

    # ── Stage 5b: Full CV report ──────────────────────────────────────────────
    cv_report = build_cv_report(best_models, split.X_train_proc, split.y_train, kf)
    cv_report.to_csv(REPORTS_DIR / "cv_report_full.csv")
    logger.info("Full CV report saved → %s", REPORTS_DIR / "cv_report_full.csv")

    # ── Stage 6: Evaluation ───────────────────────────────────────────────────
    summary = run_evaluation(
        best_models=best_models,
        results=results,
        X_test_proc=split.X_test_proc,
        y_test=split.y_test,
        X_train_proc=split.X_train_proc,
        y_train=split.y_train,
        feature_names=split.feature_names,
        kf=kf,
        save_dir=OUTPUTS_DIR,
        log_transformed=split.log_transformed,
    )

    # ── Stage 7: Retrain best model on full dataset ───────────────────────────
    logger.info("\n%s\nSTAGE 7 — RETRAINING BEST MODEL ON FULL DATA\n%s", "=" * 60, "=" * 60)
    best_name  = summary["best_model_name"]
    best_model = summary["best_model"]

    X_full = np.vstack([split.X_train_proc, split.X_test_proc])
    y_full = np.concatenate([split.y_train,  split.y_test])
    logger.info("Retraining %s on full dataset (%s rows)...", best_name, f"{len(y_full):,}")
    best_model.fit(X_full, y_full)          # fit() directly — clone() breaks XGBoost pickling
    logger.info("Retraining complete.")

    # ── Stage 8: Persist artefacts ────────────────────────────────────────────
    logger.info("\n%s\nSTAGE 8 — SAVING ARTEFACTS\n%s", "=" * 60, "=" * 60)

    joblib.dump(best_model,         MODELS_DIR / "best_model.pkl")
    joblib.dump(split.preprocessor, MODELS_DIR / "preprocessor.pkl")
    logger.info("Saved: models/best_model.pkl (%s)", best_name)
    logger.info("Saved: models/preprocessor.pkl")

    # ── Final summary ─────────────────────────────────────────────────────────
    row    = summary["comparison_table"].loc[best_name]
    r2_col = "R² (BDT)" if "R² (BDT)" in row.index else "R² (log)"
    logger.info("\n%s\nPIPELINE COMPLETE\n%s\n"
                "  Best model : %s\n"
                "  Test R²    : %s\n"
                "  MAE        : BDT %s\n"
                "  RMSE       : BDT %s\n"
                "  CV R² Mean : %s  (±%s)\n"
                "  Outputs    : %s",
                "=" * 60, "=" * 60,
                best_name,
                row[r2_col],
                f"{row['MAE (BDT)']:,.0f}",
                f"{row['RMSE (BDT)']:,.0f}",
                row["CV R² Mean"], row["CV R² Std"],
                OUTPUTS_DIR.resolve())

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Flight Fare Prediction — end-to-end ML pipeline"
    )
    parser.add_argument("--data",     type=Path, default=None,
                        help="Path to the CSV dataset (default: config.DATA_PATH)")
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip EDA plot generation")
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=OUTPUTS_DIR)

    from src.config import DATA_PATH
    data_path = args.data if args.data else DATA_PATH

    run_pipeline(data_path=data_path, skip_eda=args.skip_eda)


if __name__ == "__main__":
    main()
