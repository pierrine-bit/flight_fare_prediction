
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DATA_PATH, OUTPUTS_DIR, MODELS_DIR
from src.logger import setup_logging
from src.data_loader import load_data, inspect_data
from src.data_cleaner import clean_pipeline, audit_missing_values
from src.feature_engineering import engineer_features, build_preprocessor, split_and_transform

logger = logging.getLogger(__name__)


def retrain(data_path: Path) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=OUTPUTS_DIR)

    # ── Check best_model.pkl exists ───────────────────────────────────────────
    model_path = MODELS_DIR / "best_model.pkl"
    if not model_path.exists():
        logger.error(
            "best_model.pkl not found in %s. "
            "Run `python main.py` first to train and save the best model.",
            OUTPUTS_DIR,
        )
        sys.exit(1)

    # ── Stage 1: Load ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("RETRAIN — Loading data from %s", data_path)
    logger.info("=" * 60)
    df_raw = load_data(data_path)
    inspect_data(df_raw)

    # ── Stage 2: Clean ────────────────────────────────────────────────────────
    df_clean = clean_pipeline(df_raw)
    audit_missing_values(df_clean)

    # ── Stage 3: Feature engineering ─────────────────────────────────────────
    df_feat = engineer_features(df_clean)
    split   = split_and_transform(df_feat, build_preprocessor())

    logger.info(
        "Data ready — %d train rows, %d test rows, %d features",
        split.X_train_proc.shape[0], split.X_test_proc.shape[0], split.X_train_proc.shape[1],
    )

    # ── Stage 4: Load saved best model ────────────────────────────────────────
    best_model = joblib.load(model_path)
    best_name  = type(best_model).__name__
    logger.info("Loaded best model: %s from %s", best_name, model_path)


    # ── Stage 5: Retrain on full dataset ─────────────────────────────────────
    X_full = np.vstack([split.X_train_proc, split.X_test_proc])
    y_full = np.concatenate([split.y_train,  split.y_test])

    logger.info(
        "Retraining %s on full dataset (%s rows)...",
        best_name, f"{len(y_full):,}",
    )
    best_model.fit(X_full, y_full)
    logger.info("Retraining complete.")

    # ── Stage 6: Evaluate on held-out test split ──────────────────────────────
    y_pred = best_model.predict(split.X_test_proc)

    if split.log_transformed:
        y_true_bdt = np.expm1(np.array(split.y_test))
        y_pred_bdt = np.expm1(np.array(y_pred))
    else:
        y_true_bdt = np.array(split.y_test)
        y_pred_bdt = np.array(y_pred)

    r2   = r2_score(split.y_test, y_pred)
    mae  = mean_absolute_error(y_true_bdt, y_pred_bdt)
    rmse = mean_squared_error(y_true_bdt, y_pred_bdt) ** 0.5

    logger.info("Evaluation on test split:")
    logger.info("  Model : %s", best_name)
    logger.info("  R²    : %.4f", r2)
    logger.info("  MAE   : BDT %s", f"{mae:,.0f}")
    logger.info("  RMSE  : BDT %s", f"{rmse:,.0f}")

    # ── Stage 6b: Update residual std for 95% prediction intervals ────────────
    residual_std = float((y_true_bdt - y_pred_bdt).std())
    ci_half      = 1.96 * residual_std
    with open(MODELS_DIR / "residual_std.json", "w") as _f:
        json.dump({"residual_std": residual_std}, _f)
    logger.info(
        "Prediction interval (95%%) → ± BDT %s  (residual std = BDT %s)",
        f"{ci_half:,.0f}", f"{residual_std:,.0f}",
    )

    # ── Stage 7: Save artefacts ───────────────────────────────────────────────
    version = datetime.now().strftime("%Y%m%d%H%M")

    # Overwrite canonical files (used by app.py and the Airflow DAG)
    joblib.dump(best_model,         MODELS_DIR / "best_model.pkl")
    joblib.dump(split.preprocessor, MODELS_DIR / "preprocessor.pkl")

    # Versioned copies for rollback
    joblib.dump(best_model,         MODELS_DIR / f"best_model_{version}.pkl")
    joblib.dump(split.preprocessor, MODELS_DIR / f"preprocessor_{version}.pkl")

    logger.info("Saved → models/best_model.pkl + models/best_model_%s.pkl (%s)", version, best_name)
    logger.info("Saved → models/preprocessor.pkl + models/preprocessor_%s.pkl", version)
    logger.info("=" * 60)
    logger.info("RETRAIN COMPLETE")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Retrain the saved best Flight Fare model on fresh data"
    )
    parser.add_argument(
        "--data", type=Path, default=None,
        help="Path to CSV dataset (default: config.DATA_PATH)",
    )
    args = parser.parse_args()

    from src.config import DATA_PATH as _default_data_path
    data_path = args.data if args.data else _default_data_path

    retrain(data_path)


if __name__ == "__main__":
    main()
