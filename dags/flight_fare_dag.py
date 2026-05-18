"""
dags/flight_fare_dag.py — Airflow DAG for daily Flight Fare model retrain.

Retrains only the saved best model (not all 7) to keep each daily run under
minutes rather than the hours required for full model selection.

Failure email fires immediately on any task failure.
Success email (last task) includes R², MAE, RMSE, and 95% prediction interval.

Task graph: prevalidate_csv → load_data → clean_data → feature_engineering
            → retrain_model → evaluate_model → save_artefacts → notify_success
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.utils.email import send_email

# Ensure src.* imports resolve correctly when tasks run inside Docker/Airflow
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NOTIFICATION_EMAIL = "nkurangapierrine@gmail.com"

# Raw CSV column names (before any renaming) used by task_prevalidate_csv.
# Derived from COLUMN_RENAME_MAP keys + columns that arrive already correctly named.
_RAW_REQUIRED_COLUMNS = [
    "Airline", "Class",                    # arrive with correct names
    "Base Fare (BDT)", "Tax & Surcharge (BDT)", "Total Fare (BDT)",
    "Departure Date & Time", "Duration (hrs)",
    "Stopovers", "Seasonality", "Days Before Departure",
    "Aircraft Type", "Booking Source",
]


# ── Failure callback ──────────────────────────────────────────────────────────

def failure_notification(context: dict) -> None:
    """Fires once per failed task instance — not once per DAG run."""
    dag_id    = context["dag"].dag_id
    task_id   = context["task_instance"].task_id
    run_id    = context["run_id"]
    exec_dt   = context["logical_date"].strftime("%Y-%m-%d %H:%M UTC")
    exception = context.get("exception", "Unknown error")

    send_email(
        to=NOTIFICATION_EMAIL,
        subject=f"[FAILED] [{dag_id}] Task {task_id}",
        html_content=f"""
        <h3>Task failed</h3>
        <ul>
            <li><b>DAG:</b> {dag_id}</li>
            <li><b>Failed task:</b> {task_id}</li>
            <li><b>Run ID:</b> {run_id}</li>
            <li><b>Execution date:</b> {exec_dt}</li>
            <li><b>Error:</b> {exception}</li>
        </ul>
        <p>Check logs at <a href="http://localhost:8080">http://localhost:8080</a>.</p>
        """,
    )


# ── Default task arguments ────────────────────────────────────────────────────

default_args = {
    "owner":            "flight-fare-pipeline",
    "depends_on_past":  False,
    "email_on_failure": False,   # handled via failure_notification callback
    "email_on_retry":   False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
}


# ── Logging initialiser ───────────────────────────────────────────────────────

def _init_logging() -> None:
    """Configure structured logging to OUTPUTS_DIR/pipeline.log."""
    from src.config import OUTPUTS_DIR
    from src.logger import setup_logging
    setup_logging(log_dir=OUTPUTS_DIR)


# ── Task callables ────────────────────────────────────────────────────────────

def task_prevalidate_csv(**context) -> None:
    """Fail fast before the pipeline starts if the source CSV is missing columns
    or contains no fares above the minimum threshold."""
    _init_logging()
    import logging
    import pandas as pd
    from src.config import DATA_PATH, MIN_VALID_FARE

    logger = logging.getLogger(__name__)

    df = pd.read_csv(DATA_PATH, nrows=0)   # header only — fast
    missing = [c for c in _RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Source CSV is missing expected columns: {missing}")

    # Sample row count to check minimum-fare threshold without loading full file
    df_sample = pd.read_csv(DATA_PATH, usecols=["Base Fare (BDT)"])
    valid_rows = (df_sample["Base Fare (BDT)"] >= MIN_VALID_FARE).sum()
    if valid_rows == 0:
        raise ValueError(f"No rows with Base Fare >= BDT {MIN_VALID_FARE} found in CSV.")

    context["ti"].xcom_push(key="valid_rows", value=int(valid_rows))
    logger.info("CSV pre-validation passed — %s rows above minimum fare threshold", f"{valid_rows:,}")


def task_load_data(**context) -> None:
    _init_logging()
    import logging
    from src.config import DATA_PATH
    from src.data_loader import load_data, inspect_data

    logger = logging.getLogger(__name__)
    df = load_data(DATA_PATH)
    inspect_data(df)
    context["ti"].xcom_push(key="raw_shape", value=list(df.shape))
    logger.info("Loaded %s rows x %d cols", f"{df.shape[0]:,}", df.shape[1])


def task_clean_data(**context) -> None:
    """Serialises df_clean.pkl to outputs/ for the feature engineering task."""
    _init_logging()
    import logging
    import pickle
    from src.config import DATA_PATH, OUTPUTS_DIR
    from src.data_loader import load_data
    from src.data_cleaner import clean_pipeline, audit_missing_values

    logger = logging.getLogger(__name__)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df_clean = clean_pipeline(load_data(DATA_PATH))
    audit_missing_values(df_clean)
    assert df_clean.isnull().sum().sum() == 0, "Nulls remain after cleaning"

    with open(OUTPUTS_DIR / "df_clean.pkl", "wb") as f:
        pickle.dump(df_clean, f)

    context["ti"].xcom_push(key="clean_shape", value=list(df_clean.shape))
    logger.info("Clean shape: %s x %d", f"{df_clean.shape[0]:,}", df_clean.shape[1])


def task_feature_engineering(**context) -> None:
    """Preprocessor is fitted on X_train only — no leakage into the test set."""
    _init_logging()
    import logging
    import pickle
    import joblib
    from src.config import OUTPUTS_DIR
    from src.feature_engineering import engineer_features, build_preprocessor, split_and_transform

    logger = logging.getLogger(__name__)

    with open(OUTPUTS_DIR / "df_clean.pkl", "rb") as f:
        df_clean = pickle.load(f)

    (X_train, X_test, y_train, y_test,
     X_train_proc, X_test_proc,
     feature_names, preprocessor,
     log_transformed) = split_and_transform(engineer_features(df_clean), build_preprocessor())

    with open(OUTPUTS_DIR / "split_data.pkl", "wb") as f:
        pickle.dump({
            "X_train":         X_train,
            "X_test":          X_test,
            "y_train":         y_train,
            "y_test":          y_test,
            "X_train_proc":    X_train_proc,
            "X_test_proc":     X_test_proc,
            "feature_names":   feature_names,
            "log_transformed": log_transformed,
        }, f)

    from src.config import MODELS_DIR
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    context["ti"].xcom_push(key="log_transformed", value=log_transformed)
    context["ti"].xcom_push(key="n_features",      value=X_train_proc.shape[1])
    logger.info("%d features | log_transformed=%s", X_train_proc.shape[1], log_transformed)


def task_retrain_model(**context) -> None:
    """
    Retrains on the full dataset (train + test combined).
    sklearn.base.clone() is avoided — get_params() fails on pickled XGBoost models.
    """
    _init_logging()
    import logging
    import pickle
    import joblib
    import numpy as np
    from src.config import OUTPUTS_DIR, MODELS_DIR

    logger = logging.getLogger(__name__)

    best_model = joblib.load(MODELS_DIR / "best_model.pkl")
    best_name  = type(best_model).__name__

    with open(OUTPUTS_DIR / "split_data.pkl", "rb") as f:
        d = pickle.load(f)

    X_full = np.vstack([d["X_train_proc"], d["X_test_proc"]])
    y_full = np.concatenate([d["y_train"], d["y_test"]])

    logger.info("Retraining %s on %s rows...", best_name, f"{len(y_full):,}")
    best_model.fit(X_full, y_full)

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    context["ti"].xcom_push(key="retrained_model_name", value=best_name)
    logger.info("Saved retrained %s -> models/best_model.pkl", best_name)


def task_evaluate_model(**context) -> None:
    _init_logging()
    import json
    import logging
    import pickle
    import joblib
    import numpy as np
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from src.config import OUTPUTS_DIR, MODELS_DIR

    logger = logging.getLogger(__name__)

    best_model = joblib.load(MODELS_DIR / "best_model.pkl")
    best_name  = type(best_model).__name__

    with open(OUTPUTS_DIR / "split_data.pkl", "rb") as f:
        d = pickle.load(f)

    y_pred = best_model.predict(d["X_test_proc"])

    # Reverse log1p to report errors in interpretable BDT units
    if d["log_transformed"]:
        y_true_bdt = np.expm1(np.array(d["y_test"]))
        y_pred_bdt = np.expm1(np.array(y_pred))
    else:
        y_true_bdt = np.array(d["y_test"])
        y_pred_bdt = np.array(y_pred)

    r2   = r2_score(d["y_test"], y_pred)
    mae  = mean_absolute_error(y_true_bdt, y_pred_bdt)
    rmse = mean_squared_error(y_true_bdt, y_pred_bdt) ** 0.5

    logger.info("Evaluation — %s | R2=%.4f | MAE=BDT %s | RMSE=BDT %s",
                best_name, r2, f"{mae:,.0f}", f"{rmse:,.0f}")

    residual_std = float((y_true_bdt - y_pred_bdt).std())
    ci_half      = 1.96 * residual_std
    with open(MODELS_DIR / "residual_std.json", "w") as _f:
        json.dump({"residual_std": residual_std}, _f)
    logger.info("95%% CI -> +/- BDT %s  (residual std = BDT %s)",
                f"{ci_half:,.0f}", f"{residual_std:,.0f}")

    context["ti"].xcom_push(key="r2",           value=round(r2, 4))
    context["ti"].xcom_push(key="mae",          value=round(mae, 0))
    context["ti"].xcom_push(key="rmse",         value=round(rmse, 0))
    context["ti"].xcom_push(key="residual_std", value=round(residual_std, 0))


def task_save_artefacts(**context) -> None:
    _init_logging()
    import logging
    import joblib
    from src.config import MODELS_DIR

    logger = logging.getLogger(__name__)

    best_name = type(joblib.load(MODELS_DIR / "best_model.pkl")).__name__
    for artefact in ["best_model.pkl", "preprocessor.pkl", "residual_std.json"]:
        path = MODELS_DIR / artefact
        assert path.exists(), f"Expected artefact missing: {path}"
        logger.info("  %-30s  %7.1f KB", artefact, path.stat().st_size / 1024)

    logger.info("All artefacts verified (%s). Pipeline run complete.", best_name)


def task_notify_success(**context) -> None:
    """Dedicated task rather than on_success_callback — requires the smtp_default connection."""
    _init_logging()
    import logging
    from airflow.utils.email import send_email

    logger = logging.getLogger(__name__)

    ti      = context["ti"]
    dag_id  = context["dag"].dag_id
    run_id  = context["run_id"]
    exec_dt = context["logical_date"].strftime("%Y-%m-%d %H:%M UTC")

    r2           = ti.xcom_pull(task_ids="evaluate_model", key="r2")
    mae          = ti.xcom_pull(task_ids="evaluate_model", key="mae")
    rmse         = ti.xcom_pull(task_ids="evaluate_model", key="rmse")
    residual_std = ti.xcom_pull(task_ids="evaluate_model", key="residual_std")

    metrics_html = ""
    if r2 is not None:
        ci_half = round(1.96 * residual_std, 0) if residual_std else "N/A"
        metrics_html = f"""
        <h4>Model Performance</h4>
        <ul>
            <li><b>R2  :</b> {r2}</li>
            <li><b>MAE :</b> BDT {mae:,.0f}</li>
            <li><b>RMSE:</b> BDT {rmse:,.0f}</li>
            <li><b>95% Prediction Interval:</b> +/- BDT {ci_half:,.0f}</li>
        </ul>"""

    send_email(
        to=NOTIFICATION_EMAIL,
        subject=f"[SUCCESS] [{dag_id}] Pipeline completed",
        html_content=f"""
        <h3>All tasks completed successfully</h3>
        <ul>
            <li><b>DAG:</b> {dag_id}</li>
            <li><b>Run ID:</b> {run_id}</li>
            <li><b>Execution date:</b> {exec_dt}</li>
        </ul>
        {metrics_html}
        <p>Retrained model saved to <code>outputs/best_model.pkl</code>.</p>
        """,
    )
    logger.info("Success notification sent to %s", NOTIFICATION_EMAIL)


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="flight_fare_prediction",
    default_args=default_args,
    description="Daily retrain of the best flight fare model",
    schedule="0 2 * * *",   # 02:00 UTC daily
    start_date=datetime(2025, 1, 1),
    catchup=False,           # do not back-fill missed runs
    tags=["ml", "regression", "flight-fare"],
    on_failure_callback=failure_notification,
) as dag:

    t0 = PythonOperator(task_id="prevalidate_csv",     python_callable=task_prevalidate_csv)
    t1 = PythonOperator(task_id="load_data",           python_callable=task_load_data)
    t2 = PythonOperator(task_id="clean_data",          python_callable=task_clean_data)
    t3 = PythonOperator(task_id="feature_engineering", python_callable=task_feature_engineering)
    t4 = PythonOperator(task_id="retrain_model",       python_callable=task_retrain_model)
    t5 = PythonOperator(task_id="evaluate_model",      python_callable=task_evaluate_model)
    t6 = PythonOperator(task_id="save_artefacts",      python_callable=task_save_artefacts)
    t7 = PythonOperator(task_id="notify_success",      python_callable=task_notify_success)

    t0 >> t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7
