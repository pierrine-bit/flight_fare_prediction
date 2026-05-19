"""data_cleaner.py =-step 2: clean the raw dataset: schema validation, dtype coercion, value-range checks, missing-value imputation, and duplicate removal."""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
from .config import (
    EXPECTED_DTYPES, EXPECTED_CATEGORIES, NUMERIC_BOUNDS,
    IMPUTATION_STRATEGY, MIN_VALID_FARE, COLUMN_RENAME_MAP,
    OUTPUTS_DIR, REPORTS_DIR,
)

logger = logging.getLogger(__name__)


# ── Internal helper ───────────────────────────────────────────────────────────

def _log(step: str, message: str) -> None:
    """Log a named pipeline step with a consistent prefix."""
    logger.info("[%s] %s", step, message)


# ── Step 2.1: Rename raw CSV columns to clean internal names ──────────────────

def standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_RENAME_MAP)
    _log("standardise_column_names",
         f"Renamed {len(COLUMN_RENAME_MAP)} configured columns. "
         f"Active columns: {list(df.columns)}")
    return df


# ── Step 2.2: Schema validation ──────────────────────────────────────────────

def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if any required column is missing — prevents a confusing KeyError later."""
    required = set(EXPECTED_DTYPES.keys())
    present  = set(df.columns)
    missing  = required - present

    if missing:
        raise ValueError(
            f"Schema validation failed — {len(missing)} required column(s) missing:\n"
            + "\n".join(f"  • {col}  ({EXPECTED_DTYPES[col][1]})" for col in sorted(missing))
            + "\n\nCheck COLUMN_RENAME_MAP in config.py or verify the CSV structure."
        )

    _log("validate_schema",
         f"Schema OK — all {len(required)} required columns present.")


# ── Step 2.3: Data type validation and coercion ──────────────────────────────

def validate_dtypes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate column dtypes and apply minimum coercion on mismatch:
    datetime → to_datetime | float64/int64 → to_numeric | object → astype(str)
    """
    df = df.copy()
    report_rows = []

    for col, (expected, description) in EXPECTED_DTYPES.items():
        if col not in df.columns:
            report_rows.append({
                "column":      col,
                "expected":    expected,
                "actual":      "MISSING",
                "status":      "SKIP",
                "action":      "Column absent in DataFrame — check COLUMN_RENAME_MAP",
                "description": description,
            })
            continue

        actual    = str(df[col].dtype)
        already_ok = (expected in actual) or (actual == expected)
        action     = "—"

        if not already_ok:
            if expected == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
                action  = "pd.to_datetime(errors='coerce')"
            elif expected in ("float64", "int64"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                action  = f"pd.to_numeric → will cast to {expected} post-imputation"
            elif expected == "object":
                df[col] = df[col].astype(str)
                action  = "astype(str)"

        coerced_null_count = df[col].isnull().sum()
        report_rows.append({
            "column":       col,
            "expected":     expected,
            "actual":       str(df[col].dtype),
            "status":       "OK" if already_ok else "FIXED",
            "action":       action,
            "nulls_after":  coerced_null_count,
            "description":  description,
        })

    report  = pd.DataFrame(report_rows).set_index("column")
    n_fixed = (report["status"] == "FIXED").sum()

    # Log the full report including the human-readable description so reviewers
    # can verify each column's expected type at a glance without consulting config.py
    logger.info("── Dtype Validation Report ──────────────────────────────────\n%s",
                report[["description", "expected", "actual", "status", "action", "nulls_after"]].to_string())
    _log("validate_dtypes",
         f"{n_fixed} column(s) corrected. "
         f"New NaNs introduced by coercion: {report['nulls_after'].sum()}")
    return df, report


# ── Step 2.4: Numeric value-range validation ──────────────────────────────────

def validate_value_ranges(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove rows outside the physically valid bounds defined in config.NUMERIC_BOUNDS."""
    df = df.copy()
    report_rows = []

    logger.info("── Numeric Value-Range Validation ───────────────────────────")
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col not in df.columns:
            continue

        below_mask = df[col] < lo
        above_mask = df[col] > hi
        bad_mask   = below_mask | above_mask

        n_below  = int(below_mask.sum())
        n_above  = int(above_mask.sum())
        n_bad    = int(bad_mask.sum())
        pct_bad  = round(n_bad / len(df) * 100, 3)

        if n_bad > 0:
            bad_stats = df.loc[bad_mask, col].describe().round(2).to_dict()
            _log("validate_value_ranges",
                 f"{col}: removing {n_bad} rows ({pct_bad}%) "
                 f"[{n_below} below {lo:,} | {n_above} above {hi:,}]. "
                 f"Bad-row stats: {bad_stats}")
            df = df[~bad_mask].reset_index(drop=True)

        report_rows.append({
            "column":      col,
            "bound_lo":    lo,
            "bound_hi":    hi,
            "n_below":     n_below,
            "n_above":     n_above,
            "n_removed":   n_bad,
            "pct_removed": pct_bad,
        })

    range_report  = pd.DataFrame(report_rows).set_index("column")
    total_removed = range_report["n_removed"].sum()
    _log("validate_value_ranges",
         f"Total rows removed by range checks: {total_removed:,}. Remaining: {len(df):,}")
    return df, range_report


# ── Step 2.5: Categorical value validation ─────────────────────────────────────

def validate_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """Audit categorical columns against expected value sets. Does NOT remove rows."""
    report_rows = []
    logger.info("── Categorical Value Validation ─────────────────────────────")

    for col, expected_vals in EXPECTED_CATEGORIES.items():
        if col not in df.columns or expected_vals is None:
            continue

        actual_vals = set(df[col].dropna().unique())
        unexpected  = actual_vals - set(expected_vals)

        if unexpected:
            counts = df[col].value_counts()
            for val in sorted(unexpected):
                report_rows.append({
                    "column":         col,
                    "unexpected_val": val,
                    "count":          counts.get(val, 0),
                    "pct":            round(counts.get(val, 0) / len(df) * 100, 2),
                    "expected_set":   str(expected_vals),
                })
            _log("validate_categorical_values",
                 f"{col}: {len(unexpected)} unexpected value(s): {sorted(unexpected)}")
        else:
            _log("validate_categorical_values",
                 f"{col}: all values within expected set ✓")

    if report_rows:
        anomaly_report = pd.DataFrame(report_rows)
        logger.info("Anomaly detail:\n%s", anomaly_report.to_string(index=False))
    else:
        anomaly_report = pd.DataFrame()
        logger.info("No categorical anomalies detected.")

    return anomaly_report


# ── Step 2.6: Categorical value normalisation ────────────────────────────────

def normalise_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise known categorical inconsistencies.
    'First Class' → 'First' aligns with the expected Class values in config.
    """
    df = df.copy()
    if "Class" in df.columns:
        before = (df["Class"] == "First Class").sum()
        df["Class"] = df["Class"].replace("First Class", "First")
        if before > 0:
            _log("normalise_categorical_values",
                 f"Class: renamed {before:,} 'First Class' → 'First'")
    return df


# ── Step 2.7: Remove duplicate rows ──────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before    = len(df)
    df        = df.drop_duplicates().reset_index(drop=True)
    n_removed = before - len(df)
    _log("remove_duplicates",
         f"Removed {n_removed} fully duplicate rows. Remaining: {len(df):,}")
    return df


# ── Step 2.8: Missing-value audit (read-only) ─────────────────────────────────

def audit_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a null-count audit DataFrame. Read-only — does not modify df."""
    from .config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET

    null_counts  = df.isnull().sum()
    null_pct     = (null_counts / len(df) * 100).round(3)
    all_features = set(CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET])

    audit_rows = []
    for col in df.columns:
        audit_rows.append({
            "column":     col,
            "dtype":      str(df[col].dtype),
            "null_count": null_counts[col],
            "null_%":     null_pct[col],
            "strategy":   IMPUTATION_STRATEGY.get(col, "—"),
            "is_feature": "✓" if col in all_features else "",
        })

    audit_df   = pd.DataFrame(audit_rows).set_index("column")
    missing_df = audit_df[audit_df["null_count"] > 0]

    if missing_df.empty:
        logger.info("── Missing Value Audit (total nulls: 0) ── No missing values found.")
    else:
        logger.info("── Missing Value Audit (total nulls: %d) ──\n%s",
                    null_counts.sum(),
                    missing_df[["dtype", "null_count", "null_%", "strategy", "is_feature"]].to_string())
    return audit_df


# ── Step 2.9: Robust missing-value imputation ─────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using the strategy defined in config.IMPUTATION_STRATEGY.
    Strategy is config-driven: adding or removing columns from the dataset requires
    only a change to IMPUTATION_STRATEGY, not to this function.

    Supported strategies
    ────────────────────
    'median'   → fill with column median (numerics)
    'mean'     → fill with column mean   (numerics)
    'mode'     → fill with most-frequent value (categoricals)
    'Unknown'  → literal string fill    (sparse categoricals like Aircraft_Type)
    'drop'     → drop rows with null in this column (e.g. Date — cannot impute time)

    Catch-all fallback
    ──────────────────
    Any column with remaining nulls that is NOT in IMPUTATION_STRATEGY gets:
    - numerics  → median  (safe, distribution-preserving)
    - objects   → mode    (least-biasing fill for categoricals)
    A warning is logged so the gap is visible without causing a pipeline crash.

    Post-imputation consistency check
    ──────────────────────────────────
    If both Base_Fare and Tax_Surcharge are present, Total_Fare is recomputed to
    guarantee internal consistency (a hard assertion verifies zero remaining nulls).
    """
    df = df.copy()

    audit_before = audit_missing_values(df)
    if audit_before["null_count"].sum() == 0:
        _log("handle_missing_values", "No missing values — imputation skipped.")
        return df

    logger.info("── Applying Imputation (config-driven via IMPUTATION_STRATEGY) ─")

    # ── Primary pass: apply each column's declared strategy ──────────────────
    # Iterates over IMPUTATION_STRATEGY (config.py) — adding a new column to the
    # dataset only requires a config change, not a code change here.
    for col, strategy in IMPUTATION_STRATEGY.items():
        if col not in df.columns or not df[col].isnull().any():
            continue  # column absent or already clean — skip

        n_null = int(df[col].isnull().sum())

        if strategy == "median":
            # Median is preferred over mean for skewed numeric columns (e.g. fare)
            # because it is resistant to outlier influence.
            if df[col].isnull().all():
                logger.warning("[handle_missing_values] '%s' is entirely null → filling with 0.", col)
                df[col] = df[col].fillna(0)
            else:
                fill_val = df[col].median()
                df[col]  = df[col].fillna(fill_val)
                _log("handle_missing_values",
                     f"{col}: filled {n_null} NaN(s) with MEDIAN = {fill_val:,.4f}")

        elif strategy == "mean":
            # Mean fill for symmetric numeric distributions where outliers are not a concern.
            if df[col].isnull().all():
                logger.warning("[handle_missing_values] '%s' is entirely null → filling with 0.", col)
                df[col] = df[col].fillna(0)
            else:
                fill_val = df[col].mean()
                df[col]  = df[col].fillna(fill_val)
                _log("handle_missing_values",
                     f"{col}: filled {n_null} NaN(s) with MEAN = {fill_val:,.4f}")

        elif strategy == "mode":
            # Mode fill for categoricals — preserves the existing frequency distribution
            # rather than introducing a new or synthetic category.
            if df[col].isnull().all():
                logger.warning("[handle_missing_values] '%s' is entirely null → filling with 'Unknown'.", col)
                df[col] = df[col].fillna("Unknown")
            else:
                fill_val = df[col].mode()[0]
                df[col]  = df[col].fillna(fill_val)
                _log("handle_missing_values",
                     f"{col}: filled {n_null} NaN(s) with MODE = '{fill_val}'")

        elif strategy == "drop":
            # Temporal columns (e.g. Date) cannot be imputed — a fake date would
            # corrupt Month, Weekday, and Days_Before_Departure feature derivations.
            n_before = len(df)
            df = df.dropna(subset=[col]).reset_index(drop=True)
            _log("handle_missing_values",
                 f"{col}: dropped {n_before - len(df)} row(s) — cannot impute '{col}'")

        else:
            # Any unrecognised strategy string is used as a literal fill value —
            # supports sparse categoricals like Aircraft_Type → 'Unknown'.
            df[col] = df[col].fillna(strategy)
            _log("handle_missing_values",
                 f"{col}: filled {n_null} NaN(s) with literal '{strategy}'")

    # ── Special handling: Date column (temporal — must drop, not impute) ─────
    if "Date" in df.columns and "Date" not in IMPUTATION_STRATEGY and df["Date"].isnull().any():
        n_before = len(df)
        df = df.dropna(subset=["Date"]).reset_index(drop=True)
        _log("handle_missing_values",
             f"Date: dropped {n_before - len(df)} row(s) — temporal features cannot be imputed.")

   
    # requiring edits to this function — only IMPUTATION_STRATEGY needs updating.
    still_null_cols = [c for c in df.columns if df[c].isnull().any()]
    if still_null_cols:
        logger.warning("[handle_missing_values] %d column(s) with nulls not in "
                       "IMPUTATION_STRATEGY — applying generic fallback. "
                       "Add these to config.IMPUTATION_STRATEGY for explicit control: %s",
                       len(still_null_cols), still_null_cols)
        for col in still_null_cols:
            if df[col].isnull().all():
                logger.warning("  '%s' is entirely null — skipping (cannot compute fill).", col)
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].median()
                df[col]  = df[col].fillna(fill_val)
                logger.warning("  '%s' → generic MEDIAN fallback = %.4f", col, fill_val)
            else:
                fill_val = df[col].mode()[0]
                df[col]  = df[col].fillna(fill_val)
                logger.warning("  '%s' → generic MODE fallback = '%s'", col, fill_val)

    # ── Post-imputation consistency check ─────────────────────────────────────
    if {"Base_Fare", "Tax_Surcharge"}.issubset(df.columns):
        df["Total_Fare"] = df["Base_Fare"] + df["Tax_Surcharge"]
        _log("handle_missing_values",
             "Total_Fare recomputed from Base_Fare + Tax_Surcharge (consistency check).")

    # ── Final verification: assert zero remaining nulls ───────────────────────
    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls == 0:
        _log("handle_missing_values", f"✓ All missing values resolved. Shape: {df.shape}")
    else:
        still_null = df.columns[df.isnull().any()].tolist()
        logger.error("[handle_missing_values] %d null(s) still remain in: %s",
                     remaining_nulls, still_null)

    assert remaining_nulls == 0, (
        f"{remaining_nulls} missing value(s) remain after imputation. "
        f"Columns: {df.columns[df.isnull().any()].tolist()}"
    )
    return df


# ── Step 2.10: Fix invalid fare entries ──────────────────────────────────────

def fix_invalid_fares(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Drop rows where Base_Fare < MIN_VALID_FARE.
    2. Recompute Total_Fare = Base_Fare + Tax_Surcharge for internal consistency.
    """
    df = df.copy()

    n_invalid = int((df["Base_Fare"] < MIN_VALID_FARE).sum())
    if n_invalid > 0:
        invalid_stats = df.loc[df["Base_Fare"] < MIN_VALID_FARE, "Base_Fare"].describe()
        _log("fix_invalid_fares",
             f"Removing {n_invalid} rows with Base_Fare < BDT {MIN_VALID_FARE:,}. "
             f"Stats: {invalid_stats.round(2).to_dict()}")
        df = df[df["Base_Fare"] >= MIN_VALID_FARE].reset_index(drop=True)

    df["Total_Fare"] = df["Base_Fare"] + df["Tax_Surcharge"]
    _log("fix_invalid_fares",
         f"Total_Fare recomputed. "
         f"Range: BDT {df['Total_Fare'].min():,.0f} – {df['Total_Fare'].max():,.0f}. "
         f"Mean: BDT {df['Total_Fare'].mean():,.0f}")
    return df


# ── Public pipeline ───────────────────────────────────────────────────────────

def clean_pipeline(df: pd.DataFrame,
                   save_dir: Path = REPORTS_DIR) -> pd.DataFrame:
    """
    Execute all cleaning steps in order and save validation reports to save_dir.

    Order dependency
    ────────────────
    1. standardise_column_names    
    2. validate_schema             
    3. validate_dtypes             
    4. validate_value_ranges       
    5. validate_categorical_values 
    6. normalise_categorical_values
    7. remove_duplicates           
    8. audit_missing_values        
    9. handle_missing_values       
    10. fix_invalid_fares          

    """
    logger.info("\n%s\nDATA CLEANING PIPELINE\n%s", "=" * 65, "=" * 65)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = standardise_column_names(df)
    validate_schema(df)

    df, dtype_report = validate_dtypes(df)
    # Save full report including description column for explicit documentation
    dtype_report.to_csv(save_dir / "validation_dtypes.csv")
    logger.info("Dtype report saved → %s", save_dir / "validation_dtypes.csv")

    df, range_report = validate_value_ranges(df)
    range_report.to_csv(save_dir / "validation_ranges.csv")
    logger.info("Range report saved → %s", save_dir / "validation_ranges.csv")

    cat_report = validate_categorical_values(df)
    if not cat_report.empty:
        cat_report.to_csv(save_dir / "validation_categoricals.csv", index=False)
        logger.info("Categorical anomaly report saved → %s",
                    save_dir / "validation_categoricals.csv")

    df = normalise_categorical_values(df)
    df = remove_duplicates(df)

    missing_report = audit_missing_values(df)
    missing_report.to_csv(save_dir / "validation_missing.csv")
    logger.info("Missing value audit saved → %s", save_dir / "validation_missing.csv")

    df = handle_missing_values(df)
    df = fix_invalid_fares(df)

    logger.info("[clean_pipeline] ✓ Complete. Final shape: %s rows × %d columns",
                f"{df.shape[0]:,}", df.shape[1])
    return df
