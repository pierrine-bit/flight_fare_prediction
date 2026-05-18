
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from .config import (
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET,
    RANDOM_STATE, TEST_SIZE, LOG_TRANSFORM_TARGET,
)

logger = logging.getLogger(__name__)


# ── Split result container ────────────────────────────────────────────────────

@dataclass
class SplitData:
    X_train:         pd.DataFrame
    X_test:          pd.DataFrame
    y_train:         pd.Series
    y_test:          pd.Series
    X_train_proc:    np.ndarray
    X_test_proc:     np.ndarray
    feature_names:   list[str]
    preprocessor:    ColumnTransformer
    log_transformed: bool

    def as_tuple(self):
        """Return fields as an ordered 9-tuple (used internally by __iter__)."""
        return (self.X_train, self.X_test, self.y_train, self.y_test,
                self.X_train_proc, self.X_test_proc,
                self.feature_names, self.preprocessor, self.log_transformed)

    def __iter__(self):
        """Support tuple unpacking: X_train, X_test, ... = split_and_transform(...)"""
        return iter(self.as_tuple())


# ── Step 3.1: Feature engineering ────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive Month, Weekday, Stop_Type, and Route from existing columns.
    Route is for EDA labels only — not passed to the model.
    """
    df = df.copy()

    df["Month"]   = df["Date"].dt.month
    df["Weekday"] = df["Date"].dt.dayofweek
    df["Stop_Type"] = df["Stop_Raw"].apply(
        lambda x: "Non_Stop" if str(x).strip().lower() == "direct" else "With_Stop"
    )
    df["Route"] = df["Source"] + "→" + df["Destination"]

    logger.info("New features added: Month, Weekday, Stop_Type, Route")
    logger.info("Stop_Type value counts:\n%s", df["Stop_Type"].value_counts().to_string())
    logger.info("Season value counts:\n%s",    df["Season"].value_counts().to_string())
    logger.info("Month range: %d – %d | Weekday range: %d – %d",
                df["Month"].min(), df["Month"].max(),
                df["Weekday"].min(), df["Weekday"].max())
    return df


# ── Step 3.2: Preprocessor ────────────────────────────────────────────────────

def build_preprocessor(
    cat_features: list[str] = CATEGORICAL_FEATURES,
    num_features: list[str] = NUMERICAL_FEATURES,
) -> ColumnTransformer:
    """Return an unfitted ColumnTransformer: StandardScaler on numerics, OHE on categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract the ordered feature names produced by the fitted preprocessor."""
    ohe_names = list(
        preprocessor.named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    return NUMERICAL_FEATURES + ohe_names


# ── Step 3.3: Split and transform ────────────────────────────────────────────

def split_and_transform(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer | None = None,
    log_transform_target: bool = LOG_TRANSFORM_TARGET,
) -> SplitData:
    """
    Split into train/test, optionally log-transform the target, fit the
    preprocessor on X_train only, and transform both splits.
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()

    cat_present = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    num_present = [c for c in NUMERICAL_FEATURES   if c in df.columns]

    if cat_present != CATEGORICAL_FEATURES or num_present != NUMERICAL_FEATURES:
        logger.warning("Some configured features are missing.")
        logger.warning("  Expected cat : %s", CATEGORICAL_FEATURES)
        logger.warning("  Found    cat : %s", cat_present)
        logger.warning("  Expected num : %s", NUMERICAL_FEATURES)
        logger.warning("  Found    num : %s", num_present)
        preprocessor = build_preprocessor(cat_present, num_present)

    X = df[cat_present + num_present]
    y = df[TARGET].astype(float)

    if log_transform_target:
        logger.info("Applying log1p transform to '%s'", TARGET)
        logger.info("  Before: min=%s  max=%s  mean=%s  skew=%.3f",
                    f"{y.min():,.0f}", f"{y.max():,.0f}", f"{y.mean():,.0f}", y.skew())
        y = np.log1p(y)
        logger.info("  After : min=%.3f  max=%.3f  mean=%.3f  skew=%.3f",
                    y.min(), y.max(), y.mean(), y.skew())
        logger.info("  → Skew reduced from %.3f to %.3f  (closer to 0 = more symmetric)",
                    df[TARGET].skew(), y.skew())
    else:
        logger.info("No target transform (log_transform_target=False)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    X_train_proc  = preprocessor.fit_transform(X_train)
    X_test_proc   = preprocessor.transform(X_test)
    feature_names = _get_feature_names(preprocessor)

    logger.info("Split complete: Train=%s rows | Test=%s rows | Features after OHE=%d | Target=%s",
                f"{X_train.shape[0]:,}", f"{X_test.shape[0]:,}",
                X_train_proc.shape[1],
                "log1p(BDT)" if log_transform_target else "raw BDT")

    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_proc=X_train_proc,
        X_test_proc=X_test_proc,
        feature_names=feature_names,
        preprocessor=preprocessor,
        log_transformed=log_transform_target,
    )
