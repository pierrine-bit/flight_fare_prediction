

from __future__ import annotations
import logging
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import xgboost as xgb
import lightgbm as lgb

from .config import RANDOM_STATE, CV_FOLDS, GRID_CV_FOLDS, N_ITER

logger = logging.getLogger(__name__)


# ── Model catalogue ───────────────────────────────────────────────────────────

def get_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge":             Ridge(),                                     # convex solver — no random_state
        "Lasso":             Lasso(random_state=RANDOM_STATE, max_iter=10_000),
        "Decision Tree":     DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest":     RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost":           xgb.XGBRegressor(
                                 random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
                                 tree_method="hist",
                             ),
        "LightGBM":          lgb.LGBMRegressor(
                                 random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
                             ),
    }


# ── Hyperparameter grids ──────────────────────────────────────────────────────

def get_grid_params() -> dict:
    """
    Exhaustive GridSearchCV grids — every combination is evaluated.

    Which models use GridSearchCV and why
    ──────────────────────────────────────
    Ridge       : single continuous hyperparameter (α); 15-value log-spaced grid covers
                  6 orders of magnitude. 15 × 5 folds = 75 fits — trivially fast.

    Lasso       : same α scale as Ridge; L1 path is more sensitive to fine-grained α,
                  so the grid is denser in the [1e-4, 0.1] region where L1 regularisation
                  transitions from near-OLS to strong feature selection.
                  9 α × 1 max_iter × 5 folds = 45 fits.

    Decision Tree : unlike ensembles, a single decision tree has a compact, well-understood
                  parameter space. An exhaustive 3-axis grid (max_depth × min_samples_leaf ×
                  min_samples_split) covers 5 × 4 × 3 = 60 combinations × 5 folds = 300 fits —
                  fast enough for GridSearchCV while being more comprehensive than random sampling.
                  ccp_alpha and max_features are excluded as secondary parameters that add
                  dimensionality without proportionate gain.
    """
    return {
        "Ridge": {
            # Log-spaced from very small (near-OLS) to large (heavy shrinkage)
            "alpha": [1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000],
        },
        "Lasso": {
            # Denser near the transition zone (1e-4 → 0.1) where feature selection occurs
            "alpha":    [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 1.0],
            "max_iter": [10_000],   # high enough to guarantee convergence at small α values
        },
        "Decision Tree": {
            # Exhaustive 3-axis grid: 5 × 4 × 3 = 60 combinations × 5 folds = 300 fits
            # max_depth controls the complexity of the tree (bias-variance trade-off)
            "max_depth":         [4, 5, 6, 8, 10],
            # min_samples_leaf prevents leaves with very few samples (over-fitting guard)
            "min_samples_leaf":  [1, 2, 5, 10],
            # min_samples_split controls the minimum size of a node before it can be split
            "min_samples_split": [2, 5, 10],
        },
    }


def get_random_params() -> dict:
    """
    RandomizedSearchCV distributions for high-dimensional ensemble search spaces.

    Why RandomizedSearchCV for these models
    ─────────────────────────────────────────
    Random Forest : 5 × 5 × 4 × 3 × 4 × 1 = 1,200 combinations — full GridSearch would
                   require 1,200 × 5 folds = 6,000 fits × ~3 min each → infeasible.
                   RandomizedSearchCV with n_iter=50 samples 50 diverse points, giving
                   broad coverage while keeping wall-clock time under 45 minutes.

    XGBoost/LightGBM : 5 × 6 × 6 × 5 × 5 × 6 × 5 × 3 ≈ 810,000 combinations.
                   Random sampling is the only practical approach. n_iter=50 with
                   RANDOM_STATE ensures reproducibility.

    Note: Hyperparameter interactions (e.g. learning_rate × n_estimators,
    subsample × colsample_bytree) are best handled by random sampling rather than
    a grid, as correlated parameters cluster around the diagonal of a grid search.
    """
    return {
        "Random Forest": {
            "n_estimators":      [100, 200, 300, 400, 500],
            "max_depth":         [6, 8, 10, 12, None],
            "min_samples_leaf":  [1, 2, 4, 5],
            "min_samples_split": [2, 5, 10],
            "max_features":      ["sqrt", "log2", 0.3, 0.5],
            "bootstrap":         [True],
        },
        "XGBoost": {
            "n_estimators":     [200, 300, 400, 500, 600],
            "learning_rate":    [0.005, 0.01, 0.03, 0.05, 0.1, 0.15],
            "max_depth":        [3, 4, 5, 6, 7, 8],
            "subsample":        [0.5, 0.6, 0.7, 0.8, 1.0],
            "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
            "reg_alpha":        [0, 0.05, 0.1, 0.3, 0.5, 1.0],
            "reg_lambda":       [0.5, 1, 1.5, 2, 3],
            "min_child_weight": [1, 3, 5],
        },
        "LightGBM": {
            "n_estimators":      [200, 300, 400, 500, 600],
            "learning_rate":     [0.005, 0.01, 0.03, 0.05, 0.1],
            # num_leaves replaces max_depth as the primary complexity control in LightGBM
            "num_leaves":        [31, 63, 95, 127, 191, 255],
            "min_child_samples": [5, 10, 20, 30, 50],
            "subsample":         [0.5, 0.6, 0.7, 0.8, 1.0],
            "colsample_bytree":  [0.5, 0.6, 0.7, 0.8, 1.0],
            "reg_alpha":         [0, 0.05, 0.1, 0.3, 0.5],
            "reg_lambda":        [0.5, 1, 1.5, 2, 3],
            "max_depth":         [-1, 6, 8, 10],  # -1 = unlimited (LightGBM default)
        },
    }


# ── Evaluation helper ─────────────────────────────────────────────────────────

def _evaluate(
    name: str,
    y_true,
    y_pred,
    cv_mean: float,
    cv_std: float,
    elapsed: float,
    log_transformed: bool = False,
) -> dict:
    """
    Package test-set metrics into a standard dict.
    When log_transformed=True, MAE/RMSE are in original BDT space via np.expm1.
    """
    r2 = round(r2_score(y_true, y_pred), 4)

    if log_transformed:
        y_true_bdt = np.expm1(np.array(y_true))
        y_pred_bdt = np.expm1(np.array(y_pred))
        mae    = round(mean_absolute_error(y_true_bdt, y_pred_bdt), 0)
        rmse   = round(mean_squared_error(y_true_bdt, y_pred_bdt) ** 0.5, 0)
        r2_bdt = round(r2_score(y_true_bdt, y_pred_bdt), 4)
    else:
        mae    = round(mean_absolute_error(y_true, y_pred), 0)
        rmse   = round(mean_squared_error(y_true, y_pred) ** 0.5, 0)
        r2_bdt = r2

    return {
        "Model":           name,
        "R² (log)":        r2,
        "R² (BDT)":        r2_bdt,
        "MAE (BDT)":       mae,
        "RMSE (BDT)":      rmse,
        "CV R² Mean":      round(cv_mean, 4),
        "CV R² Std":       round(cv_std, 4),
        "Train Time (s)":  round(elapsed, 1),
        "Log Transformed": log_transformed,
    }


# ── Per-fold CV detail ────────────────────────────────────────────────────────

def _detailed_cv(model, X: np.ndarray, y, kf: KFold, name: str) -> tuple[float, float]:
    """
    Run 5-fold cross-validation on the best estimator and log per-fold R² with
    ASCII bars. Returns (mean, std) for inclusion in the comparison table.
    """
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=1)

    fold_lines = [f"    {name} — {CV_FOLDS}-Fold CV Detail:"]
    for i, s in enumerate(scores, 1):
        bar = "█" * max(0, int(s * 35))
        fold_lines.append(f"      Fold {i}: {s:.4f}  {bar}")
    stability = "(stable ✓)" if scores.std() < 0.03 else "(⚠ high fold variance)"
    fold_lines.append(f"      {'─' * 45}")
    fold_lines.append(f"      Mean : {scores.mean():.4f}  |  Std : {scores.std():.4f}  {stability}")
    logger.info("\n%s", "\n".join(fold_lines))

    return scores.mean(), scores.std()


# ── Per-model CV report ────────────────────────────────────────────────────────

def build_cv_report(
    best_models: dict,
    X_train: np.ndarray,
    y_train,
    kf: KFold,
) -> pd.DataFrame:
    """
    Per-fold CV R² for every model — comprehensive cross-validation report.

    Columns returned
    ─────────────────
    Fold 1 … Fold N  : R² score for each fold
    Mean             : average CV R² — primary model-selection criterion
    Std              : standard deviation across folds — stability measure
    Min / Max        : worst and best fold performance — reveals outlier folds
    Range            : Max − Min — total spread; large range = unstable generalisation
    CI_lower / CI_upper : 95% confidence interval via t-distribution (df=N-1, 2.776 = t_{4,0.025})
    Stable           : ✓ if Std < 0.03 (low fold variance), ⚠ if Std ≥ 0.03 (investigate)

    Interpretation
    ──────────────
    A model is considered robust when:
      - Std < 0.03  (scores are consistent across different data subsets)
      - Range < 0.05 (no single fold is dramatically better or worse)
      - CI_lower > 0 (confidence interval does not include zero)
    """
    logger.info("\n%s\nCROSS-VALIDATION REPORT (%d-Fold) — ALL MODELS\n%s",
                "─" * 65, CV_FOLDS, "─" * 65)

    rows = {}
    for name, model in best_models.items():
        scores  = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2", n_jobs=1)
        mean    = scores.mean()
        std     = scores.std()
        # 95% CI using t-distribution (df = n_folds - 1); t_{4, 0.025} = 2.776
        ci_half = 2.776 * (std / np.sqrt(len(scores)))

        rows[name] = {
            **{f"Fold {i+1}": round(s, 4) for i, s in enumerate(scores)},
            "Mean":      round(mean, 4),
            "Std":       round(std, 4),
            "Min":       round(scores.min(), 4),    # worst fold — identifies data subsets the model struggles on
            "Max":       round(scores.max(), 4),    # best fold
            "Range":     round(scores.max() - scores.min(), 4),  # total spread across folds
            "CI_lower":  round(mean - ci_half, 4),
            "CI_upper":  round(mean + ci_half, 4),
            "Stable":    "✓" if std < 0.03 else "⚠",
        }

    cv_report = pd.DataFrame(rows).T

    # Sort by Mean descending for easy comparison
    cv_sorted = cv_report.sort_values("Mean", ascending=False)
    logger.info("\n%s", cv_sorted.to_string())
    logger.info(
        "95%% CI = Mean ± 2.776 × (Std / √%d)  |  Stable = Std < 0.03  |  "
        "Range = Max − Min fold score",
        CV_FOLDS,
    )

    # Flag any model where a single fold is an outlier (Range > 0.05)
    unstable = cv_report[cv_report["Range"] > 0.05]
    if not unstable.empty:
        logger.warning("Models with high fold-score range (>0.05) — investigate data heterogeneity:\n%s",
                       unstable[["Mean", "Std", "Min", "Max", "Range"]].to_string())

    return cv_report


# ── Main training function ────────────────────────────────────────────────────

def train_all_models(
    X_train: np.ndarray,
    y_train,
    X_test: np.ndarray,
    y_test,
    kf: KFold | None = None,
    log_transformed: bool = False,
) -> tuple[dict, list[dict]]:
    """
    Train and tune all models. Returns fitted best estimators and metrics list.

    Tuning strategy
    ───────────────
    Linear Regression            → No tuning (baseline reference).
    Ridge, Lasso                 → GridSearchCV (exhaustive) — single α parameter; compact grid.
    Decision Tree                → GridSearchCV (exhaustive) — 60 combos (5×4×3); fast enough for full search.
    Random Forest, XGBoost, LGB  → RandomizedSearchCV (n_iter=50) — search spaces too large for exhaustive search.
    """
    if kf is None:
        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    models        = get_models()
    grid_params   = get_grid_params()
    random_params = get_random_params()

    best_models: dict       = {}
    results:     list[dict] = []

    logger.info("\n%s\nMODEL TRAINING & HYPERPARAMETER TUNING\n"
                "  GridSearchCV       : cv=%d folds, exhaustive\n"
                "    -> Ridge (15 alpha), Lasso (9 alpha), Decision Tree (60 combos)\n"
                "  RandomizedSearchCV : cv=%d folds, n_iter=%d\n"
                "    -> Random Forest, XGBoost, LightGBM (high-dim search spaces)\n"
                "  Final CV report    : %d-fold (shared KFold — identical splits for all models)\n%s",
                "=" * 65, GRID_CV_FOLDS, GRID_CV_FOLDS, N_ITER, CV_FOLDS, "=" * 65)

    for name, model in models.items():
        logger.info("\n%s\n  Training: %s", "─" * 60, name)
        t0 = time.time()

        if name in grid_params:
            n_combos = _grid_size(grid_params[name])
            logger.info("  Method: GridSearchCV (exhaustive, cv=%d) — %d combos × %d folds = %d fits",
                        GRID_CV_FOLDS, n_combos, GRID_CV_FOLDS, n_combos * GRID_CV_FOLDS)
            searcher = GridSearchCV(
                estimator=model, param_grid=grid_params[name],
                cv=GRID_CV_FOLDS, scoring="r2",
                n_jobs=-1, refit=True, verbose=0, return_train_score=True,
            )
            searcher.fit(X_train, y_train)
            best      = searcher.best_estimator_
            best_idx  = searcher.best_index_
            train_r2  = searcher.cv_results_["mean_train_score"][best_idx]
            val_r2    = searcher.cv_results_["mean_test_score"][best_idx]
            gap       = train_r2 - val_r2
            logger.info("  Best params: %s", searcher.best_params_)
            logger.info("  CV train R²=%.4f | CV val R²=%.4f | gap=%.4f %s",
                        train_r2, val_r2, gap,
                        "(slight overfit)" if gap > 0.05 else "(healthy)")

        elif name in random_params:
            logger.info("  Method: RandomizedSearchCV (n_iter=%d, cv=%d)", N_ITER, GRID_CV_FOLDS)
            searcher = RandomizedSearchCV(
                estimator=model, param_distributions=random_params[name],
                n_iter=N_ITER, cv=GRID_CV_FOLDS, scoring="r2",
                n_jobs=1, random_state=RANDOM_STATE,
                refit=True, verbose=0, return_train_score=True,
            )
            searcher.fit(X_train, y_train)
            best     = searcher.best_estimator_
            best_idx = searcher.best_index_
            train_r2 = searcher.cv_results_["mean_train_score"][best_idx]
            val_r2   = searcher.cv_results_["mean_test_score"][best_idx]
            gap      = train_r2 - val_r2
            logger.info("  Best params: %s", searcher.best_params_)
            logger.info("  CV train R²=%.4f | CV val R²=%.4f | gap=%.4f %s",
                        train_r2, val_r2, gap,
                        "(slight overfit)" if gap > 0.05 else "(healthy)")

        else:
            logger.info("  Method: No tuning (baseline)")
            best = model.fit(X_train, y_train)

        cv_mean, cv_std = _detailed_cv(best, X_train, y_train, kf, name)

        y_pred  = best.predict(X_test)
        elapsed = time.time() - t0
        metrics = _evaluate(name, y_test, y_pred, cv_mean, cv_std, elapsed,
                            log_transformed=log_transformed)
        results.append(metrics)
        best_models[name] = best

        r2_label = metrics["R² (log)"] if log_transformed else metrics["R² (BDT)"]
        logger.info("  ✓ %s: R²=%s%.4f | MAE=BDT %s | RMSE=BDT %s | CV=%.4f±%.4f | [%.1fs]",
                    name, "log " if log_transformed else "", r2_label,
                    f"{metrics['MAE (BDT)']:,.0f}", f"{metrics['RMSE (BDT)']:,.0f}",
                    metrics["CV R² Mean"], metrics["CV R² Std"], elapsed)

    logger.info("✓ All %d models trained and evaluated.", len(best_models))
    return best_models, results


# ── Internal helper ───────────────────────────────────────────────────────────

def _grid_size(param_grid: dict) -> int:
    size = 1
    for v in param_grid.values():
        size *= len(v)
    return size
