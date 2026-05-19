
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import cross_val_score, learning_curve, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from .config import (
    OUTPUTS_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR,
    PLOT_DPI, PLOT_STYLE, PLOT_PALETTE,
    CV_FOLDS, RANDOM_STATE, GRID_CV_FOLDS,
)

logger = logging.getLogger(__name__)

sns.set_theme(style=PLOT_STYLE, palette=PLOT_PALETTE, font_scale=1.05)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str, save_dir: Path) -> None:
    """Save a figure file and close the matplotlib figure manager."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / name, dpi=PLOT_DPI, bbox_inches="tight")
    logger.info("Saved → %s", save_dir / name)
    plt.close(fig)   # remove from matplotlib's manager so inline backend doesn't auto-display it


def _bdt(x, _):
    """Formatter for BDT currency axis labels, using 'k' for thousands."""
    return f"{x/1000:.0f}k" if abs(x) >= 1000 else f"{x:.0f}"


# ── Comparison table ──────────────────────────────────────────────────────────

def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    """
    Convert the list of per-model metrics dicts into a sorted DataFrame.

    Sort order — four-level fully deterministic tiebreaker
    ───────────────────────────────────────────────────────
    1. CV R² Mean   — descending  (primary: best cross-validated generalisation)
    2. CV R² Std    — ascending   (tiebreaker 1: prefer the more stable model)
    3. RMSE (BDT)   — ascending   (tiebreaker 2: prefer lower absolute error)
    4. Model name   — ascending   (tiebreaker 3: alphabetical — fully deterministic)

    Rationale: all seven models score within 0.0012 CV R² of each other.
    RandomizedSearchCV explores different hyperparameter samples on every run,
    so two models can tie on criteria 1–3 depending on the random seed used
    during training. Without a final tiebreaker the winner would be determined
    by insertion order (which model was trained first) — arbitrary and misleading.
    Alphabetical order as the last resort guarantees the same model wins every
    time all numeric criteria are equal, making the selection fully reproducible.
    """
    df         = pd.DataFrame(results).set_index("Model")
    display_df = df.drop(columns=["Log Transformed"], errors="ignore")

    # Reset index to sort on model name as a column, then restore
    display_df = (
        display_df
        .reset_index()
        .sort_values(
            by=["CV R² Mean", "CV R² Std", "RMSE (BDT)", "Model"],
            ascending=[False, True, True, True],
        )
        .set_index("Model")
    )

    logger.info("Model Comparison Table (CV R² Mean → Std → RMSE → Name):\n%s\n%s\n%s",
                "─" * 90, display_df.to_string(), "─" * 90)

    best_name = display_df.index[0]
    log_col   = "R² (log)" if "R² (log)" in display_df.columns else "R² (BDT)"
    logger.info(
        "Best model: %s | %s=%.4f | R²(BDT)=%.4f | CV Mean=%.4f ±%.4f",
        best_name, log_col, display_df.loc[best_name, log_col],
        display_df.loc[best_name, "R² (BDT)"],
        display_df.loc[best_name, "CV R² Mean"],
        display_df.loc[best_name, "CV R² Std"],
    )
    return display_df


# ── Dashboard 1: CV Comparison (horizontal bar) ───────────────────────────────

def plot_cv_comparison(
    best_models: dict,
    X_train: np.ndarray,
    y_train,
    kf: KFold,
    save_dir: Path = PLOTS_DIR,
    report_dir: Path = REPORTS_DIR,
    precomputed: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Bar chart of CV R² mean ± 95% CI for every model.
    Returns the cv_df DataFrame (also saved to model_cv_comparison.csv).
    Uses precomputed means/stds from the training run when available (fast path);
    otherwise re-runs cross_val_score for each model.
    """
    from .models import build_cv_report

    if precomputed is not None:
        cv_df = precomputed[["CV R² Mean", "CV R² Std"]].rename(
            columns={"CV R² Mean": "Mean", "CV R² Std": "Std"}
        )
        ci_half = 2.776 * (cv_df["Std"] / np.sqrt(CV_FOLDS))
        cv_df["CI_lower"] = (cv_df["Mean"] - ci_half).round(4)
        cv_df["CI_upper"] = (cv_df["Mean"] + ci_half).round(4)
        cv_df["Stable"]   = cv_df["Std"].apply(lambda s: "✓" if s < 0.03 else "⚠")
        logger.info("Cross-Validation Summary (from training run):\n%s", cv_df.to_string())
    else:
        cv_df = build_cv_report(best_models, X_train, y_train, kf)

    save_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(report_dir / "model_cv_comparison.csv")

    sorted_cv = cv_df.sort_values("Mean", ascending=True)
    ci_err    = (sorted_cv["Mean"] - sorted_cv["CI_lower"]).fillna(
        sorted_cv.get("Std", pd.Series(0, index=sorted_cv.index))
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(sorted_cv.index, sorted_cv["Mean"],
                   xerr=ci_err, color="steelblue",
                   capsize=5, edgecolor="white", linewidth=0.8)

    for bar, (name, row) in zip(bars, sorted_cv.iterrows()):
        ax.text(bar.get_width() + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{row['Mean']:.4f} ±{row.get('Std', 0):.4f}",
                va="center", fontsize=9)

    ax.axvline(0.90, color="red",    linestyle="--", linewidth=1.2, label="R²=0.90 (strong)")
    ax.axvline(0.80, color="orange", linestyle=":",  linewidth=1.0, label="R²=0.80 (acceptable)")
    ax.set_title(f"Cross-Validation R² Mean ± 95% CI  ({CV_FOLDS}-Fold)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("CV R² Mean")
    ax.legend(loc="lower right")
    ax.set_xlim(0, min(1.05, sorted_cv["Mean"].max() + 0.10))

    plt.tight_layout()
    _save(fig, "model_cv_comparison.png", save_dir)
    return cv_df


# ── Private draw helpers (shared by dashboard and standalone functions) ───────

def _draw_predicted_vs_actual(ax: plt.Axes, y_true: np.ndarray,
                               y_pred: np.ndarray, r2: float,
                               title_suffix: str = "") -> None:
    ax.scatter(y_true, y_pred, alpha=0.25, s=12, color="steelblue", label="Predictions")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit (y = x)")
    ax.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_title(f"Predicted vs. Actual{title_suffix}", fontsize=12)
    ax.set_xlabel("Actual Fare (BDT)")
    ax.set_ylabel("Predicted Fare (BDT)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_bdt))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_bdt))
    ax.legend(fontsize=9)


def _draw_residuals_scatter(ax: plt.Axes, x: np.ndarray,
                             residuals: np.ndarray, x_label: str) -> None:
    ax.scatter(x, residuals, alpha=0.25, s=12, color="coral")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_title("Residuals vs. Predicted Values", fontsize=12)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Residual")


def _draw_residuals_dist(ax: plt.Axes, residuals_bdt: np.ndarray) -> None:
    sns.histplot(residuals_bdt, bins=60, kde=True, color="coral", ax=ax)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(residuals_bdt.mean(), color="red", linewidth=1,
               label=f"Mean = BDT{residuals_bdt.mean():,.0f}")
    ax.set_title("Residuals Distribution (BDT)", fontsize=12)
    ax.set_xlabel("Actual − Predicted (BDT)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_bdt))
    ax.legend(fontsize=9)


def _draw_feature_importance(ax: plt.Axes, model, feature_names: list[str],
                              model_name: str) -> None:
    # Tree-based models expose feature_importances_ (mean gain / impurity decrease).
    # Linear models expose coef_; we use absolute values so sign doesn't hide magnitude.
    # Priority: feature_importances_ checked first — it's more informative for ensembles.
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names)
        xlabel = "Importance Score (Gain)"
    elif hasattr(model, "coef_"):
        importances = pd.Series(np.abs(model.coef_), index=feature_names)
        xlabel = "|Coefficient|"
    else:
        # Fallback for model types that expose neither attribute (e.g. KNN, SVR).
        ax.text(0.5, 0.5, "Feature importances not available\nfor this model type",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        return

    # Select top 15 by importance, then re-sort ascending so the largest bar is at the top.
    top = importances.sort_values(ascending=False).head(15).sort_values()
    top.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(f"Top 15 Feature Importances — {model_name}", fontsize=12)
    ax.set_xlabel(xlabel)
    # Annotate each bar with its numeric score for quick comparison.
    for i, val in enumerate(top.values):
        ax.text(val + importances.max() * 0.005, i, f"{val:.3f}", va="center", fontsize=8)


# ── Dashboard 2: Best Model Diagnostics (2×2) ─────────────────────────────────

def plot_best_model_dashboard(
    y_test,
    y_pred,
    model,
    model_name: str,
    feature_names: list[str],
    save_dir: Path = PLOTS_DIR,
    log_transformed: bool = False,
) -> plt.Figure:
    """
    2×2 diagnostics panel for the best model:
      [0,0] Predicted vs actual scatter   [0,1] Residuals vs predicted
      [1,0] Residuals distribution        [1,1] Feature importance (top 15)

    When log_transformed=True, scatter and residuals are in original BDT scale
    (inverse-transformed from log space for interpretability).
    """
    y_true_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)

    if log_transformed:
        y_bdt_true = np.expm1(y_true_arr)
        y_bdt_pred = np.expm1(y_pred_arr)
        scale_note = " (inverse-transformed to BDT)"
        res_x      = y_pred_arr
        res_vals   = y_true_arr - y_pred_arr
        x_label    = "Predicted Fare (log scale)"
    else:
        y_bdt_true = y_true_arr
        y_bdt_pred = y_pred_arr
        scale_note = ""
        res_x      = y_bdt_pred
        res_vals   = y_bdt_true - y_bdt_pred
        x_label    = "Predicted Fare (BDT)"

    r2_val   = r2_score(y_bdt_true, y_bdt_pred)
    mae_val  = mean_absolute_error(y_bdt_true, y_bdt_pred)
    rmse_val = mean_squared_error(y_bdt_true, y_bdt_pred) ** 0.5

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Best Model Diagnostics — {model_name}  "
        f"(R²={r2_val:.4f}  MAE=BDT{mae_val:,.0f}  RMSE=BDT{rmse_val:,.0f})",
        fontsize=13, fontweight="bold", y=1.01,
    )

    _draw_predicted_vs_actual(axes[0, 0], y_bdt_true, y_bdt_pred, r2_val, scale_note)
    _draw_residuals_scatter(axes[0, 1], res_x, res_vals, x_label)
    _draw_residuals_dist(axes[1, 0], y_bdt_true - y_bdt_pred)
    _draw_feature_importance(axes[1, 1], model, feature_names, model_name)

    plt.tight_layout()
    _save(fig, "model_diagnostics.png", save_dir)
    return fig


# ── Standalone plot helpers (also used by notebooks) ─────────────────────────

def plot_predicted_vs_actual(y_test, y_pred, model_name: str,
                             save_dir: Path = PLOTS_DIR,
                             log_transformed: bool = False) -> plt.Figure:
    """Scatter of predicted vs. actual fares with y=x diagonal."""
    y_true = np.expm1(np.array(y_test)) if log_transformed else np.array(y_test)
    y_pred = np.expm1(np.array(y_pred)) if log_transformed else np.array(y_pred)
    note   = " (inverse-transformed to BDT)" if log_transformed else ""

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(f"Predicted vs. Actual — {model_name}", fontsize=13, fontweight="bold")
    _draw_predicted_vs_actual(ax, y_true, y_pred, r2_score(y_true, y_pred), note)
    plt.tight_layout()
    _save(fig, f"predicted_vs_actual_{model_name.replace(' ', '_')}.png", save_dir)
    return fig


def plot_residuals(y_test, y_pred, model_name: str,
                   save_dir: Path = PLOTS_DIR,
                   log_transformed: bool = False) -> plt.Figure:
    """Two-panel residual analysis: scatter vs predicted and distribution."""
    y_true = np.array(y_test)
    y_pred = np.array(y_pred)
    res_bdt = (np.expm1(y_true) - np.expm1(y_pred)) if log_transformed else (y_true - y_pred)
    x_label = "Predicted Fare (log scale)" if log_transformed else "Predicted Fare (BDT)"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Residual Analysis — {model_name}", fontsize=14,
                 fontweight="bold", y=1.01)
    _draw_residuals_scatter(axes[0], y_pred, y_true - y_pred, x_label)
    _draw_residuals_dist(axes[1], res_bdt)
    plt.tight_layout()
    _save(fig, f"residuals_{model_name.replace(' ', '_')}.png", save_dir)
    return fig


# plot_cv_fold_scores is the original name used by notebooks
plot_cv_fold_scores = plot_cv_comparison


# ── Learning Curve ────────────────────────────────────────────────────────────

def plot_learning_curve(model, X_train: np.ndarray, y_train,
                        kf: KFold, model_name: str,
                        save_dir: Path = PLOTS_DIR) -> plt.Figure:
    """Training vs. validation R² as training set size grows (bias-variance view)."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        cv=3, scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 6),
        n_jobs=1,
    )

    t_mean, t_std = train_scores.mean(1), train_scores.std(1)
    v_mean, v_std = val_scores.mean(1),   val_scores.std(1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, t_mean, "o-", color="steelblue",  label="Training R²")
    ax.fill_between(train_sizes, t_mean - t_std, t_mean + t_std,
                    alpha=0.15, color="steelblue")
    ax.plot(train_sizes, v_mean, "o-", color="darkorange", label="Validation R² (CV)")
    ax.fill_between(train_sizes, v_mean - v_std, v_mean + v_std,
                    alpha=0.15, color="darkorange")
    ax.set_title(f"Learning Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("R²")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.4)

    plt.tight_layout()
    _save(fig, "model_learning_curve.png", save_dir)
    return fig


# ── Ridge vs Lasso ────────────────────────────────────────────────────────────

def plot_ridge_vs_lasso(ridge_model, lasso_model,
                        feature_names: list[str],
                        save_dir: Path = PLOTS_DIR) -> plt.Figure:
    """Side-by-side comparison of Ridge and Lasso coefficients (top 15 each)."""
    ridge_coefs  = pd.Series(ridge_model.coef_, index=feature_names)
    lasso_coefs  = pd.Series(lasso_model.coef_, index=feature_names)
    n_zero_lasso = int((lasso_coefs == 0).sum())

    logger.info("Ridge: 0 features zeroed | Lasso: %d features zeroed (out of %d)",
                n_zero_lasso, len(lasso_coefs))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Ridge vs. Lasso — Regularization Effect on Coefficients",
                 fontsize=14, fontweight="bold", y=1.01)

    for ax, coefs, title, color in zip(
        axes,
        [ridge_coefs, lasso_coefs],
        [f"Ridge  (α={getattr(ridge_model, 'alpha', '?')})",
         f"Lasso  (α={getattr(lasso_model, 'alpha', '?')})  — "
         f"{n_zero_lasso} features zeroed"],
        ["steelblue", "darkorange"],
    ):
        top = coefs.abs().sort_values(ascending=False).head(15).index
        coefs[top].sort_values().plot(kind="barh", ax=ax, color=color)
        ax.set_title(title, fontsize=12)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Coefficient Value")

    plt.tight_layout()
    _save(fig, "regularization_comparison.png", save_dir)
    return fig


# ── Run all evaluation ────────────────────────────────────────────────────────

def run_evaluation(
    best_models: dict,
    results: list[dict],
    X_test_proc: np.ndarray,
    y_test,
    X_train_proc: np.ndarray,
    y_train,
    feature_names: list[str],
    kf: KFold,
    save_dir: Path = OUTPUTS_DIR,
    log_transformed: bool = False,
) -> dict:
    """
    Run the full evaluation suite (comparison table, CV chart, diagnostics,
    learning curve, Ridge vs Lasso). Returns a summary dict with the comparison
    table, CV scores, best model name/object, and figure objects.

    Outputs are routed to sub-directories:
      plots/   — all .png figures
      reports/ — all .csv files
      models/  — residual_std.json
    """
    logger.info("\n%s\nMODEL EVALUATION\n  Target scale: %s\n%s",
                "=" * 65,
                "log1p(BDT) — plots inverse-transformed to BDT" if log_transformed
                else "raw BDT",
                "=" * 65)

    plots_dir   = save_dir / "plots"
    reports_dir = save_dir / "reports"
    models_dir  = save_dir / "models"
    for d in [plots_dir, reports_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    comparison = build_comparison_table(results)
    cv_scores  = plot_cv_comparison(
        best_models, X_train_proc, y_train, kf,
        save_dir=plots_dir, report_dir=reports_dir,
        precomputed=comparison,
    )

    best_name  = comparison.index[0]
    best_model = best_models[best_name]
    y_pred     = best_model.predict(X_test_proc)

    logger.info("Best model: %s", best_name)

    fig_diag = plot_best_model_dashboard(
        y_test, y_pred, best_model, best_name, feature_names,
        plots_dir, log_transformed=log_transformed,
    )
    fig_lc = plot_learning_curve(
        best_model, X_train_proc, y_train, kf, best_name, plots_dir,
    )

    fig_reg = None
    if "Ridge" in best_models and "Lasso" in best_models:
        fig_reg = plot_ridge_vs_lasso(
            best_models["Ridge"], best_models["Lasso"], feature_names, plots_dir,
        )

    comparison.to_csv(reports_dir / "model_metrics.csv")
    logger.info("Metrics saved → '%s'", reports_dir / "model_metrics.csv")

    # ── Compute and save residual std for prediction intervals ────────────────
    y_true_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    if log_transformed:
        y_bdt_true = np.expm1(y_true_arr)
        y_bdt_pred = np.expm1(y_pred_arr)
    else:
        y_bdt_true = y_true_arr
        y_bdt_pred = y_pred_arr

    residual_std = float((y_bdt_true - y_bdt_pred).std())
    ci_half      = 1.96 * residual_std

    with open(models_dir / "residual_std.json", "w") as f:
        json.dump({"residual_std": residual_std}, f)

    logger.info(
        "Prediction interval (95%%) → ± BDT %s  (residual std = BDT %s)",
        f"{ci_half:,.0f}", f"{residual_std:,.0f}",
    )

    return {
        "comparison_table": comparison,
        "cv_fold_scores":   cv_scores,
        "best_model_name":  best_name,
        "best_model":       best_model,
        "log_transformed":  log_transformed,
        "residual_std":     residual_std,
        "figures": {
            "cv_comparison":         cv_scores,
            "best_model_diagnostics": fig_diag,
            "learning_curve":        fig_lc,
            "ridge_vs_lasso":        fig_reg,
        },
    }
