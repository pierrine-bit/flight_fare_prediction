"""
eda.py — Step 4: Exploratory Data Analysis.

Four dashboard figures keep outputs/ tidy:
  eda_fare_overview.png   — fare distributions + temporal patterns (2x2)
  eda_airline_analysis.png — airline fare comparisons (1x2)
  eda_route_analysis.png  — route popularity and cost (1x2)
  eda_correlations.png    — numerical feature correlation heatmap
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from .config import OUTPUTS_DIR, PLOTS_DIR, PLOT_DPI, PLOT_STYLE, PLOT_PALETTE, NUMERICAL_FEATURES

logger = logging.getLogger(__name__)

sns.set_theme(style=PLOT_STYLE, palette=PLOT_PALETTE, font_scale=1.05)

_MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str, save_dir: Path) -> None:
    """Save a figure to disk and close it to avoid duplicate inline display."""
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / name, dpi=PLOT_DPI, bbox_inches="tight")
    logger.info("Saved → %s", save_dir / name)
    plt.close(fig)   # prevent inline backend from auto-displaying a duplicate


def _bdt(x, _):
    """Compact BDT tick formatter: 150000 → '150k'."""
    return f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"


# ── Dashboard 1: Fare Overview (2×2) ─────────────────────────────────────────

def plot_fare_overview_dashboard(df: pd.DataFrame,
                                 save_dir: Path = PLOTS_DIR) -> plt.Figure:
    """
    2×2 panel: raw fare histogram, log-transformed histogram,
    monthly trend line, and fare-by-season boxplot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Flight Fare — Distribution & Temporal Patterns",
                 fontsize=15, fontweight="bold", y=1.01)

    # ── [0,0] Raw fare histogram ──────────────────────────────────────────────
    ax = axes[0, 0]
    sns.histplot(df["Total_Fare"], bins=60, kde=True, color="steelblue", ax=ax)
    ax.set_title("Total Fare Distribution (BDT)", fontsize=12)
    ax.set_xlabel("Total Fare (BDT)")
    ax.set_ylabel("Count")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_bdt))

    # ── [0,1] Log-transformed histogram ──────────────────────────────────────
    ax = axes[0, 1]
    sns.histplot(np.log1p(df["Total_Fare"]), bins=60, kde=True,
                 color="darkorange", ax=ax)
    ax.set_title("log(1 + Total Fare) — After Log Transform", fontsize=12)
    ax.set_xlabel("log(1 + Total Fare)")
    ax.set_ylabel("Count")
    skew_raw = df["Total_Fare"].skew()
    skew_log = np.log1p(df["Total_Fare"]).skew()
    ax.text(0.97, 0.93,
            f"Raw skew: {skew_raw:.2f}\nLog skew: {skew_log:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # ── [1,0] Monthly trend ───────────────────────────────────────────────────
    ax = axes[1, 0]
    monthly = df.groupby("Month")["Total_Fare"].agg(["mean", "std"]).reset_index()
    ax.plot(monthly["Month"], monthly["mean"], "o-",
            linewidth=2.5, color="teal", markersize=7, label="Mean fare")
    ax.fill_between(monthly["Month"],
                    monthly["mean"] - monthly["std"],
                    monthly["mean"] + monthly["std"],
                    alpha=0.15, color="teal", label="±1 std")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(_MONTH_LABELS, fontsize=9)
    ax.set_title("Average Monthly Fare Trend", fontsize=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Total Fare (BDT)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_bdt))
    ax.grid(axis="y", alpha=0.35)
    ax.legend(fontsize=9)

    for peak_month, label in [(4, "Eid-ul-Fitr"), (7, "Eid-ul-Adha")]:
        row = monthly[monthly["Month"] == peak_month]
        if not row.empty:
            val = row["mean"].values[0]
            ax.annotate(label, xy=(peak_month, val),
                        xytext=(peak_month + 0.5, val + 350),
                        arrowprops=dict(arrowstyle="->", color="red"),
                        fontsize=8, color="red")

    # ── [1,1] Fare by season (boxplot) ────────────────────────────────────────
    ax = axes[1, 1]
    season_order = (df.groupby("Season")["Total_Fare"]
                    .median().sort_values(ascending=False).index.tolist())
    palette = dict(zip(season_order,
                       ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"][:len(season_order)]))
    sns.boxplot(data=df, x="Season", y="Total_Fare",
                order=season_order, hue="Season", palette=palette,
                legend=False, fliersize=2.5, ax=ax)
    ax.set_title("Total Fare by Travel Season", fontsize=12)
    ax.set_xlabel("Season")
    ax.set_ylabel("Total Fare (BDT)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_bdt))

    for i, season in enumerate(season_order):
        median = df[df["Season"] == season]["Total_Fare"].median()
        ax.text(i, median + 100, f"{median:,.0f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    _save(fig, "eda_fare_overview.png", save_dir)
    return fig


# ── Dashboard 2: Airline Analysis (1×2) ──────────────────────────────────────

def plot_airline_dashboard(df: pd.DataFrame,
                           save_dir: Path = PLOTS_DIR) -> plt.Figure:
    """
    1×2 panel: mean fare per airline (horizontal bar, annotated) and
    fare spread per airline (boxplot, sorted by median, labels rotated 90°).
    """
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.suptitle("Airline Fare Analysis", fontsize=15, fontweight="bold", y=1.01)

    summary = (
        df.groupby("Airline")["Total_Fare"]
        .agg(mean_fare="mean", count="count")
        .sort_values("mean_fare", ascending=True)   # ascending for horizontal bar
        .reset_index()
    )

    # ── [0] Average fare — horizontal bar (no crowding on y-axis) ────────────
    ax = axes[0]
    sns.barplot(data=summary, y="Airline", x="mean_fare",
                hue="Airline", palette="Blues_d", legend=False, ax=ax, orient="h")
    ax.set_title("Average Total Fare by Airline", fontsize=13)
    ax.set_ylabel("Airline")
    ax.set_xlabel("Average Fare (BDT)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_bdt))

    for bar, (_, row) in zip(ax.patches, summary.iterrows()):
        ax.text(bar.get_width() + summary["mean_fare"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"BDT {row['mean_fare']:,.0f}  (n={int(row['count']):,})",
                va="center", fontsize=8)

    # ── [1] Fare spread boxplot — rotate labels 90° to prevent overlap ────────
    ax = axes[1]
    order = (df.groupby("Airline")["Total_Fare"]
             .median().sort_values(ascending=False).index)
    sns.boxplot(data=df, x="Airline", y="Total_Fare", order=order,
                hue="Airline", palette="Set2", legend=False,
                fliersize=2.5, ax=ax)
    ax.set_title("Fare Spread by Airline (Median-Sorted)", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Total Fare (BDT)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_bdt))
    ax.tick_params(axis="x", rotation=90, labelsize=8)

    plt.tight_layout()
    _save(fig, "eda_airline_analysis.png", save_dir)
    return fig


# ── Dashboard 3: Route Intelligence (1×2) ────────────────────────────────────

def plot_route_dashboard(df: pd.DataFrame,
                         save_dir: Path = PLOTS_DIR) -> plt.Figure:
    """
    1×2 panel: top-10 routes by flight count and top-5 routes by average fare.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Route Intelligence", fontsize=15, fontweight="bold", y=1.01)

    # ── [0] Popular routes (horizontal bar) ──────────────────────────────────
    ax = axes[0]
    route_counts = (
        df["Route"].value_counts().head(10)
        .rename_axis("Route").reset_index(name="Count")
    )
    sns.barplot(data=route_counts, y="Route", x="Count",
                hue="Route", palette="viridis", legend=False, ax=ax, orient="h")
    ax.set_title("Top 10 Most Popular Routes (Flight Count)", fontsize=13)
    ax.set_xlabel("Number of Flights")
    ax.set_ylabel("Route")

    for bar in ax.patches:
        ax.text(bar.get_width() + 8,
                bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width()):,}", va="center", fontsize=9)

    # ── [1] Expensive routes (bar) ────────────────────────────────────────────
    ax = axes[1]
    top_routes = (
        df.groupby("Route")["Total_Fare"]
        .mean().sort_values(ascending=False).head(5)
        .reset_index().rename(columns={"Total_Fare": "Avg_Fare"})
    )
    sns.barplot(data=top_routes, x="Route", y="Avg_Fare",
                hue="Route", palette="Reds_d", legend=False, ax=ax)
    ax.set_title("Top 5 Most Expensive Routes (Average Fare)", fontsize=13)
    ax.set_xlabel("Route")
    ax.set_ylabel("Average Total Fare (BDT)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_bdt))

    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 60,
                f"BDT {bar.get_height():,.0f}",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    _save(fig, "eda_route_analysis.png", save_dir)
    return fig


# ── Dashboard 4: Correlation Heatmap (full-width) ────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame,
                             save_dir: Path = PLOTS_DIR) -> plt.Figure:
    """Lower-triangle Pearson heatmap for numerical features + target."""
    cols = NUMERICAL_FEATURES + ["Total_Fare"]
    corr = df[[c for c in cols if c in df.columns]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, linewidths=0.5,
                annot_kws={"size": 10}, ax=ax)
    ax.set_title("Correlation Heatmap — Numerical Features", fontsize=14,
                 fontweight="bold")

    plt.tight_layout()
    _save(fig, "eda_correlations.png", save_dir)
    return fig


# ── Run all EDA dashboards ────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame,
            save_dir: Path = PLOTS_DIR) -> dict[str, plt.Figure]:
    """Run all four EDA dashboards and return a name → Figure mapping."""
    logger.info("\n%s\nEXPLORATORY DATA ANALYSIS\n%s", "=" * 60, "=" * 60)

    figs: dict[str, plt.Figure] = {
        "fare_overview":    plot_fare_overview_dashboard(df, save_dir),
        "airline_analysis": plot_airline_dashboard(df, save_dir),
        "route_analysis":   plot_route_dashboard(df, save_dir),
        "correlations":     plot_correlation_heatmap(df, save_dir),
    }

    logger.info("EDA complete — %d dashboard figures saved to '%s'", len(figs), save_dir)
    return figs
