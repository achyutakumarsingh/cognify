"""
==============================================================================
Uncertainty Visualizer — Stage 4 Uncertainty Quantification
==============================================================================
Generates diagnostic visualizations for probabilistic prediction intervals.

Plots Produced
--------------
1. interval_over_time.html
   Interactive Plotly time series showing actuals, point predictions, and
   shaded prediction bands for both quantile and conformal methods,
   for a sample of items.

2. interval_width_histogram.png
   Dual-panel histogram comparing the distribution of interval widths
   for Quantile Regression vs Conformal Prediction.

3. uncertainty_by_item.png
   Horizontal bar chart of mean interval width per item, coloured by
   empirical coverage (green = good, red = undercoverage).

4. uncertainty_by_store.html
   Interactive Plotly grouped bar chart of mean width and coverage
   broken down by store.

5. uncertainty_distribution.png
   KDE of conformity scores (|y − ŷ| on validation set) showing the
   distribution that drives the conformal threshold.

6. quantile_vs_conformal_comparison.png
   Four-panel comparison: Coverage scatter, Width scatter, IS distribution,
   and a summary radar chart.
==============================================================================
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for server-side rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

from src.utils.helpers import setup_logger, get_project_root

logger = setup_logger()

# Global style
plt.style.use("seaborn-v0_8-whitegrid")
FONT_FAMILY = "DejaVu Sans"
DPI = 300


class UncertaintyVisualizer:
    """
    Professional diagnostic visualizer for Stage 4 uncertainty intervals.

    Parameters
    ----------
    config : dict
        Full uncertainty config (loaded from config/uncertainty.yaml).
    out_dir : str
        Relative path (from project root) to the plots output directory.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        out_dir: str = "outputs/plots/",
    ) -> None:
        self.config = config
        self.root = get_project_root()
        self.out_dir = self.root / out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.viz_cfg = config.get("visualization", {})
        self.n_series = self.viz_cfg.get("n_series_to_plot", 4)
        self.dpi = self.viz_cfg.get("dpi", 300)

        # Colour palette
        self.c_actual = self.viz_cfg.get("color_actual", "#2C3E50")
        self.c_q_med = self.viz_cfg.get("color_quantile_median", "#2980B9")
        self.c_q_band = self.viz_cfg.get("color_quantile_band", "#AED6F1")
        self.c_cp_pt = self.viz_cfg.get("color_conformal_point", "#117A65")
        self.c_cp_band = self.viz_cfg.get("color_conformal_band", "#A9DFBF")

    # ------------------------------------------------------------------
    # 1. Interval over time (Plotly interactive)
    # ------------------------------------------------------------------

    def plot_interval_over_time(
        self,
        test_df: pd.DataFrame,
        quantile_intervals: pd.DataFrame,
        conformal_intervals: pd.DataFrame,
        day_index_col: str = "day_index",
        actual_col: str = "actual",
        item_col: str = "item_id",
    ) -> None:
        """
        Interactive time-series plot of prediction intervals.

        Parameters
        ----------
        test_df : pd.DataFrame
            Must have columns: item_id, day_index, actual.
        quantile_intervals : pd.DataFrame
            Must have columns: lower, median, upper (same row order as test_df).
        conformal_intervals : pd.DataFrame
            Must have columns: lower, point, upper (same row order as test_df).
        """
        out_path = self.out_dir / "interval_over_time.html"
        logger.info("[UncertaintyVisualizer] Plotting interval_over_time...")

        all_items = test_df[item_col].unique()
        sampled = np.random.default_rng(42).choice(
            all_items, size=min(self.n_series, len(all_items)), replace=False
        )

        # Attach interval columns to test_df for slicing
        plot_df = test_df.copy().reset_index(drop=True)
        plot_df["q_lower"] = quantile_intervals["lower"].values
        plot_df["q_median"] = quantile_intervals["median"].values
        plot_df["q_upper"] = quantile_intervals["upper"].values
        plot_df["cp_lower"] = conformal_intervals["lower"].values
        plot_df["cp_point"] = conformal_intervals["point"].values
        plot_df["cp_upper"] = conformal_intervals["upper"].values

        fig = go.Figure()
        colors = px.colors.qualitative.Safe

        for i, item in enumerate(sampled):
            c = colors[i % len(colors)]
            mask = plot_df[item_col] == item
            sub = plot_df[mask].sort_values(day_index_col)
            x = sub[day_index_col].values

            # Actuals
            fig.add_trace(go.Scatter(
                x=x, y=sub[actual_col].values,
                mode="lines+markers",
                name=f"{item} — Actual",
                line=dict(color=c, width=2),
                marker=dict(size=4),
                legendgroup=str(item),
            ))

            # Quantile band
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([sub["q_upper"].values, sub["q_lower"].values[::-1]]),
                fill="toself",
                fillcolor=f"rgba(41,128,185,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{item} — Quantile 90% PI",
                legendgroup=str(item) + "_q",
                showlegend=(i == 0),
            ))
            fig.add_trace(go.Scatter(
                x=x, y=sub["q_median"].values,
                mode="lines",
                name=f"{item} — Q Median",
                line=dict(color=self.c_q_med, width=1.5, dash="dot"),
                legendgroup=str(item) + "_q",
                showlegend=(i == 0),
            ))

            # Conformal band
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([sub["cp_upper"].values, sub["cp_lower"].values[::-1]]),
                fill="toself",
                fillcolor=f"rgba(17,122,101,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{item} — Conformal 90% PI",
                legendgroup=str(item) + "_cp",
                showlegend=(i == 0),
            ))
            fig.add_trace(go.Scatter(
                x=x, y=sub["cp_point"].values,
                mode="lines",
                name=f"{item} — CP Point",
                line=dict(color=self.c_cp_pt, width=1.5, dash="dash"),
                legendgroup=str(item) + "_cp",
                showlegend=(i == 0),
            ))

        fig.update_layout(
            title="Prediction Intervals over Time — Quantile vs Conformal (90% PI)",
            xaxis_title="Day Index",
            yaxis_title="Sales Volume",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=550,
        )
        fig.write_html(str(out_path))
        logger.info(f"[UncertaintyVisualizer] Saved: {out_path}")

    # ------------------------------------------------------------------
    # 2. Interval width histogram
    # ------------------------------------------------------------------

    def plot_interval_width_histogram(
        self,
        quantile_widths: np.ndarray,
        conformal_widths: np.ndarray,
    ) -> None:
        """Dual-panel histogram: Quantile vs Conformal interval widths."""
        out_path = self.out_dir / "interval_width_histogram.png"
        logger.info("[UncertaintyVisualizer] Plotting interval width histogram...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
        fig.suptitle(
            "Prediction Interval Width Distribution (90% PI)",
            fontsize=14, fontweight="bold", y=1.01
        )

        # Quantile
        axes[0].hist(
            quantile_widths, bins=60, color=self.c_q_med, alpha=0.75,
            edgecolor="white", linewidth=0.5
        )
        axes[0].axvline(
            np.mean(quantile_widths), color="#1A5276", lw=2, ls="--",
            label=f"Mean = {np.mean(quantile_widths):.2f}"
        )
        axes[0].set_title("Quantile Regression", fontsize=12)
        axes[0].set_xlabel("Interval Width")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()

        # Conformal
        axes[1].hist(
            conformal_widths, bins=60, color=self.c_cp_pt, alpha=0.75,
            edgecolor="white", linewidth=0.5
        )
        axes[1].axvline(
            np.mean(conformal_widths), color="#0E4D3A", lw=2, ls="--",
            label=f"Mean = {np.mean(conformal_widths):.2f}"
        )
        axes[1].set_title("Split Conformal Prediction", fontsize=12)
        axes[1].set_xlabel("Interval Width")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(str(out_path), dpi=self.dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"[UncertaintyVisualizer] Saved: {out_path}")

    # ------------------------------------------------------------------
    # 3. Uncertainty by item
    # ------------------------------------------------------------------

    def plot_uncertainty_by_item(
        self,
        item_eval_q: pd.DataFrame,
        item_eval_cp: pd.DataFrame,
    ) -> None:
        """
        Horizontal bar chart of mean interval width per item,
        coloured by empirical coverage.

        Parameters
        ----------
        item_eval_q, item_eval_cp : pd.DataFrame
            Output of IntervalEvaluator.coverage_by_group().
            Must have columns: group, coverage, mean_width.
        """
        out_path = self.out_dir / "uncertainty_by_item.png"
        logger.info("[UncertaintyVisualizer] Plotting uncertainty_by_item...")

        fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(item_eval_q) * 0.35)))
        fig.suptitle(
            "Per-Item Uncertainty: Mean Interval Width & Empirical Coverage (90% PI)",
            fontsize=13, fontweight="bold"
        )

        def _draw_panel(ax, df: pd.DataFrame, title: str, bar_color: str):
            df_sorted = df.sort_values("mean_width", ascending=True)
            cov = df_sorted["coverage"].values
            # Colour by coverage: green = good, red = undercoverage
            norm = plt.Normalize(vmin=max(0, cov.min() - 0.05), vmax=1.0)
            colors_map = plt.cm.RdYlGn(norm(cov))

            bars = ax.barh(
                df_sorted["group"], df_sorted["mean_width"],
                color=colors_map, edgecolor="white", linewidth=0.5
            )
            ax.set_xlabel("Mean Interval Width")
            ax.set_title(title, fontsize=11)
            sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label="Empirical Coverage")

        _draw_panel(axes[0], item_eval_q, "Quantile Regression", self.c_q_med)
        _draw_panel(axes[1], item_eval_cp, "Split Conformal", self.c_cp_pt)

        plt.tight_layout()
        plt.savefig(str(out_path), dpi=self.dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"[UncertaintyVisualizer] Saved: {out_path}")

    # ------------------------------------------------------------------
    # 4. Uncertainty by store (Plotly)
    # ------------------------------------------------------------------

    def plot_uncertainty_by_store(
        self,
        store_eval_q: pd.DataFrame,
        store_eval_cp: pd.DataFrame,
    ) -> None:
        """
        Grouped bar chart: mean interval width by store for both methods.
        """
        out_path = self.out_dir / "uncertainty_by_store.html"
        logger.info("[UncertaintyVisualizer] Plotting uncertainty_by_store...")

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Quantile Regression — Width",
            x=store_eval_q["group"].astype(str),
            y=store_eval_q["mean_width"],
            marker_color=self.c_q_med,
            opacity=0.85,
        ))
        fig.add_trace(go.Bar(
            name="Split Conformal — Width",
            x=store_eval_cp["group"].astype(str),
            y=store_eval_cp["mean_width"],
            marker_color=self.c_cp_pt,
            opacity=0.85,
        ))
        # Coverage as line traces
        fig.add_trace(go.Scatter(
            name="Quantile — Coverage",
            x=store_eval_q["group"].astype(str),
            y=store_eval_q["coverage"],
            mode="lines+markers",
            line=dict(color="#1A5276", dash="dot"),
            yaxis="y2",
        ))
        fig.add_trace(go.Scatter(
            name="Conformal — Coverage",
            x=store_eval_cp["group"].astype(str),
            y=store_eval_cp["coverage"],
            mode="lines+markers",
            line=dict(color="#0E6655", dash="dot"),
            yaxis="y2",
        ))

        fig.update_layout(
            title="Uncertainty by Store/Item — Interval Width & Coverage (90% PI)",
            xaxis_title="Group",
            yaxis=dict(title="Mean Interval Width"),
            yaxis2=dict(
                title="Empirical Coverage",
                overlaying="y", side="right",
                range=[0, 1.05],
                tickformat=".0%",
            ),
            barmode="group",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=500,
        )
        fig.write_html(str(out_path))
        logger.info(f"[UncertaintyVisualizer] Saved: {out_path}")

    # ------------------------------------------------------------------
    # 5. Conformity score distribution
    # ------------------------------------------------------------------

    def plot_uncertainty_distribution(
        self,
        conformity_scores: np.ndarray,
        q_hat: float,
        alpha: float = 0.10,
    ) -> None:
        """
        KDE of conformity scores |y − ŷ| on the calibration set,
        with the q̂ threshold marked.
        """
        out_path = self.out_dir / "uncertainty_distribution.png"
        logger.info("[UncertaintyVisualizer] Plotting uncertainty_distribution...")

        fig, ax = plt.subplots(figsize=(12, 5))

        sns.histplot(
            conformity_scores, bins=80, stat="density",
            color=self.c_cp_pt, alpha=0.6, ax=ax, label="Conformity Scores"
        )
        # Overlay KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(conformity_scores, bw_method="scott")
        x_range = np.linspace(0, np.percentile(conformity_scores, 99), 500)
        ax.plot(x_range, kde(x_range), color="#0E4D3A", lw=2, label="KDE")

        ax.axvline(
            q_hat, color="#C0392B", lw=2.5, ls="--",
            label=f"q̂ = {q_hat:.3f} (threshold for {1-alpha:.0%} coverage)"
        )

        # Shade covered region
        covered_x = x_range[x_range <= q_hat]
        ax.fill_between(
            covered_x, kde(covered_x), alpha=0.15, color="#27AE60",
            label=f"Covered region ({np.mean(conformity_scores <= q_hat):.1%})"
        )

        ax.set_xlabel("Conformity Score |y − ŷ|", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(
            f"Distribution of Conformity Scores — Calibration Set\n"
            f"(n = {len(conformity_scores):,}  |  α = {alpha}  |  Target coverage = {1-alpha:.0%})",
            fontsize=12
        )
        ax.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(str(out_path), dpi=self.dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"[UncertaintyVisualizer] Saved: {out_path}")

    # ------------------------------------------------------------------
    # 6. Quantile vs Conformal comparison
    # ------------------------------------------------------------------

    def plot_quantile_vs_conformal_comparison(
        self,
        q_result: Dict[str, Any],
        cp_result: Dict[str, Any],
        q_winkler: np.ndarray,
        cp_winkler: np.ndarray,
        q_widths: np.ndarray,
        cp_widths: np.ndarray,
    ) -> None:
        """
        Four-panel comparison figure:
          A) IS (Winkler) Score CDF
          B) Interval width CDF
          C) Coverage & Width summary bar chart
          D) Metric summary table
        """
        out_path = self.out_dir / "quantile_vs_conformal_comparison.png"
        logger.info("[UncertaintyVisualizer] Plotting quantile_vs_conformal_comparison...")

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(
            "Quantile Regression vs Split Conformal Prediction\n"
            "Method Comparison — 90% Prediction Intervals",
            fontsize=15, fontweight="bold"
        )
        gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.35)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # ── A: Winkler CDF ─────────────────────────────────────────────
        for scores, label, col in [
            (q_winkler, "Quantile Regression", self.c_q_med),
            (cp_winkler, "Split Conformal", self.c_cp_pt),
        ]:
            sorted_s = np.sort(scores)
            cdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
            ax1.plot(sorted_s, cdf, label=label, color=col, lw=2)
        ax1.set_xlabel("Interval Score (Winkler)")
        ax1.set_ylabel("Cumulative Probability")
        ax1.set_title("A) Winkler Score CDF")
        ax1.legend(fontsize=9)
        ax1.set_xlim(left=0)

        # ── B: Width CDF ────────────────────────────────────────────────
        for widths, label, col in [
            (q_widths, "Quantile Regression", self.c_q_med),
            (cp_widths, "Split Conformal", self.c_cp_pt),
        ]:
            sorted_w = np.sort(widths)
            cdf_w = np.arange(1, len(sorted_w) + 1) / len(sorted_w)
            ax2.plot(sorted_w, cdf_w, label=label, color=col, lw=2)
        ax2.set_xlabel("Interval Width")
        ax2.set_ylabel("Cumulative Probability")
        ax2.set_title("B) Interval Width CDF")
        ax2.legend(fontsize=9)

        # ── C: Key metrics bar chart ─────────────────────────────────────
        metrics_q = {
            "Coverage": q_result["empirical_coverage"],
            "Mean Width\n(÷10)": q_result["average_interval_width"] / 10,
            "MIS\n(÷10)": q_result["mean_interval_score"] / 10,
            "MSIS\n(÷10)": q_result["msis"] / 10,
        }
        metrics_cp = {
            "Coverage": cp_result["empirical_coverage"],
            "Mean Width\n(÷10)": cp_result["average_interval_width"] / 10,
            "MIS\n(÷10)": cp_result["mean_interval_score"] / 10,
            "MSIS\n(÷10)": cp_result["msis"] / 10,
        }
        x_keys = list(metrics_q.keys())
        x_pos = np.arange(len(x_keys))
        bar_w = 0.35
        ax3.bar(x_pos - bar_w / 2, [metrics_q[k] for k in x_keys],
                bar_w, label="Quantile", color=self.c_q_med, alpha=0.85)
        ax3.bar(x_pos + bar_w / 2, [metrics_cp[k] for k in x_keys],
                bar_w, label="Conformal", color=self.c_cp_pt, alpha=0.85)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(x_keys, fontsize=9)
        ax3.set_title("C) Key Metric Comparison")
        ax3.legend(fontsize=9)
        ax3.axhline(0.90, color="red", ls="--", lw=1.5, alpha=0.7, label="Coverage target")

        # ── D: Summary table ─────────────────────────────────────────────
        ax4.axis("off")
        table_data = [
            ["Metric", "Quantile", "Conformal"],
            ["Coverage Target", "90%", "90%"],
            ["Empirical Coverage", f"{q_result['empirical_coverage']:.4f}", f"{cp_result['empirical_coverage']:.4f}"],
            ["Coverage Error", f"{q_result['coverage_error']:.4f}", f"{cp_result['coverage_error']:.4f}"],
            ["Avg Width", f"{q_result['average_interval_width']:.4f}", f"{cp_result['average_interval_width']:.4f}"],
            ["Mean IS", f"{q_result['mean_interval_score']:.4f}", f"{cp_result['mean_interval_score']:.4f}"],
            ["MSIS", f"{q_result['msis']:.4f}", f"{cp_result['msis']:.4f}"],
            ["Sharpness", f"{q_result['sharpness']:.4f}", f"{cp_result['sharpness']:.4f}"],
            ["Zero-demand Cov", f"{q_result['zero_demand_metrics'].get('zero_coverage', float('nan')):.4f}", f"{cp_result['zero_demand_metrics'].get('zero_coverage', float('nan')):.4f}"],
            ["Guarantee", "None", "≥ 1−α (finite n)"],
        ]
        table = ax4.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        # Style header row
        for j in range(3):
            table[0, j].set_facecolor("#2C3E50")
            table[0, j].set_text_props(color="white", fontweight="bold")
        ax4.set_title("D) Summary Table", fontsize=11, pad=14)

        plt.savefig(str(out_path), dpi=self.dpi, bbox_inches="tight")
        plt.close()
        logger.info(f"[UncertaintyVisualizer] Saved: {out_path}")
