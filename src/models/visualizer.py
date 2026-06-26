"""
==============================================================================
Forecast Visualizer — Supply Chain Risk Triage
==============================================================================
Responsibility:
    Generate professional diagnostic visualizations to evaluate XGBoost 
    forecasts:
      - Feature Importance (Gain / Weight)
      - Actual vs Predicted Scatter
      - Time Series Overlay (subset of series)
      - Residual Distribution
==============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from src.utils.helpers import setup_logger, get_project_root

logger = setup_logger()

class ForecastVisualizer:
    def __init__(self, out_dir: str = "outputs/plots"):
        self.out_dir = get_project_root() / out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Set matplotlib style
        plt.style.use("seaborn-v0_8-whitegrid")

    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 20):
        """
        Plots feature importance (Gain and Weight).
        """
        logger.info(f"[Visualizer] Plotting top {top_n} feature importances...")
        top_df = importance_df.head(top_n).sort_values("gain", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top_df["feature"], top_df["gain"], color="steelblue")
        ax.set_xlabel("Gain (Average Split Improvement)")
        ax.set_title(f"Top {top_n} Features by Gain")
        
        out_path = self.out_dir / "feature_importance_gain.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved: {out_path}")

    def plot_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray, sample_frac: float = 0.05):
        """
        Scatter plot of actuals vs predicted. Uses a subsample to prevent overplotting.
        """
        logger.info("[Visualizer] Plotting Actual vs Predicted...")
        
        df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
        
        # Subsample if too large
        if len(df) > 10000 and sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.scatterplot(data=df, x="Actual", y="Predicted", alpha=0.1, ax=ax, color="indigo")
        
        # Perfect prediction line
        max_val = max(df["Actual"].max(), df["Predicted"].max())
        ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label="Perfect Prediction")
        
        ax.set_title("Actual vs Predicted Sales")
        ax.legend()
        
        out_path = self.out_dir / "actual_vs_predicted.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    def plot_residual_distribution(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Plots the histogram of residuals (y_true - y_pred).
        """
        logger.info("[Visualizer] Plotting Residual Distribution...")
        residuals = y_true - y_pred
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(residuals, bins=100, kde=True, color="crimson", ax=ax)
        ax.axvline(0, color='k', linestyle='--', lw=2)
        
        ax.set_xlabel("Residual (Actual - Predicted)")
        ax.set_title("Distribution of Forecast Residuals")
        
        out_path = self.out_dir / "residual_distribution.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    def plot_forecast_over_time(self, df: pd.DataFrame, target_col: str = "sales", pred_col: str = "prediction", n_series: int = 3):
        """
        Plots actuals and predictions over time for a sample of series using Plotly.
        Expects df to have: item_id, day_index, target_col, pred_col.
        """
        logger.info(f"[Visualizer] Plotting forecast over time for {n_series} sampled items...")
        
        items = df["item_id"].unique()
        sampled_items = np.random.choice(items, size=min(n_series, len(items)), replace=False)
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Plotly
        for i, item in enumerate(sampled_items):
            item_df = df[df["item_id"] == item].sort_values("day_index")
            c = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=item_df["day_index"], y=item_df[target_col],
                mode='lines', name=f"Actual: {item}",
                line=dict(color=c, width=2)
            ))
            fig.add_trace(go.Scatter(
                x=item_df["day_index"], y=item_df[pred_col],
                mode='lines', name=f"Pred: {item}",
                line=dict(color=c, width=2, dash='dash')
            ))

        fig.update_layout(
            title="Forecast vs Actual over Time (Sampled Items)",
            xaxis_title="Day Index",
            yaxis_title="Sales Volume",
            template="plotly_white",
            hovermode="x unified"
        )
        
        out_path = self.out_dir / "forecast_over_time.html"
        fig.write_html(str(out_path))
        logger.info(f"Saved interactive plot: {out_path}")
