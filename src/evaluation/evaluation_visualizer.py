import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Any

class EvaluationVisualizer:
    """
    Generates publication-quality visualizations for Stage 5.
    """
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set professional style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {"Conformal": "#1f77b4", "Quantile": "#ff7f0e"}

    def plot_calibration_curves(self, conformal_metrics: List[Dict[str, Any]], quantile_metrics: Dict[str, Any], filepath: str):
        """
        Plots Nominal vs Empirical Coverage.
        Conformal gets a full curve. Quantile gets a single point at its native training level.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
        
        # Conformal curve
        conf_x = [m["target_coverage"] for m in conformal_metrics]
        conf_y = [m["empirical_coverage"] for m in conformal_metrics]
        ax.plot(conf_x, conf_y, marker='o', color=self.colors["Conformal"], linewidth=2, label="Split Conformal")
        
        # Quantile point
        q_x = quantile_metrics["target_coverage"]
        q_y = quantile_metrics["empirical_coverage"]
        ax.plot(q_x, q_y, marker='X', color=self.colors["Quantile"], markersize=12, linestyle='None', label="Quantile Regression (Native)")
        
        ax.set_title("Calibration Curve: Nominal vs Empirical Coverage", fontsize=14, fontweight='bold')
        ax.set_xlabel("Target (Nominal) Coverage", fontsize=12)
        ax.set_ylabel("Empirical Coverage", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filepath, dpi=300)
        plt.close()

    def plot_reliability_diagram(self, conformal_metrics: List[Dict[str, Any]], filepath: str):
        """
        Plots Average Interval Width vs Target Coverage for Conformal.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        conf_x = [m["target_coverage"] * 100 for m in conformal_metrics]
        conf_width = [m["avg_interval_width"] for m in conformal_metrics]
        
        ax.bar(conf_x, conf_width, width=4, color=self.colors["Conformal"], alpha=0.8, edgecolor='black')
        
        for i, v in enumerate(conf_width):
            ax.text(conf_x[i], v + 0.05, f"{v:.2f}", ha='center', fontsize=10)
            
        ax.set_title("Interval Width Trade-off (Conformal Prediction)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Confidence Level (%)", fontsize=12)
        ax.set_ylabel("Average Interval Width", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filepath, dpi=300)
        plt.close()

    def plot_volatility_stress_test(self, df_volatility: pd.DataFrame, filepath: str):
        """
        Bar charts for Coverage and Width by Volatility Regime.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot Coverage
        sns.barplot(data=df_volatility, x="Regime", y="Coverage", hue="Method", 
                    palette=[self.colors["Conformal"], self.colors["Quantile"]], ax=axes[0])
        axes[0].axhline(0.90, color='k', linestyle='--', label="Target (90%)")
        axes[0].set_title("Coverage under Volatility Stress", fontsize=14, fontweight='bold')
        axes[0].set_ylim(0.7, 1.0)
        axes[0].legend(loc='lower right')
        
        # Plot Width
        sns.barplot(data=df_volatility, x="Regime", y="Avg_Width", hue="Method", 
                    palette=[self.colors["Conformal"], self.colors["Quantile"]], ax=axes[1])
        axes[1].set_title("Interval Width under Volatility Stress", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filepath, dpi=300)
        plt.close()

    def plot_segment_heatmap(self, df_segments: pd.DataFrame, filepath: str):
        """
        Creates an interactive Plotly heatmap for Store vs Category Coverage Error.
        """
        # Filter for stores and categories (combining them)
        # We need raw test metrics for a 2D heatmap. 
        # Alternatively, a simple bar chart of Worst/Best segments.
        # Let's do a horizontal bar chart of the top 10 worst segments for HTML/PNG.
        
        worst = df_segments.sort_values("Coverage_Error", ascending=False).head(15)
        
        fig = px.bar(
            worst, 
            x="Coverage_Error", 
            y="Segment_Name", 
            color="Segment_Type",
            text="Coverage",
            orientation='h',
            title="Worst Calibrated Segments (Absolute Coverage Error)",
            labels={"Segment_Name": "Segment", "Coverage_Error": "Absolute Error vs 90% Target"}
        )
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        
        fig.write_html(self.output_dir / filepath)
