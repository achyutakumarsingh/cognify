import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path
from typing import Dict, Any

class DecisionVisualizer:
    """
    Generates presentation-ready visualizations for the Stage 6 Decision Engine.
    """
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        
        self.risk_colors = {
            "Low": "#2ca02c",    # Green
            "Medium": "#ff7f0e", # Orange
            "High": "#d62728"    # Red
        }

    def plot_risk_distribution(self, df: pd.DataFrame, boundaries: Dict[str, float], filepath: str):
        """Histogram of the composite risk score with classification boundaries."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.histplot(data=df, x="Risk_Score", bins=50, kde=True, color="#1f77b4", ax=ax)
        
        ax.axvline(boundaries["low_threshold"], color=self.risk_colors["Low"], linestyle="--", linewidth=2, label=f'Low/Med Bound ({boundaries["low_threshold"]:.1f})')
        ax.axvline(boundaries["medium_threshold"], color=self.risk_colors["High"], linestyle="--", linewidth=2, label=f'Med/High Bound ({boundaries["medium_threshold"]:.1f})')
        
        ax.set_title("Distribution of Composite Risk Score", fontsize=14, fontweight='bold')
        ax.set_xlabel("Risk Score (0 - 100)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filepath, dpi=300)
        plt.close()

    def plot_risk_level_distribution(self, df: pd.DataFrame, filepath: str):
        """Bar chart showing the frequency of each risk level."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        counts = df["Risk_Level"].value_counts().reindex(["Low", "Medium", "High"])
        
        sns.barplot(x=counts.index, y=counts.values, palette=self.risk_colors, ax=ax)
        
        for i, v in enumerate(counts.values):
            ax.text(i, v + (v * 0.02), str(v), ha='center', fontsize=11)
            
        ax.set_title("Risk Level Distribution", fontsize=14, fontweight='bold')
        ax.set_ylabel("Number of Predictions", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filepath, dpi=300)
        plt.close()

    def plot_top_high_risk_items(self, df: pd.DataFrame, filepath: str):
        """Horizontal bar chart of the top 10 items with the highest average risk score."""
        top_items = df.groupby("item_id")["Risk_Score"].mean().sort_values(ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_items.values, y=top_items.index, color=self.risk_colors["High"], ax=ax)
        
        ax.set_title("Top 10 High-Risk Items", fontsize=14, fontweight='bold')
        ax.set_xlabel("Average Risk Score", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filepath, dpi=300)
        plt.close()

    def plot_risk_heatmap(self, df: pd.DataFrame, filepath: str):
        """Interactive heatmap showing the percentage of High-Risk predictions by Store vs Category."""
        # Calculate % of High Risk by store and category
        df["is_high_risk"] = (df["Risk_Level"] == "High").astype(int)
        
        heatmap_data = df.groupby(["store_id", "cat_id"])["is_high_risk"].mean().reset_index()
        
        fig = px.density_heatmap(
            heatmap_data, 
            x="store_id", 
            y="cat_id", 
            z="is_high_risk", 
            histfunc="avg",
            color_continuous_scale="Reds",
            title="Percentage of High-Risk Predictions by Store and Category",
            labels={"store_id": "Store", "cat_id": "Category", "is_high_risk": "% High Risk"}
        )
        
        fig.update_layout(coloraxis_colorbar=dict(tickformat=".1%"))
        fig.write_html(self.output_dir / filepath)
