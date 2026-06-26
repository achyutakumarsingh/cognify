import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List

class BusinessVisualizer:
    """
    Generates presentation-ready visualizations for Business Impact (Stage 7).
    """
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        
        self.colors = {"Baseline": "#7f7f7f", "Proposed": "#2ca02c"}

    def plot_cost_breakdown(self, report: Dict[str, Any], filepath: str):
        """Bar chart comparing Total Cost, Holding Cost, and Stockout Cost."""
        metrics = ["Holding_Cost", "Stockout_Cost", "Total_Cost"]
        
        data = []
        for m in metrics:
            data.append({"Metric": m, "Method": "Baseline", "Cost": report["Baseline"][m]})
            data.append({"Metric": m, "Method": "Proposed", "Cost": report["Proposed"][m]})
            
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df, x="Metric", y="Cost", hue="Method", palette=self.colors, ax=ax)
        
        ax.set_title("Financial Cost Breakdown: Baseline vs. Proposed", fontsize=14, fontweight='bold')
        ax.set_ylabel("Total Cost ($)", fontsize=12)
        ax.set_xlabel("")
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filepath, dpi=300)
        plt.close()

    def plot_service_level_improvement(self, report: Dict[str, Any], filepath: str):
        """Simple bar chart comparing service levels."""
        fig, ax = plt.subplots(figsize=(6, 6))
        
        methods = ["Baseline", "Proposed"]
        levels = [report["Baseline"]["Service_Level"] * 100, report["Proposed"]["Service_Level"] * 100]
        
        bars = ax.bar(methods, levels, color=[self.colors["Baseline"], self.colors["Proposed"]], width=0.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center', va='bottom', fontsize=12)
            
        ax.set_title("Service Level Comparison", fontsize=14, fontweight='bold')
        ax.set_ylabel("Service Level (%)", fontsize=12)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filepath, dpi=300)
        plt.close()

    def plot_scenario_performance(self, scenarios: Dict[str, Any], filepath: str):
        """Plots cost savings across the 3 scenarios."""
        scenario_names = []
        savings = []
        
        for name, data in scenarios.items():
            scenario_names.append(name.replace("_", " "))
            savings.append(data["Impact"]["Net_Savings"])
            
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=scenario_names, y=savings, color="#1f77b4", ax=ax)
        
        ax.set_title("Net Savings by Demand Scenario", fontsize=14, fontweight='bold')
        ax.set_ylabel("Net Savings ($)", fontsize=12)
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filepath, dpi=300)
        plt.close()

    def plot_sensitivity_analysis(self, sweep_results: List[Dict[str, Any]], filepath: str):
        """Heatmap of Net Savings across Holding and Stockout rates."""
        df = pd.DataFrame(sweep_results)
        pivot = df.pivot(index="Stockout_Rate", columns="Holding_Rate", values="Net_Savings")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Greens", ax=ax, cbar_kws={'label': 'Net Savings ($)'})
        
        ax.set_title("Sensitivity Analysis: Net Savings Robustness", fontsize=14, fontweight='bold')
        ax.set_xlabel("Holding Cost Rate", fontsize=12)
        ax.set_ylabel("Stockout Penalty Rate", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filepath, dpi=300)
        plt.close()
