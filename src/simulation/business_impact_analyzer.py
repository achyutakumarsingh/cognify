import pandas as pd
from typing import Dict, Any

class BusinessImpactAnalyzer:
    """
    Computes aggregated KPIs and financial summaries comparing the Baseline and Proposed systems.
    """
    def __init__(self):
        pass

    def compute_kpis(self, df: pd.DataFrame, prefix: str) -> Dict[str, float]:
        """
        Computes key operational and financial metrics for a policy (Baseline or Proposed).
        """
        total_demand = df["actual"].sum()
        total_sold = df[f"{prefix}_Sold_Units"].sum()
        
        # Fill Rate: % of unit demand fulfilled
        fill_rate = total_sold / total_demand if total_demand > 0 else 1.0
        
        # Service Level (Type 1): % of instances where demand was fully met without stockout
        service_level = (df[f"{prefix}_Stockout_Units"] == 0).mean()
        
        # Costs
        total_holding = df[f"{prefix}_Holding_Cost"].sum()
        total_stockout = df[f"{prefix}_Stockout_Cost"].sum()
        total_cost = df[f"{prefix}_Total_Cost"].sum()
        
        return {
            "Total_Demand": float(total_demand),
            "Total_Sold": float(total_sold),
            "Stockout_Units": float(df[f"{prefix}_Stockout_Units"].sum()),
            "Overstock_Units": float(df[f"{prefix}_Overstock_Units"].sum()),
            "Fill_Rate": float(fill_rate),
            "Service_Level": float(service_level),
            "Holding_Cost": float(total_holding),
            "Stockout_Cost": float(total_stockout),
            "Total_Cost": float(total_cost)
        }

    def generate_comparison_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates the full comparison between Baseline and Proposed.
        """
        baseline_kpis = self.compute_kpis(df, "Baseline")
        proposed_kpis = self.compute_kpis(df, "Proposed")
        
        savings = baseline_kpis["Total_Cost"] - proposed_kpis["Total_Cost"]
        cost_reduction_pct = savings / baseline_kpis["Total_Cost"] if baseline_kpis["Total_Cost"] > 0 else 0
        
        stockout_reduction = baseline_kpis["Stockout_Units"] - proposed_kpis["Stockout_Units"]
        
        # ROI: Savings / Additional Holding Cost (if any)
        extra_holding = proposed_kpis["Holding_Cost"] - baseline_kpis["Holding_Cost"]
        roi = (savings / extra_holding) if extra_holding > 0 else float("inf")
        
        return {
            "Baseline": baseline_kpis,
            "Proposed": proposed_kpis,
            "Impact": {
                "Net_Savings": float(savings),
                "Cost_Reduction_Pct": float(cost_reduction_pct),
                "Stockout_Reduction_Units": float(stockout_reduction),
                "Extra_Holding_Cost": float(extra_holding),
                "ROI": float(roi)
            }
        }
