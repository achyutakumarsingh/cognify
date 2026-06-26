import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.simulation.cost_model import CostModel
from src.simulation.business_impact_analyzer import BusinessImpactAnalyzer

class ScenarioEngine:
    """
    Evaluates business impact across different demand scenarios and sensitivity sweeps.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.volatility_pct = config["simulation"]["scenarios"]["high_volatility_pct"]
        self.analyzer = BusinessImpactAnalyzer()
        self.cost_model = CostModel(config)

    def evaluate_scenarios(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Splits data into Normal, Holiday, and High Volatility and evaluates KPIs.
        """
        # Determine threshold
        vol_threshold = df["rolling_std_28"].quantile(self.volatility_pct)
        
        scenarios = {
            "Holiday_Event": df[df["has_any_event"] == 1],
            "High_Volatility": df[df["rolling_std_28"] >= vol_threshold],
            "Normal_Demand": df[(df["has_any_event"] == 0) & (df["rolling_std_28"] < vol_threshold)]
        }
        
        results = {}
        for name, sdf in scenarios.items():
            if len(sdf) > 0:
                results[name] = self.analyzer.generate_comparison_report(sdf)
                results[name]["Count"] = len(sdf)
                
        return results

    def run_sensitivity_sweep(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Recomputes costs under varying holding and stockout rates to test robustness.
        """
        h_rates = self.config["simulation"]["sensitivity"]["holding_cost_rates"]
        s_rates = self.config["simulation"]["sensitivity"]["stockout_penalty_rates"]
        
        sweep_results = []
        
        for h in h_rates:
            for s in s_rates:
                # Recalculate costs temporarily
                temp_df = df.copy()
                temp_df = self.cost_model.evaluate_financials(temp_df, "Baseline", holding_rate=h, stockout_rate=s)
                temp_df = self.cost_model.evaluate_financials(temp_df, "Proposed", holding_rate=h, stockout_rate=s)
                
                kpi_rep = self.analyzer.generate_comparison_report(temp_df)
                
                sweep_results.append({
                    "Holding_Rate": h,
                    "Stockout_Rate": s,
                    "Baseline_Total_Cost": kpi_rep["Baseline"]["Total_Cost"],
                    "Proposed_Total_Cost": kpi_rep["Proposed"]["Total_Cost"],
                    "Net_Savings": kpi_rep["Impact"]["Net_Savings"]
                })
                
        return sweep_results
