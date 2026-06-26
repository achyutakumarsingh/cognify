import pandas as pd
import numpy as np
from typing import Dict, Any

class RecommendationEngine:
    """
    Generates human-readable root causes, operational actions, 
    and proxy business impacts based on risk level and normalized components.
    """
    def __init__(self, config: Dict[str, Any]):
        self.rec_cfg = config["decision"]["recommendations"]
        
    def _determine_root_cause(self, row: pd.Series) -> str:
        if row["Risk_Level"] == "Low":
            return "Demand is stable and predictable."
            
        # Find the dominant component
        comps = {
            "volatile historical demand": row["comp_volat"],
            "large historical forecast errors": row["comp_error"],
            "poor historical calibration for this segment": row["comp_calib"],
            "exceptionally wide prediction intervals": row["comp_width"]
        }
        dominant_factor = max(comps, key=comps.get)
        return f"High uncertainty driven by {dominant_factor}."

    def generate_recommendations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies rules to generate text fields and impact estimates.
        """
        # Root Cause
        df["Root_Cause"] = df.apply(self._determine_root_cause, axis=1)
        
        # Actions & Interventions
        df["Recommended_Action"] = df["Risk_Level"].apply(lambda x: self.rec_cfg[x]["action"])
        df["Intervention"] = df["Risk_Level"].apply(lambda x: self.rec_cfg[x]["intervention"])
        
        # Business Impact Preview
        # The recommended safety stock buffer (proxy) is dynamically derived
        def _compute_buffer(row):
            pct = self.rec_cfg[row["Risk_Level"]]["safety_stock_buffer_percentile"]
            # Interpolate between point (50th approx) and upper bound (95th approx)
            # This is just a proxy for Stage 6 before real inventory simulation
            buffer = row["point"] + (row["upper_sym"] - row["point"]) * pct
            # Ensure buffer >= point
            return max(row["point"], buffer)

        df["Recommended_Stock_Level"] = df.apply(_compute_buffer, axis=1).round(2)
        df["Inventory_Increase"] = (df["Recommended_Stock_Level"] - df["point"]).round(2)
        
        # Confidence Level (inverse of normalized risk score, scaled to 0-100%)
        # Just a presentation metric for the dashboard
        df["Confidence_Level"] = (100.0 - df["Risk_Score"]).round(1)
        
        return df
