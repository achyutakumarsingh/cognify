import pandas as pd
import numpy as np
from typing import Dict, Any

class RiskClassifier:
    """
    Categorizes the continuous Risk Score into discrete business levels (Low, Medium, High)
    using statistically justified percentiles derived from the data distribution.
    """
    def __init__(self, config: Dict[str, Any]):
        self.thresholds_cfg = config["decision"]["thresholds"]
        self.low_pct = self.thresholds_cfg["low_risk_max_pct"]
        self.med_pct = self.thresholds_cfg["medium_risk_max_pct"]
        
    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns a Risk_Level based on score percentiles.
        """
        # Calculate absolute score boundaries based on percentiles
        self.low_bound = df["Risk_Score"].quantile(self.low_pct)
        self.med_bound = df["Risk_Score"].quantile(self.med_pct)
        
        conditions = [
            (df["Risk_Score"] <= self.low_bound),
            (df["Risk_Score"] > self.low_bound) & (df["Risk_Score"] <= self.med_bound),
            (df["Risk_Score"] > self.med_bound)
        ]
        choices = ["Low", "Medium", "High"]
        
        df["Risk_Level"] = np.select(conditions, choices, default="High")
        return df
    
    def get_boundaries(self) -> Dict[str, float]:
        """Returns the computed absolute score boundaries for reporting."""
        return {
            "low_threshold": self.low_bound,
            "medium_threshold": self.med_bound
        }
