import numpy as np
import pandas as pd
from typing import Dict, Any

class RiskScorer:
    """
    Computes a composite Risk Score (0-100) based on uncertainty intervals, 
    forecast errors, volatility features, and calibration penalties.
    """
    def __init__(self, config: Dict[str, Any]):
        self.weights = config["decision"]["score_weights"]
        
    def _normalize(self, series: pd.Series) -> pd.Series:
        """Min-Max normalization to 0-1 range. Handles edge cases where min == max."""
        c_min = series.min()
        c_max = series.max()
        if np.isclose(c_min, c_max):
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - c_min) / (c_max - c_min)

    def compute_risk_scores(self, df_preds: pd.DataFrame, df_features: pd.DataFrame, df_segments: pd.DataFrame) -> pd.DataFrame:
        """
        Merge predictions, features, and segment calibration, then compute the score.
        """
        # We assume df_preds and df_features have perfectly aligned indices (from the same test split)
        # Or we can just copy necessary columns if they are strictly ordered.
        # It's safest to just assign the features we need if indices match.
        df = df_preds.copy()
        df["rolling_std_28"] = df_features["rolling_std_28"].values
        df["is_zero_sales"] = df_features["is_zero_sales"].values
        
        # Calculate raw components
        df["raw_error"] = (df["actual"] - df["point"]).abs()
        df["raw_width"] = df["width_sym"]
        df["raw_volatility"] = df["rolling_std_28"]
        
        # Merge calibration penalty
        # Extract cat_id for merging
        df["cat_id"] = df["item_id"].astype(str).str.split("_").str[0]
        
        # Filter segments to just Conformal (or whatever method we chose) and cat_id
        # We will use category-level calibration error as the penalty
        df_cat = df_segments[(df_segments["Segment_Type"] == "cat_id") & (df_segments["Method"] == "Conformal")]
        df_cat_map = df_cat.set_index("Segment_Name")["Coverage_Error"].to_dict()
        
        df["raw_calib_penalty"] = df["cat_id"].map(df_cat_map).fillna(0.0)
        
        # Normalize components
        norm_width = self._normalize(df["raw_width"])
        norm_error = self._normalize(df["raw_error"])
        norm_volat = self._normalize(df["raw_volatility"])
        norm_calib = self._normalize(df["raw_calib_penalty"])
        
        # Compute composite score
        score = (
            self.weights["interval_width"] * norm_width +
            self.weights["forecast_error"] * norm_error +
            self.weights["historical_volatility"] * norm_volat +
            self.weights["calibration_penalty"] * norm_calib
        ) * 100.0  # Scale to 0-100
        
        df["Risk_Score"] = score.round(2)
        
        # Keep components for root cause analysis
        df["comp_width"] = norm_width
        df["comp_error"] = norm_error
        df["comp_volat"] = norm_volat
        df["comp_calib"] = norm_calib
        
        return df
