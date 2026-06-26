import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# We can reuse the metric calculations from Stage 4
from src.uncertainty.interval_evaluator import IntervalEvaluator

class CalibrationEvaluator:
    """
    Core Evaluation Engine for Stage 5.
    Calculates calibration metrics, dynamic intervals, and segment/volatility analyses.
    """

    def __init__(self, config: Dict[str, Any], meta: Dict[str, Any], df_test: pd.DataFrame):
        self.config = config
        self.meta = meta
        self.df_test = df_test  # Contains features (rolling_std_28, item_id, store_id)
        
        # We need the training y array for MSIS. In Stage 5, we don't have X_train loaded.
        # But wait, MSIS requires the insample mean absolute diff. 
        # For simplicity, if we don't have train_y, we can compute MSIS using the mean of y_test, 
        # or we just skip MSIS if train_y isn't available. Let's see if we can just compute Winkler/Sharpness/Coverage.
        # Since we just want pure evaluation, we will compute metrics directly here or via IntervalEvaluator.

    def generate_conformal_intervals(self, alpha: float, point_preds: np.ndarray) -> pd.DataFrame:
        """
        Dynamically generate Conformal intervals for any alpha using Stage 4 calibration scores.
        q_hat = Quantile(scores, ceil((n+1)(1-alpha)) / n)
        """
        cal_meta = self.meta.get("conformal_prediction", {})
        scores = cal_meta.get("conformity_scores", [])
        n = len(scores)
        
        if not scores:
            raise ValueError("No conformity scores found in interval_metadata.")
        
        # Finite-sample correction quantile
        target_quantile = math.ceil((n + 1) * (1 - alpha)) / n
        target_quantile = min(target_quantile, 1.0)
        
        q_hat = np.quantile(scores, target_quantile, method="higher")
        
        lower = point_preds - q_hat
        upper = point_preds + q_hat
        
        # Zero-bound demand (no negative sales)
        lower = np.maximum(lower, 0.0)
        upper = np.maximum(upper, 0.0)
        
        return pd.DataFrame({
            "point": point_preds,
            "lower": lower,
            "upper": upper,
            "width": upper - lower
        })

    def compute_metrics(
        self, 
        y_true: np.ndarray, 
        lower: np.ndarray, 
        upper: np.ndarray, 
        alpha: float
    ) -> Dict[str, Any]:
        """
        Compute pure evaluation metrics: Coverage, AIW, Sharpness, Winkler Score.
        """
        width = upper - lower
        aiw = float(np.mean(width))
        
        # Coverage
        covered = (y_true >= lower) & (y_true <= upper)
        empirical_coverage = float(np.mean(covered))
        coverage_error = abs(empirical_coverage - (1 - alpha))
        
        # Zero-demand coverage (if demand is exactly 0)
        zero_mask = (y_true == 0)
        zero_cov = float(np.mean(covered[zero_mask])) if zero_mask.sum() > 0 else 0.0
        
        # Winkler Score
        # IS = width + (2/alpha)*(lower - y)*I(y < lower) + (2/alpha)*(y - upper)*I(y > upper)
        penalty_lower = (2.0 / alpha) * (lower - y_true) * (y_true < lower)
        penalty_upper = (2.0 / alpha) * (y_true - upper) * (y_true > upper)
        winkler = width + penalty_lower + penalty_upper
        mean_winkler = float(np.mean(winkler))
        
        # Sharpness (Variance of interval widths, or just mean width. Commonly AIW is sharpness)
        # Some define sharpness as the mean width itself. We'll report both AIW and std(width).
        sharpness = float(np.std(width))
        
        return {
            "target_coverage": 1 - alpha,
            "empirical_coverage": empirical_coverage,
            "coverage_error": coverage_error,
            "avg_interval_width": aiw,
            "sharpness_std": sharpness,
            "winkler_score": mean_winkler,
            "zero_demand_coverage": zero_cov
        }

    def evaluate_volatility(self, df_metrics: pd.DataFrame, volatility_feature: str, threshold: float) -> pd.DataFrame:
        """
        Split observations by a volatility feature (e.g. rolling_std_28 > 75th percentile).
        df_metrics must contain: actual, lower, upper, and the volatility_feature.
        """
        # Determine High vs Normal volatility
        cutoff = df_metrics[volatility_feature].quantile(threshold)
        
        df_metrics["volatility_regime"] = np.where(
            df_metrics[volatility_feature] >= cutoff,
            "High Volatility",
            "Normal"
        )
        
        results = []
        for (method, regime), group in df_metrics.groupby(["Method", "volatility_regime"]):
            width = group["upper"] - group["lower"]
            covered = (group["actual"] >= group["lower"]) & (group["actual"] <= group["upper"])
            
            results.append({
                "Method": method,
                "Regime": regime,
                "Count": len(group),
                "Coverage": float(np.mean(covered)),
                "Avg_Width": float(np.mean(width))
            })
            
        return pd.DataFrame(results)

    def evaluate_segments(self, df_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate calibration by store_id, cat_id, dept_id, state_id.
        Extracts hierarchies from item_id and store_id.
        """
        df = df_metrics.copy()
        
        # Extract hierarchies if they don't exist
        if "state_id" not in df.columns:
            df["state_id"] = df["store_id"].astype(str).str.split("_").str[0]
        if "cat_id" not in df.columns:
            df["cat_id"] = df["item_id"].astype(str).str.split("_").str[0]
        if "dept_id" not in df.columns:
            # HOBBIES_1_001 -> HOBBIES_1
            df["dept_id"] = df["item_id"].astype(str).apply(lambda x: "_".join(x.split("_")[:2]) if isinstance(x, str) else x)
            
        segments = ["store_id", "state_id", "cat_id", "dept_id"]
        
        all_results = []
        for seg in segments:
            for (method, seg_val), group in df.groupby(["Method", seg]):
                width = group["upper"] - group["lower"]
                covered = (group["actual"] >= group["lower"]) & (group["actual"] <= group["upper"])
                
                target = 0.90
                cov = float(np.mean(covered))
                
                all_results.append({
                    "Method": method,
                    "Segment_Type": seg,
                    "Segment_Name": seg_val,
                    "Count": len(group),
                    "Coverage": cov,
                    "Coverage_Error": abs(cov - target),
                    "Avg_Width": float(np.mean(width))
                })
                
        return pd.DataFrame(all_results)
