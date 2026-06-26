"""
==============================================================================
Forecast Evaluator — Supply Chain Risk Triage
==============================================================================
Responsibility:
    Compute standard forecasting metrics (RMSE, MAE, MAPE, WAPE) as well as
    the official M5 competition metric, RMSSE (Root Mean Squared Scaled Error).
==============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.utils.helpers import setup_logger

logger = setup_logger()

class ForecastEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    @staticmethod
    def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Weighted Absolute Percentage Error
        # sum(|y - y_hat|) / sum(y)
        sum_y = np.sum(y_true)
        if sum_y == 0:
            return 0.0
        return (np.sum(np.abs(y_true - y_pred)) / sum_y) * 100

    @staticmethod
    def _rmsse(y_true: np.ndarray, y_pred: np.ndarray, train_y: np.ndarray, train_item_ids: np.ndarray, item_ids: np.ndarray) -> float:
        """
        Calculates the Root Mean Squared Scaled Error (RMSSE).
        
        RMSSE scales the MSE of the forecast by the MSE of a naive one-step-ahead
        forecast on the training data. This makes it scale-independent and safe
        for zero-inflated series.
        
        Parameters
        ----------
        y_true : np.ndarray (N,)
        y_pred : np.ndarray (N,)
        train_y : np.ndarray (M,) - The historical training sales
        train_item_ids : np.ndarray (M,) - The item IDs for the training sales
        item_ids : np.ndarray (N,) - The item IDs for the evaluation set
        
        Returns
        -------
        float : Average RMSSE across items
        """
        # Note: True M5 WRMSSE involves weighting by 28-day revenue.
        # Here we calculate average RMSSE across series for a robust proxy.
        
        # 1. Compute naive forecast error per item in train
        # Group train_y by train_item_ids and calculate difference
        df_train = pd.DataFrame({"y": train_y, "id": train_item_ids})
        
        # Naive forecast is shift(1)
        df_train["y_shift"] = df_train.groupby("id")["y"].shift(1)
        df_train = df_train.dropna()
        
        # Denominator (scale) per item
        scale_per_item = df_train.groupby("id").apply(
            lambda g: np.mean((g["y"] - g["y_shift"])**2)
        ).to_dict()

        # 2. Compute forecast error per item in eval
        df_eval = pd.DataFrame({"y": y_true, "y_pred": y_pred, "id": item_ids})
        mse_per_item = df_eval.groupby("id").apply(
            lambda g: np.mean((g["y"] - g["y_pred"])**2)
        ).to_dict()

        # 3. Calculate RMSSE per item
        rmsse_vals = []
        for item, mse in mse_per_item.items():
            scale = scale_per_item.get(item, 1.0)
            if scale == 0:
                scale = 1.0 # Fallback if perfectly constant history
            rmsse_vals.append(np.sqrt(mse / scale))

        return np.mean(rmsse_vals)

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series, 
                 train_y: Optional[pd.Series] = None, 
                 train_item_ids: Optional[pd.Series] = None, 
                 eval_item_ids: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Evaluate predictions using multiple metrics.
        """
        y_true_arr = y_true.values
        y_pred_arr = y_pred.values

        # Ensure predictions are >= 0
        y_pred_arr = np.maximum(y_pred_arr, 0)

        metrics = {
            "RMSE": float(self._rmse(y_true_arr, y_pred_arr)),
            "MAE": float(self._mae(y_true_arr, y_pred_arr)),
            "MAPE": float(self._mape(y_true_arr, y_pred_arr)),
            "WAPE": float(self._wape(y_true_arr, y_pred_arr)),
        }
        
        # Calculate RMSSE if we have historical data
        if train_y is not None and train_item_ids is not None and eval_item_ids is not None:
            try:
                metrics["RMSSE"] = float(self._rmsse(
                    y_true_arr, y_pred_arr, 
                    train_y.values, train_item_ids.values, eval_item_ids.values
                ))
            except Exception as e:
                logger.warning(f"[Evaluator] Failed to compute RMSSE: {e}")
                metrics["RMSSE"] = float('nan')

        # Log metrics
        logger.info("━" * 40)
        logger.info("Forecast Evaluation Metrics")
        logger.info("━" * 40)
        for k, v in metrics.items():
            logger.info(f"{k:>6}: {v:.4f}")
        logger.info("━" * 40)

        return metrics
