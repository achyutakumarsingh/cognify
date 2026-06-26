"""
==============================================================================
XGBoost Trainer — Supply Chain Risk Triage
==============================================================================
Responsibility:
    Handles configuration, training, and persistence of the XGBoost Regressor.
    Designed to easily accept custom parameters from Optuna and seamlessly 
    output point predictions. Later stages can extend this for Quantiles.
==============================================================================
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import pandas as pd
import numpy as np
import xgboost as xgb

from src.utils.helpers import setup_logger, get_project_root

logger = setup_logger()

class XGBoostTrainer:
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the trainer with static configuration.
        """
        self.config = config
        self.model = None
        self.feature_names = None
        self.best_iteration = None

    def _prepare_dmatrix(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> xgb.DMatrix:
        """
        Converts pandas DataFrame to XGBoost DMatrix, handling categoricals
        if enable_categorical is set to True.
        """
        enable_cat = self.config.get("model", {}).get("enable_categorical", True)
        
        # XGBoost DMatrix creation
        return xgb.DMatrix(
            data=X,
            label=y,
            enable_categorical=enable_cat
        )

    def train(self, 
              X_train: pd.DataFrame, 
              y_train: pd.Series, 
              X_val: pd.DataFrame, 
              y_val: pd.Series,
              params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Trains the XGBoost model using Early Stopping on the validation set.
        """
        logger.info("[Trainer] Preparing DMatrices...")
        dtrain = self._prepare_dmatrix(X_train, y_train)
        dval = self._prepare_dmatrix(X_val, y_val)

        self.feature_names = X_train.columns.tolist()

        # Build full parameter dictionary
        full_params = {}
        # 1. Base model params (objective, tree_method)
        full_params.update({k: v for k, v in self.config.get("model", {}).items() if not isinstance(v, dict)})
        # 2. Static static_params
        full_params.update(self.config.get("model", {}).get("static_params", {}))
        # 3. Dynamic tuning params (from Optuna)
        if params:
            full_params.update(params)

        # Extract training-specific settings
        num_boost_round = full_params.pop("n_estimators", 1000)
        early_stopping_rounds = full_params.pop("early_stopping_rounds", 50)
        
        evals = [(dtrain, "train"), (dval, "val")]
        
        logger.info(f"[Trainer] Starting training with params: {full_params}")
        
        # Train model
        self.model = xgb.train(
            params=full_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=50
        )
        
        self.best_iteration = self.model.best_iteration
        
        logger.info(f"[Trainer] Training completed. Best iteration: {self.best_iteration}")
        
        # Extract evaluation history if needed
        # (For advanced logging, but usually Optuna captures the final metric)
        return {"best_iteration": self.best_iteration, "best_score": self.model.best_score}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates point predictions.
        Ensures strict non-negativity since we are predicting sales.
        """
        if self.model is None:
            raise ValueError("Model is not trained. Call train() first or load a saved model.")
            
        dtest = self._prepare_dmatrix(X)
        
        # Point prediction using best iteration
        preds = self.model.predict(dtest, iteration_range=(0, self.best_iteration + 1 if self.best_iteration else 0))
        
        # Sales cannot be negative
        preds = np.maximum(preds, 0)
        
        return preds

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Returns Gain and Weight importance for all features.
        """
        if self.model is None:
            raise ValueError("Model is not trained.")
            
        gain = self.model.get_score(importance_type="gain")
        weight = self.model.get_score(importance_type="weight")
        
        df = pd.DataFrame({
            "feature": list(gain.keys()),
            "gain": list(gain.values()),
            "weight": [weight.get(f, 0) for f in gain.keys()]
        })
        return df.sort_values("gain", ascending=False).reset_index(drop=True)

    def save_model(self, path: str):
        """Saves the trained XGBoost model."""
        if self.model is None:
            raise ValueError("Model is not trained.")
        
        out_path = get_project_root() / path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(out_path))
        logger.info(f"[Trainer] Model saved to {out_path}")

    def load_model(self, path: str):
        """Loads a trained XGBoost model."""
        in_path = get_project_root() / path
        if not in_path.exists():
            raise FileNotFoundError(f"Model file not found: {in_path}")
            
        self.model = xgb.Booster()
        self.model.load_model(str(in_path))
        logger.info(f"[Trainer] Model loaded from {in_path}")
