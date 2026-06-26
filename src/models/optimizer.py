"""
==============================================================================
Hyperparameter Optimizer — Supply Chain Risk Triage
==============================================================================
Responsibility:
    Execute systematic Bayesian optimization over XGBoost parameters using 
    Optuna. Evaluates trials using early stopping on the validation set.
==============================================================================
"""

import optuna
import pandas as pd
from typing import Dict, Any, Tuple
import logging

from src.utils.helpers import setup_logger
from src.models.trainer import XGBoostTrainer

logger = setup_logger()

class HyperparameterOptimizer:
    def __init__(self, config: Dict[str, Any], trainer: XGBoostTrainer):
        self.config = config
        self.trainer = trainer
        self.tuning_config = config.get("tuning", {})
        
        # Suppress overly verbose optuna logs unless error
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _objective(self, trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """
        Optuna objective function.
        """
        params_space = self.tuning_config.get("params", {})
        
        # Sample parameters
        sampled_params = {}
        if "learning_rate" in params_space:
            sampled_params["learning_rate"] = trial.suggest_float("learning_rate", params_space["learning_rate"][0], params_space["learning_rate"][1], log=True)
        if "max_depth" in params_space:
            sampled_params["max_depth"] = trial.suggest_int("max_depth", params_space["max_depth"][0], params_space["max_depth"][1])
        if "min_child_weight" in params_space:
            sampled_params["min_child_weight"] = trial.suggest_int("min_child_weight", params_space["min_child_weight"][0], params_space["min_child_weight"][1])
        if "subsample" in params_space:
            sampled_params["subsample"] = trial.suggest_float("subsample", params_space["subsample"][0], params_space["subsample"][1])
        if "colsample_bytree" in params_space:
            sampled_params["colsample_bytree"] = trial.suggest_float("colsample_bytree", params_space["colsample_bytree"][0], params_space["colsample_bytree"][1])
        if "gamma" in params_space:
            sampled_params["gamma"] = trial.suggest_float("gamma", params_space["gamma"][0], params_space["gamma"][1])
        if "reg_alpha" in params_space:
            sampled_params["reg_alpha"] = trial.suggest_float("reg_alpha", params_space["reg_alpha"][0], params_space["reg_alpha"][1])
        if "reg_lambda" in params_space:
            sampled_params["reg_lambda"] = trial.suggest_float("reg_lambda", params_space["reg_lambda"][0], params_space["reg_lambda"][1])

        # Train model for this trial
        # XGBoostTrainer will handle DMatrix creation internally
        try:
            results = self.trainer.train(X_train, y_train, X_val, y_val, params=sampled_params)
            # The trainer sets model.best_score based on the early_stopping evaluation
            best_score = results.get("best_score", float('inf'))
            return best_score
            
        except Exception as e:
            logger.error(f"[Optimizer] Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()

    def optimize(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Dict[str, Any], optuna.Study]:
        """
        Runs the hyperparameter optimization loop.
        """
        n_trials = self.tuning_config.get("n_trials", 10)
        logger.info(f"[Optimizer] Starting Optuna tuning for {n_trials} trials...")
        
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.config.get("pipeline", {}).get("seed", 42)))
        
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            n_jobs=1,  # Keep 1 to avoid excessive memory usage with large datasets
            show_progress_bar=True
        )
        
        logger.info("[Optimizer] Tuning complete!")
        logger.info(f"[Optimizer] Best Trial: {study.best_trial.number}")
        logger.info(f"[Optimizer] Best Value (RMSE): {study.best_value:.4f}")
        logger.info(f"[Optimizer] Best Params:")
        for k, v in study.best_params.items():
            logger.info(f"    {k}: {v}")
            
        return study.best_params, study
