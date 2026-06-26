"""
==============================================================================
Stage 3 Pipeline — Demand Forecasting Engine
==============================================================================
Responsibility:
    Orchestrate data loading, hyperparameter tuning, model training, evaluation,
    and visualization. Outputs the trained XGBoost model and artifacts for 
    Stage 4.
==============================================================================
"""

import argparse
import json
import yaml
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.helpers import setup_logger, get_project_root
from src.models.data_loader import ForecastDataLoader
from src.models.trainer import XGBoostTrainer
from src.models.optimizer import HyperparameterOptimizer
from src.models.evaluator import ForecastEvaluator
from src.models.visualizer import ForecastVisualizer

logger = setup_logger()

def load_config(path: str) -> dict:
    with open(get_project_root() / path, "r") as f:
        return yaml.safe_load(f)

def run_pipeline(dev_mode: bool = False):
    logger.info("============================================================")
    logger.info("STAGE 3 │ Demand Forecasting Engine")
    logger.info(f"Mode    : {'DEV (Subset)' if dev_mode else 'FULL'}")
    logger.info("============================================================")
    
    root = get_project_root()
    config = load_config("config/forecasting.yaml")
    
    # Ensure output directories exist
    out_dirs = config.get("outputs", {})
    for d in out_dirs.values():
        (root / d).mkdir(parents=True, exist_ok=True)
        
    if dev_mode:
        config["tuning"]["n_trials"] = 3
        config["model"]["static_params"]["n_estimators"] = 50
    
    # 1. Load Data
    loader = ForecastDataLoader()
    # To save memory, load Train and Val first for Optuna, drop Train, then load Test.
    # We will need historical train target for RMSSE, so we'll keep that series.
    
    logger.info("\n[Step 1/5] Loading Data...")
    X_train, y_train = loader.load_split("train", dev_mode=dev_mode)
    X_val, y_val = loader.load_split("val", dev_mode=dev_mode)
    
    # Keep train series and items for RMSSE calculation later
    train_y_series = y_train.copy()
    train_item_ids = X_train["item_id"].copy()
    
    # 2. Hyperparameter Optimization
    logger.info("\n[Step 2/5] Hyperparameter Tuning...")
    trainer = XGBoostTrainer(config)
    optimizer = HyperparameterOptimizer(config, trainer)
    
    best_params, _ = optimizer.optimize(X_train, y_train, X_val, y_val)
    
    # Save optimized parameters
    params_path = root / out_dirs["models_dir"] / "optimized_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"Saved optimized parameters to {params_path}")
    
    # 3. Final Model Training
    logger.info("\n[Step 3/5] Final Model Training on best params...")
    # (Usually we would train on Train+Val, but keeping consistent splits is safer for strict validation)
    # We retrain using the best_params.
    trainer.train(X_train, y_train, X_val, y_val, params=best_params)
    
    # Save model
    model_path = out_dirs["models_dir"] + "xgboost_best_model.json"
    trainer.save_model(model_path)
    
    # Save feature importance
    importance_df = trainer.get_feature_importance()
    importance_path = root / out_dirs["reports_dir"] / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    
    # Free memory of X_train before loading Test
    logger.info("Freeing training data memory...")
    del X_train
    del y_train
    
    # 4. Evaluation
    logger.info("\n[Step 4/5] Evaluation...")
    evaluator = ForecastEvaluator()
    
    # Predict Validation
    logger.info("Evaluating Validation Set...")
    val_preds = trainer.predict(X_val)
    val_metrics = evaluator.evaluate(y_val, pd.Series(val_preds), train_y_series, train_item_ids, X_val["item_id"])
    
    # Load and Predict Test
    logger.info("Evaluating Test Set...")
    X_test, y_test = loader.load_split("test", dev_mode=dev_mode)
    test_preds = trainer.predict(X_test)
    test_metrics = evaluator.evaluate(y_test, pd.Series(test_preds), train_y_series, train_item_ids, X_test["item_id"])
    
    # Save evaluation report
    eval_report = {
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_hyperparameters": best_params
    }
    report_path = root / out_dirs["reports_dir"] / "stage3_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=4)
        
    # Combine predictions for downstream stages
    # Note: X_val and X_test still have day_index dropped inside loader, but they preserve original index.
    val_out = pd.DataFrame({"item_id": X_val["item_id"], "prediction": val_preds, "actual": y_val})
    test_out = pd.DataFrame({"item_id": X_test["item_id"], "prediction": test_preds, "actual": y_test})
    
    val_out.to_parquet(root / out_dirs["predictions_dir"] / "val_predictions.parquet")
    test_out.to_parquet(root / out_dirs["predictions_dir"] / "test_predictions.parquet")
    
    # 5. Visualizations
    logger.info("\n[Step 5/5] Diagnostic Visualizations...")
    visualizer = ForecastVisualizer(out_dir=out_dirs["plots_dir"])
    
    visualizer.plot_feature_importance(importance_df)
    visualizer.plot_actual_vs_predicted(y_test.values, test_preds)
    visualizer.plot_residual_distribution(y_test.values, test_preds)
    
    # Prepare a df with day_index for plotting time series
    # We need day_index back. Let's extract it from the original dataset if we can,
    # or just use sequential indices for plotting.
    # A cleaner way is reading day_index directly from features.parquet
    logger.info("Loading day_index for time series plot...")
    test_day_index = pd.read_parquet(loader.features_path, columns=["day_index", "split"], filters=[("split", "==", "test")])
    if dev_mode:
        # Align indices (since loader samples series)
        test_day_index = test_day_index.loc[X_test.index]
        
    ts_df = pd.DataFrame({
        "item_id": X_test["item_id"],
        "day_index": test_day_index["day_index"],
        "sales": y_test,
        "prediction": test_preds
    })
    
    visualizer.plot_forecast_over_time(ts_df)
    
    logger.info("============================================================")
    logger.info("STAGE 3 COMPLETE ✓")
    logger.info("============================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="Run on a small subset of data")
    args = parser.parse_args()
    
    run_pipeline(dev_mode=args.dev)
