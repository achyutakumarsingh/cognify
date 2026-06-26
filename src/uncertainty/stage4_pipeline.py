"""
==============================================================================
Stage 4 Pipeline — Probabilistic Uncertainty Quantification Engine
==============================================================================
Orchestrates the full Stage 4 workflow:

  Step 1  │ Load config and Stage 3 artifacts
  Step 2  │ Load feature splits via ForecastDataLoader
  Step 3  │ Train Quantile Regression models (q=0.05, 0.50, 0.95)
  Step 4  │ Calibrate Conformal Predictor from validation residuals
  Step 5  │ Generate test-set prediction intervals (both methods)
  Step 6  │ Evaluate both methods with IntervalEvaluator
  Step 7  │ Produce all diagnostic visualizations
  Step 8  │ Save all artifacts for Stage 5 consumption

Stage 5 Compatibility
---------------------
This pipeline saves a ``stage5_inputs`` key inside every evaluation result
dict that is persisted to ``stage4_uncertainty_report.json``.  Stage 5 can
load this file directly and access:
  - empirical coverage per method
  - coverage error per method
  - per-item interval bounds
  - raw conformity scores (for calibration curve construction)
  - Winkler scores (for reliability analysis)

No Stage 4 code changes are needed for Stage 5 integration.
==============================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils.helpers import setup_logger, get_project_root
from src.models.data_loader import ForecastDataLoader
from src.uncertainty.quantile_predictor import QuantilePredictor
from src.uncertainty.conformal_predictor import ConformalPredictor
from src.uncertainty.interval_evaluator import IntervalEvaluator
from src.uncertainty.uncertainty_visualizer import UncertaintyVisualizer

logger = setup_logger(log_file="outputs/logs/stage4_uncertainty.log")


# ──────────────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    root = get_project_root()
    with open(root / path, "r") as f:
        return yaml.safe_load(f)


def ensure_output_dirs(config: dict, root: Path) -> None:
    outputs = config.get("outputs", {})
    for key, rel_path in outputs.items():
        (root / rel_path).mkdir(parents=True, exist_ok=True)
    (root / "outputs" / "logs").mkdir(parents=True, exist_ok=True)
    (root / "models" / "quantile").mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Artifact I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_json(data: dict, path: Path) -> None:
    """Save a dict to JSON, converting numpy scalars to Python native types."""

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float) and (obj != obj):  # nan
            return None
        return obj

    class _Encoder(json.JSONEncoder):
        def default(self, o):
            try:
                return _convert(o)
            except TypeError:
                return super().default(o)

    with open(path, "w") as f:
        json.dump(data, f, indent=4, cls=_Encoder)
    logger.info(f"Saved JSON → {path}")


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(str(path), index=False)
    logger.info(f"Saved parquet → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(dev_mode: bool = False) -> None:
    t_start = time.time()
    root = get_project_root()

    logger.info("=" * 68)
    logger.info("STAGE 4 │ Probabilistic Uncertainty Quantification Engine")
    logger.info(f"Mode    : {'DEV (Subset)' if dev_mode else 'FULL'}")
    logger.info("=" * 68)

    # ── Config ───────────────────────────────────────────────────────────────
    config = load_config("config/uncertainty.yaml")
    ensure_output_dirs(config, root)

    alpha = config["pipeline"]["alpha"]
    seed = config["pipeline"].get("seed", 42)
    np.random.seed(seed)

    artifacts_cfg = config.get("artifacts", {})
    outputs_cfg = config.get("outputs", {})
    pred_dir = root / outputs_cfg["predictions_dir"]
    rep_dir = root / outputs_cfg["reports_dir"]

    # ── Step 1: Load Stage 3 artifacts ───────────────────────────────────────
    logger.info("\n[Step 1/8] Loading Stage 3 artifacts...")

    s3_cfg = config.get("stage3_artifacts", {})
    val_pred_path = root / s3_cfg["val_predictions_path"]
    test_pred_path = root / s3_cfg["test_predictions_path"]

    val_pred_df = pd.read_parquet(val_pred_path)
    test_pred_df = pd.read_parquet(test_pred_path)

    logger.info(
        f"  Val predictions : {val_pred_df.shape}  "
        f"({val_pred_df['item_id'].nunique()} items)"
    )
    logger.info(
        f"  Test predictions: {test_pred_df.shape}  "
        f"({test_pred_df['item_id'].nunique()} items)"
    )

    # ── Step 2: Load feature splits ───────────────────────────────────────────
    logger.info("\n[Step 2/8] Loading feature splits via ForecastDataLoader...")
    loader = ForecastDataLoader()

    X_train, y_train = loader.load_split("train", dev_mode=dev_mode)
    X_val, y_val = loader.load_split("val", dev_mode=dev_mode)
    X_test, y_test = loader.load_split("test", dev_mode=dev_mode)

    # Keep store_id for stratified evaluation before dropping
    test_store_ids = X_test["store_id"].copy()
    test_item_ids = X_test["item_id"].copy()
    val_item_ids = X_val["item_id"].copy()

    # Keep training series for MSIS scale
    train_y_arr = y_train.values.astype(float)

    logger.info(
        f"  Train: X={X_train.shape}, val: X={X_val.shape}, test: X={X_test.shape}"
    )

    # Use strictly train for training, val for early stopping
    X_q_train = X_train
    y_q_train = y_train
    X_q_val = X_val
    y_q_val = y_val

    logger.info(f"  Quantile training size: {X_q_train.shape}")

    # Free X_train memory now that we have X_q_train
    del X_train, y_train

    # ── Step 3: Quantile Regression ───────────────────────────────────────────
    logger.info("\n[Step 3/8] Training Quantile Regression models...")
    t3 = time.time()

    qp = QuantilePredictor(config, stage3_params_path=s3_cfg["optimized_params_path"])

    qp.fit(
        X_q_train, y_q_train,
        X_val=X_q_val, y_val=y_q_val,
    )
    qp.save_models(out_dir="models/quantile/")

    logger.info(f"  Quantile training time: {time.time() - t3:.1f}s")

    # ── Step 4: Conformal Prediction — Calibration ────────────────────────────
    logger.info("\n[Step 4/8] Calibrating Conformal Predictor from val residuals...")
    t4 = time.time()

    cp = ConformalPredictor(
        config=config,
        model_path=s3_cfg["model_path"],
        alpha=alpha,
    )
    cp.load_stage3_model()

    # Calibrate from val split: use exact same predictions that Stage 3 saved
    cp.calibrate(
        y_cal=val_pred_df["actual"].astype(float),
        y_pred_cal=val_pred_df["prediction"].values.astype(float),
    )

    cal_summary = cp.get_calibration_summary()
    logger.info(
        f"  Calibration complete. n={cal_summary['n_calibration']:,}  "
        f"q̂_sym={cal_summary['q_hat_symmetric']:.4f}"
    )
    logger.info(f"  Conformal calibration time: {time.time() - t4:.1f}s")

    # ── Step 5: Generate test-set prediction intervals ─────────────────────────
    logger.info("\n[Step 5/8] Generating test-set prediction intervals...")

    # 5a. Quantile intervals
    logger.info("  [5a] Quantile prediction intervals...")
    q_interval_df = qp.predict_interval(X_test, alpha=alpha)
    # Also get full quantile predictions
    q_quantiles_df = qp.predict_quantiles(X_test)

    # Enforce non-negativity
    q_interval_df["lower"] = np.maximum(q_interval_df["lower"].values, 0.0)
    q_interval_df["upper"] = np.maximum(q_interval_df["upper"].values, 0.0)

    # 5b. Conformal intervals (symmetric)
    logger.info("  [5b] Conformal prediction intervals...")
    cp_interval_df = cp.predict_interval(X_test, alpha=alpha, use_asymmetric=False)
    cp_interval_asym_df = cp.predict_interval(X_test, alpha=alpha, use_asymmetric=True)

    # 5c. Build full output DataFrames
    quantile_out = pd.DataFrame({
        "item_id": test_item_ids.values,
        "store_id": test_store_ids.values,
        "actual": y_test.values,
        "q05": q_quantiles_df["q05"].values,
        "q50": q_quantiles_df["q50"].values,
        "q95": q_quantiles_df["q95"].values,
        "lower": q_interval_df["lower"].values,
        "median": q_interval_df["median"].values,
        "upper": q_interval_df["upper"].values,
        "width": q_interval_df["upper"].values - q_interval_df["lower"].values,
    })

    conformal_out = pd.DataFrame({
        "item_id": test_item_ids.values,
        "store_id": test_store_ids.values,
        "actual": y_test.values,
        "lower_sym": cp_interval_df["lower"].values,
        "point": cp_interval_df["point"].values,
        "upper_sym": cp_interval_df["upper"].values,
        "width_sym": cp_interval_df["width"].values,
        "lower_asym": cp_interval_asym_df["lower"].values,
        "upper_asym": cp_interval_asym_df["upper"].values,
        "width_asym": cp_interval_asym_df["width"].values,
    })

    # Verify monotonicity: q05 ≤ q50 ≤ q95
    n_cross = int(
        ((quantile_out["q05"] > quantile_out["q50"]) | (quantile_out["q50"] > quantile_out["q95"])).sum()
    )
    logger.info(
        f"  Quantile monotonicity violations after correction: {n_cross} "
        f"({100*n_cross/len(quantile_out):.2f}%)"
    )

    # ── Step 6: Evaluation ─────────────────────────────────────────────────────
    logger.info("\n[Step 6/8] Evaluating prediction intervals...")

    evaluator = IntervalEvaluator(alpha=alpha)

    y_true_arr = y_test.values.astype(float)

    # Quantile evaluation
    q_result = evaluator.evaluate(
        y_true=y_test,
        lower=quantile_out["lower"].values,
        upper=quantile_out["upper"].values,
        method_name="quantile_regression",
        item_ids=test_item_ids,
        store_ids=test_store_ids,
        train_y=train_y_arr,
        alpha=alpha,
    )

    # Conformal evaluation (symmetric)
    cp_result = evaluator.evaluate(
        y_true=y_test,
        lower=conformal_out["lower_sym"].values,
        upper=conformal_out["upper_sym"].values,
        method_name="split_conformal",
        item_ids=test_item_ids,
        store_ids=test_store_ids,
        train_y=train_y_arr,
        alpha=alpha,
    )

    # Conformal evaluation (asymmetric) — supplementary
    cp_asym_result = evaluator.evaluate(
        y_true=y_test,
        lower=conformal_out["lower_asym"].values,
        upper=conformal_out["upper_asym"].values,
        method_name="split_conformal_asymmetric",
        item_ids=test_item_ids,
        store_ids=test_store_ids,
        train_y=train_y_arr,
        alpha=alpha,
    )

    # Comparison table
    comparison_df = evaluator.compare([q_result, cp_result, cp_asym_result])
    logger.info("\n" + comparison_df.to_string(index=False))

    # Per-group coverage DataFrames for visualizer
    item_eval_q = pd.DataFrame(q_result["per_item_coverage"] or [])
    item_eval_cp = pd.DataFrame(cp_result["per_item_coverage"] or [])
    store_eval_q = pd.DataFrame(q_result["per_store_coverage"] or [])
    store_eval_cp = pd.DataFrame(cp_result["per_store_coverage"] or [])

    # ── Step 7: Visualizations ─────────────────────────────────────────────────
    logger.info("\n[Step 7/8] Generating diagnostic visualizations...")

    # Prepare test_df for time-series plot
    test_day_index = pd.read_parquet(
        loader.features_path,
        columns=["day_index", "split"],
        filters=[("split", "==", "test")],
    )
    if dev_mode:
        test_day_index = test_day_index.loc[X_test.index]

    ts_test_df = pd.DataFrame({
        "item_id": test_item_ids.values,
        "day_index": test_day_index["day_index"].values,
        "actual": y_true_arr,
    })

    visualizer = UncertaintyVisualizer(config, out_dir=outputs_cfg["plots_dir"])

    # 1. Interval over time
    visualizer.plot_interval_over_time(
        test_df=ts_test_df,
        quantile_intervals=q_interval_df,
        conformal_intervals=cp_interval_df,
    )

    # 2. Width histogram
    q_widths = quantile_out["width"].values
    cp_widths = conformal_out["width_sym"].values
    visualizer.plot_interval_width_histogram(q_widths, cp_widths)

    # 3. Uncertainty by item
    if not item_eval_q.empty and not item_eval_cp.empty:
        visualizer.plot_uncertainty_by_item(item_eval_q, item_eval_cp)
    else:
        logger.warning("[Stage4] Skipping uncertainty_by_item: empty item eval DataFrames")

    # 4. Uncertainty by store
    if not store_eval_q.empty and not store_eval_cp.empty:
        visualizer.plot_uncertainty_by_store(store_eval_q, store_eval_cp)
    else:
        logger.warning("[Stage4] Skipping uncertainty_by_store: empty store eval DataFrames")

    # 5. Conformity score distribution
    cal_scores_arr = np.array(cal_summary["conformity_scores"])
    if len(cal_scores_arr) > 0:
        visualizer.plot_uncertainty_distribution(
            conformity_scores=cal_scores_arr,
            q_hat=cal_summary["q_hat_symmetric"],
            alpha=alpha,
        )

    # 6. Quantile vs Conformal comparison
    q_winkler = np.array(q_result["stage5_inputs"]["winkler_scores_quantile_regression"])
    cp_winkler = np.array(cp_result["stage5_inputs"]["winkler_scores_split_conformal"])
    visualizer.plot_quantile_vs_conformal_comparison(
        q_result=q_result,
        cp_result=cp_result,
        q_winkler=q_winkler,
        cp_winkler=cp_winkler,
        q_widths=q_widths,
        cp_widths=cp_widths,
    )

    # ── Step 8: Save all artifacts ─────────────────────────────────────────────
    logger.info("\n[Step 8/8] Saving all artifacts...")

    # 8a. Prediction parquets
    _save_parquet(quantile_out, pred_dir / artifacts_cfg["quantile_predictions"])
    _save_parquet(conformal_out, pred_dir / artifacts_cfg["conformal_predictions"])

    # 8b. Interval metadata JSON
    interval_metadata = {
        "stage": 4,
        "alpha": alpha,
        "coverage_target": 1.0 - alpha,
        "quantile_regression": {
            "quantiles_trained": qp.quantiles,
            "n_models": len(qp._models),
            "train_rows": int(len(X_q_train)),
            "test_rows": int(len(X_test)),
        },
        "conformal_prediction": cal_summary,
    }
    _save_json(interval_metadata, rep_dir / artifacts_cfg["interval_metadata"])

    # 8c. Full uncertainty report JSON
    # Merge stage5_inputs from all methods into a single top-level section
    merged_stage5 = {}
    for r in [q_result, cp_result, cp_asym_result]:
        merged_stage5.update(r.get("stage5_inputs", {}))

    uncertainty_report = {
        "stage": 4,
        "alpha": alpha,
        "quantile_regression": {k: v for k, v in q_result.items() if k != "stage5_inputs"},
        "split_conformal": {k: v for k, v in cp_result.items() if k != "stage5_inputs"},
        "split_conformal_asymmetric": {k: v for k, v in cp_asym_result.items() if k != "stage5_inputs"},
        "stage5_inputs": merged_stage5,
    }
    _save_json(uncertainty_report, rep_dir / artifacts_cfg["uncertainty_report"])

    # 8d. Comparison table CSV
    comp_path = rep_dir / artifacts_cfg["comparison_table"]
    comparison_df.to_csv(comp_path, index=False)
    logger.info(f"Saved CSV → {comp_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 68)
    logger.info("STAGE 4 COMPLETE ✓")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info("=" * 68)
    logger.info("\nInterval Examples (first 5 test rows):")
    logger.info(
        quantile_out[["actual", "q05", "q50", "q95", "lower", "upper"]].head(5).to_string()
    )
    logger.info("\nAverage Interval Widths:")
    logger.info(f"  Quantile Regression  : {float(np.mean(q_widths)):.4f}")
    logger.info(f"  Conformal (symmetric): {float(np.mean(cp_widths)):.4f}")
    logger.info(f"\nExpected Theoretical Coverage: ≥ {1.0 - alpha:.0%}")
    logger.info(f"Empirical Coverage — Quantile : {q_result['empirical_coverage']:.4f}")
    logger.info(f"Empirical Coverage — Conformal: {cp_result['empirical_coverage']:.4f}")
    logger.info(
        f"\nConformal coverage guarantee holds: "
        f"{'✓' if cp_result['empirical_coverage'] >= (1.0 - alpha) else '✗ (check calibration set size)'}"
    )
    logger.info("\nArtifacts saved:")
    logger.info(f"  ✓ outputs/predictions/quantile_predictions.parquet")
    logger.info(f"  ✓ outputs/predictions/conformal_predictions.parquet")
    logger.info(f"  ✓ outputs/reports/stage4_interval_metadata.json")
    logger.info(f"  ✓ outputs/reports/stage4_uncertainty_report.json")
    logger.info(f"  ✓ outputs/reports/stage4_comparison_table.csv")
    logger.info(f"  ✓ outputs/plots/interval_over_time.html")
    logger.info(f"  ✓ outputs/plots/interval_width_histogram.png")
    logger.info(f"  ✓ outputs/plots/uncertainty_by_item.png")
    logger.info(f"  ✓ outputs/plots/uncertainty_by_store.html")
    logger.info(f"  ✓ outputs/plots/uncertainty_distribution.png")
    logger.info(f"  ✓ outputs/plots/quantile_vs_conformal_comparison.png")
    logger.info(f"  ✓ models/quantile/  (3 quantile model files)")
    logger.info("\nUncertainty engine is ready for Stage 5. ✓")
    logger.info("=" * 68)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 4 — Probabilistic Uncertainty Quantification Engine"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run on a small subset of data for rapid iteration.",
    )
    args = parser.parse_args()
    run_pipeline(dev_mode=args.dev)
