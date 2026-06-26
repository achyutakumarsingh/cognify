"""
==============================================================================
Interval Evaluator — Stage 4 Uncertainty Quantification
==============================================================================
Evaluates probabilistic prediction intervals using proper scoring rules and
coverage-based diagnostics.

Metrics Implemented
-------------------
1. **Empirical Coverage**
   Proportion of test observations falling within the interval:
       Coverage = mean(lower_i ≤ y_i ≤ upper_i)
   Must be ≥ 1−α for a well-calibrated method.

2. **Average Interval Width (AIW)**
   Mean interval width — the primary sharpness measure:
       AIW = mean(upper_i − lower_i)
   Lower is better given equivalent coverage.

3. **Interval Score (IS) — Gneiting & Raftery (2007)**
   A strictly proper scoring rule that penalises width AND miscoverage:
       IS_α(y, [l, u]) = (u − l)
                       + (2/α)(l − y) · 1{y < l}
                       + (2/α)(y − u) · 1{y > u}
   Lower is better.  The penalty term (2/α) dominates when coverage fails.

4. **Winkler Score**
   Equivalent to IS; computed per observation and reported as the mean.

5. **Mean Scaled Interval Score (MSIS)**
   IS normalised by the naive one-step-ahead forecast MAE on the training
   series (scale factor), making it comparable across series of different
   magnitudes.  Aligned with M5 competition evaluation philosophy.

6. **Coverage by Stratum**
   Empirical coverage broken down by:
     • item_id  — identifies which products have poor interval calibration
     • store_id — identifies which locations have poor calibration
     • zero-demand rows (actual = 0) — critical for zero-inflated demand

7. **Interval Width Distribution Summary**
   min, p25, median, p75, p95, max, mean, std of widths.

8. **Sharpness Score**
   Conditional mean width, given that coverage holds:
       Sharpness = mean(upper_i − lower_i | lower_i ≤ y_i ≤ upper_i)
   Measures interval tightness within the covered region.

Stage 5 Output Contract
-----------------------
All metrics are returned in a structured dict and persisted to
``stage4_uncertainty_report.json``.  The top-level ``stage5_inputs`` key
contains:
  - empirical_coverage_{method}
  - coverage_error_{method}
  - per_item_coverage_{method}
  - winkler_scores_{method} (full array)

Stage 5 can load these directly without calling any Stage 4 code.

References
----------
Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
prediction, and estimation. Journal of the American Statistical
Association, 102(477), 359–378.
==============================================================================
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.helpers import setup_logger

logger = setup_logger()


class IntervalEvaluator:
    """
    Evaluates probabilistic prediction intervals using proper scoring rules,
    coverage metrics, and stratified diagnostics.

    Parameters
    ----------
    alpha : float
        Nominal miscoverage level (default 0.10 → 90% intervals).
    """

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Core scalar metrics
    # ------------------------------------------------------------------

    @staticmethod
    def empirical_coverage(
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> float:
        """Proportion of observations within [lower, upper]."""
        covered = (y_true >= lower) & (y_true <= upper)
        return float(np.mean(covered))

    @staticmethod
    def average_interval_width(
        lower: np.ndarray, upper: np.ndarray
    ) -> float:
        """Mean width of the prediction intervals."""
        return float(np.mean(upper - lower))

    def interval_score(
        self,
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """
        Per-observation Interval Score (Winkler Score).

        IS_α(y, [l, u]) = (u − l)
                        + (2/α)(l − y)⁺
                        + (2/α)(y − u)⁺

        Parameters
        ----------
        y_true : np.ndarray
        lower, upper : np.ndarray
        alpha : float, optional — overrides instance alpha

        Returns
        -------
        np.ndarray of per-observation interval scores.
        """
        eff_alpha = alpha if alpha is not None else self.alpha
        width = upper - lower
        penalty_lo = (2.0 / eff_alpha) * np.maximum(lower - y_true, 0.0)
        penalty_hi = (2.0 / eff_alpha) * np.maximum(y_true - upper, 0.0)
        return width + penalty_lo + penalty_hi

    def mean_interval_score(
        self,
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        alpha: Optional[float] = None,
    ) -> float:
        """Mean Interval Score (mean Winkler Score)."""
        return float(np.mean(self.interval_score(y_true, lower, upper, alpha)))

    def msis(
        self,
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        train_y: Optional[np.ndarray] = None,
        alpha: Optional[float] = None,
    ) -> float:
        """
        Mean Scaled Interval Score.

        MSIS = MIS / scale,  where scale = mean(|y_t - y_{t-1}|) on train.

        If train_y is None, falls back to mean(|y_true|) as the scale.
        """
        mis = self.mean_interval_score(y_true, lower, upper, alpha)
        if train_y is not None and len(train_y) > 1:
            scale = float(np.mean(np.abs(np.diff(train_y.astype(float)))))
            if scale == 0.0:
                scale = 1.0
        else:
            scale = float(np.mean(np.abs(y_true))) or 1.0
        return mis / scale

    def sharpness(
        self,
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> float:
        """
        Conditional interval width given coverage.

        Reports the mean width restricted to observations that ARE covered
        by the interval.  A well-calibrated method with narrow sharpness
        is preferred.
        """
        covered_mask = (y_true >= lower) & (y_true <= upper)
        if not np.any(covered_mask):
            return float("nan")
        covered_widths = (upper - lower)[covered_mask]
        return float(np.mean(covered_widths))

    # ------------------------------------------------------------------
    # Stratified coverage
    # ------------------------------------------------------------------

    def coverage_by_group(
        self,
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        group_labels: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compute empirical coverage and mean width for each group.

        Parameters
        ----------
        y_true, lower, upper : np.ndarray
        group_labels : np.ndarray
            Group membership (e.g., item_id, store_id).

        Returns
        -------
        pd.DataFrame with columns: group, n, coverage, mean_width, mis.
        """
        df = pd.DataFrame(
            {
                "y": y_true,
                "lower": lower,
                "upper": upper,
                "group": group_labels,
            }
        )
        df["covered"] = (df["y"] >= df["lower"]) & (df["y"] <= df["upper"])
        df["width"] = df["upper"] - df["lower"]
        df["is_score"] = self.interval_score(y_true, lower, upper)

        agg = (
            df.groupby("group")
            .agg(
                n=("y", "count"),
                coverage=("covered", "mean"),
                mean_width=("width", "mean"),
                mis=("is_score", "mean"),
            )
            .reset_index()
        )
        return agg.sort_values("coverage").reset_index(drop=True)

    def zero_demand_coverage(
        self,
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> Dict[str, float]:
        """
        Coverage metrics restricted to zero-demand rows (actual = 0).

        Critical for zero-inflated demand: a well-calibrated interval
        should still include 0 for ~(1-alpha) of zero-demand observations.
        """
        mask = y_true == 0
        n_zeros = int(np.sum(mask))
        if n_zeros == 0:
            return {"n_zeros": 0, "zero_coverage": float("nan"), "zero_mean_width": float("nan")}

        zero_coverage = self.empirical_coverage(
            y_true[mask], lower[mask], upper[mask]
        )
        zero_mean_width = self.average_interval_width(lower[mask], upper[mask])

        return {
            "n_zeros": n_zeros,
            "zero_fraction": float(n_zeros / len(y_true)),
            "zero_coverage": zero_coverage,
            "zero_mean_width": zero_mean_width,
        }

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true: pd.Series,
        lower: np.ndarray,
        upper: np.ndarray,
        method_name: str = "method",
        item_ids: Optional[pd.Series] = None,
        store_ids: Optional[pd.Series] = None,
        train_y: Optional[np.ndarray] = None,
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run the full evaluation suite for one UQ method.

        Parameters
        ----------
        y_true : pd.Series — true test targets
        lower, upper : np.ndarray — interval bounds
        method_name : str — label for this method
        item_ids, store_ids : pd.Series, optional — group labels
        train_y : np.ndarray, optional — training series for MSIS scale
        alpha : float, optional — override instance alpha

        Returns
        -------
        dict with full metric breakdown.
        """
        eff_alpha = alpha if alpha is not None else self.alpha
        y_arr = y_true.values.astype(float)
        lower_arr = lower.astype(float)
        upper_arr = upper.astype(float)

        # ── Core metrics ─────────────────────────────────────────────────────
        cov = self.empirical_coverage(y_arr, lower_arr, upper_arr)
        aiw = self.average_interval_width(lower_arr, upper_arr)
        mis_val = self.mean_interval_score(y_arr, lower_arr, upper_arr, eff_alpha)
        msis_val = self.msis(y_arr, lower_arr, upper_arr, train_y, eff_alpha)
        sharp = self.sharpness(y_arr, lower_arr, upper_arr)
        cov_error = abs(cov - (1.0 - eff_alpha))

        # ── Width distribution ────────────────────────────────────────────────
        widths = upper_arr - lower_arr
        width_stats = {
            "min": float(np.min(widths)),
            "p25": float(np.percentile(widths, 25)),
            "median": float(np.median(widths)),
            "p75": float(np.percentile(widths, 75)),
            "p95": float(np.percentile(widths, 95)),
            "max": float(np.max(widths)),
            "mean": float(np.mean(widths)),
            "std": float(np.std(widths)),
        }

        # ── Winkler scores (raw array) ────────────────────────────────────────
        winkler_scores = self.interval_score(y_arr, lower_arr, upper_arr, eff_alpha)

        # ── Per-item coverage ─────────────────────────────────────────────────
        per_item_coverage: Optional[Dict] = None
        if item_ids is not None:
            item_df = self.coverage_by_group(
                y_arr, lower_arr, upper_arr, item_ids.values
            )
            per_item_coverage = item_df.to_dict(orient="records")

        # ── Per-store coverage ────────────────────────────────────────────────
        per_store_coverage: Optional[Dict] = None
        if store_ids is not None:
            store_df = self.coverage_by_group(
                y_arr, lower_arr, upper_arr, store_ids.values
            )
            per_store_coverage = store_df.to_dict(orient="records")

        # ── Zero-demand coverage ──────────────────────────────────────────────
        zero_metrics = self.zero_demand_coverage(y_arr, lower_arr, upper_arr)

        # ── Log summary ──────────────────────────────────────────────────────
        logger.info("━" * 56)
        logger.info(f"Interval Evaluation — {method_name}")
        logger.info("━" * 56)
        logger.info(f"  Coverage Target  : {1.0 - eff_alpha:.1%}")
        logger.info(f"  Empirical Coverage: {cov:.4f} ({cov:.1%})")
        logger.info(f"  Coverage Error    : {cov_error:.4f}")
        logger.info(f"  Avg Interval Width: {aiw:.4f}")
        logger.info(f"  Mean IS (Winkler) : {mis_val:.4f}")
        logger.info(f"  MSIS              : {msis_val:.4f}")
        logger.info(f"  Sharpness         : {sharp:.4f}")
        logger.info(f"  Zero-demand cov   : {zero_metrics.get('zero_coverage', 'N/A')}")
        logger.info("━" * 56)

        result = {
            "method": method_name,
            "alpha": eff_alpha,
            "coverage_target": 1.0 - eff_alpha,
            "empirical_coverage": cov,
            "coverage_error": cov_error,
            "average_interval_width": aiw,
            "mean_interval_score": mis_val,
            "msis": msis_val,
            "sharpness": sharp,
            "width_distribution": width_stats,
            "zero_demand_metrics": zero_metrics,
            "per_item_coverage": per_item_coverage,
            "per_store_coverage": per_store_coverage,
            # Stage 5 inputs: raw arrays for calibration curve construction
            "stage5_inputs": {
                f"empirical_coverage_{method_name}": cov,
                f"coverage_error_{method_name}": cov_error,
                f"per_item_coverage_{method_name}": per_item_coverage,
                f"winkler_scores_{method_name}": winkler_scores.tolist(),
                f"lower_{method_name}": lower_arr.tolist(),
                f"upper_{method_name}": upper_arr.tolist(),
                f"y_true": y_arr.tolist(),
            },
        }
        return result

    def compare(
        self,
        results: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Build a comparison table across multiple UQ methods.

        Parameters
        ----------
        results : list of dicts returned by evaluate()

        Returns
        -------
        pd.DataFrame with one row per method, columns for each key metric.
        """
        rows = []
        for r in results:
            rows.append(
                {
                    "Method": r["method"],
                    "Coverage Target": f"{r['coverage_target']:.0%}",
                    "Empirical Coverage": f"{r['empirical_coverage']:.4f}",
                    "Coverage Error": f"{r['coverage_error']:.4f}",
                    "Avg Interval Width": f"{r['average_interval_width']:.4f}",
                    "Mean IS (Winkler)": f"{r['mean_interval_score']:.4f}",
                    "MSIS": f"{r['msis']:.4f}",
                    "Sharpness": f"{r['sharpness']:.4f}",
                    "Zero Coverage": f"{r['zero_demand_metrics'].get('zero_coverage', float('nan')):.4f}",
                }
            )
        return pd.DataFrame(rows)
