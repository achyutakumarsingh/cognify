"""
==============================================================================
Conformal Predictor — Stage 4 Uncertainty Quantification
==============================================================================
Implements Split Conformal Prediction using the Stage 3 trained XGBoost model.

Mathematical Foundation
-----------------------
Split Conformal Prediction (Vovk et al. 2005; Angelopoulos & Bates 2021)
provides a *distribution-free, finite-sample coverage guarantee* without
any assumption about the data distribution.

Protocol
--------
Given a pre-trained model f and a held-out calibration set
D_cal = {(x_1, y_1), ..., (x_n, y_n)}:

  1. Compute nonconformity (conformity) scores:
       s_i = |y_i − f(x_i)|      (absolute residual)

  2. Compute the corrected empirical quantile:
       q̂ = Quantile(s_1,...,s_n, level=⌈(n+1)(1−α)⌉/n)

     The ⌈(n+1)(1−α)⌉/n correction accounts for the finite calibration
     set size and ensures the marginal coverage guarantee holds exactly.
     Without it, coverage is conservative only in the limit n → ∞.

  3. For any new test point x_test:
       C(x_test) = [f(x_test) − q̂,  f(x_test) + q̂]

  4. Theoretical guarantee (exchangeability assumption):
       P(y_test ∈ C(x_test)) ≥ 1 − α

Symmetric vs Asymmetric Intervals
----------------------------------
The standard SCP produces *symmetric* intervals centred on the point
prediction.  We also implement an *asymmetric* variant using *signed*
residuals to account for the directional bias in zero-inflated demand:

  s_i⁺ = y_i − f(x_i)   (signed residual, captures over/under-prediction)

  q̂_lo = Quantile(s_1,...,s_n, level=⌈(n+1)(α/2)⌉/n)
  q̂_hi = Quantile(s_1,...,s_n, level=⌈(n+1)(1−α/2)⌉/n)

  C_asym(x_test) = [f(x_test) + q̂_lo,  f(x_test) + q̂_hi]

This asymmetric form produces shorter intervals on the low-demand side
while expanding on the high-demand side — more appropriate for retail.

Calibration Dataset
--------------------
We use the Stage 3 *validation split* (28 days, n=853,720 rows in full
mode; n=5,600 in dev mode) as the calibration set.  This split was never
used during Stage 3 Optuna tuning (which used it only for early stopping
evaluation), so the conformity scores are unbiased.

Theoretical Coverage Guarantee
--------------------------------
Under the exchangeability assumption (i.i.d. or, more generally, any
distribution invariant to permutations of the calibration + test set):

    1 − α ≤ P(y_test ∈ C(x_test)) ≤ 1 − α + 1/(n+1)

The upper bound shows that coverage can exceed the target by at most
1/(n+1), which vanishes as n → ∞.  For n=5,600: max overshoot = 0.018%.

References
----------
Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a
Random World. Springer.

Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to
conformal prediction and distribution-free uncertainty quantification.
arXiv:2107.07511.
==============================================================================
"""

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from src.utils.helpers import setup_logger, get_project_root

logger = setup_logger()


class ConformalPredictor:
    """
    Split Conformal Predictor wrapping the Stage 3 XGBoost model.

    Uses the validation split residuals as calibration conformity scores
    to construct marginal prediction intervals with a finite-sample
    coverage guarantee.

    Parameters
    ----------
    config : dict
        Full uncertainty config (loaded from config/uncertainty.yaml).
    model_path : str
        Path (relative to project root) to the Stage 3 XGBoost model.
    alpha : float
        Miscoverage level.  Default 0.10 → 90% intervals.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_path: str = "models/xgboost_best_model.json",
        alpha: Optional[float] = None,
    ) -> None:
        self.config = config
        self.root = get_project_root()
        self.cp_config = config.get("conformal_prediction", {})
        self.pipeline_config = config.get("pipeline", {})

        self.alpha: float = (
            alpha if alpha is not None
            else self.pipeline_config.get("alpha", 0.10)
        )
        self.finite_sample_correction: bool = self.cp_config.get(
            "finite_sample_correction", True
        )
        self.compute_asymmetric: bool = self.cp_config.get(
            "compute_asymmetric", True
        )

        # Stage 3 model (loaded lazily)
        self._model: Optional[xgb.Booster] = None
        self._model_path: str = model_path

        # Calibration state
        self._is_calibrated: bool = False
        self._n_calibration: int = 0
        self._abs_scores: Optional[np.ndarray] = None   # |y - ŷ|
        self._signed_scores: Optional[np.ndarray] = None  # y - ŷ

        # Threshold(s) set during calibration
        self._q_hat_sym: Optional[float] = None          # symmetric threshold
        self._q_hat_lo: Optional[float] = None           # asymmetric lower
        self._q_hat_hi: Optional[float] = None           # asymmetric upper

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_stage3_model(self) -> "ConformalPredictor":
        """Load the Stage 3 XGBoost Booster from disk."""
        path = self.root / self._model_path
        if not path.exists():
            raise FileNotFoundError(
                f"[ConformalPredictor] Stage 3 model not found: {path}"
            )
        self._model = xgb.Booster()
        self._model.load_model(str(path))
        logger.info(
            f"[ConformalPredictor] Stage 3 model loaded from {self._model_path}"
        )
        return self

    def _require_model(self) -> None:
        if self._model is None:
            self.load_stage3_model()

    def _prepare_dmatrix(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> xgb.DMatrix:
        return xgb.DMatrix(data=X, label=y, enable_categorical=True)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _corrected_quantile_level(self, n: int, alpha: float) -> float:
        """
        Compute the finite-sample corrected quantile level.

        q_level = ceil((n + 1) * (1 - alpha)) / n

        Capped at 1.0 to handle edge cases where alpha is very small
        relative to n.
        """
        numerator = math.ceil((n + 1) * (1.0 - alpha))
        level = numerator / n
        return min(level, 1.0)

    def calibrate(
        self,
        y_cal: pd.Series,
        y_pred_cal: np.ndarray,
    ) -> "ConformalPredictor":
        """
        Fit the conformal predictor using calibration residuals.

        Computes and stores the conformity score threshold(s) from the
        calibration set.  Must be called before predict_interval().

        Parameters
        ----------
        y_cal : pd.Series
            True targets on the calibration (validation) split.
        y_pred_cal : np.ndarray
            Point predictions from the Stage 3 model on the same split.

        Returns
        -------
        self
        """
        y_true = np.asarray(y_cal, dtype=float)
        y_pred = np.asarray(y_pred_cal, dtype=float)

        self._n_calibration = len(y_true)
        n = self._n_calibration

        # Absolute nonconformity scores (symmetric)
        self._abs_scores = np.abs(y_true - y_pred)

        # Signed nonconformity scores (asymmetric)
        self._signed_scores = y_true - y_pred

        # ── Symmetric threshold ──────────────────────────────────────────────
        if self.finite_sample_correction:
            level_sym = self._corrected_quantile_level(n, self.alpha)
        else:
            level_sym = 1.0 - self.alpha

        self._q_hat_sym = float(np.quantile(self._abs_scores, level_sym))

        # ── Asymmetric thresholds ────────────────────────────────────────────
        if self.compute_asymmetric:
            if self.finite_sample_correction:
                level_lo = min(
                    math.ceil((n + 1) * (self.alpha / 2)) / n, 1.0
                )
                level_hi = self._corrected_quantile_level(n, self.alpha / 2)
            else:
                level_lo = self.alpha / 2
                level_hi = 1.0 - self.alpha / 2

            self._q_hat_lo = float(
                np.quantile(self._signed_scores, level_lo)
            )
            self._q_hat_hi = float(
                np.quantile(self._signed_scores, level_hi)
            )

        self._is_calibrated = True

        logger.info(
            f"[ConformalPredictor] Calibrated on {n:,} samples. "
            f"alpha={self.alpha:.2f} → q̂_sym={self._q_hat_sym:.4f}"
            + (
                f", q̂_lo={self._q_hat_lo:.4f}, q̂_hi={self._q_hat_hi:.4f}"
                if self.compute_asymmetric
                else ""
            )
        )
        return self

    def calibrate_from_parquet(
        self, parquet_path: str
    ) -> "ConformalPredictor":
        """
        Convenience method: calibrate directly from the Stage 3 val
        predictions parquet file (columns: item_id, prediction, actual).

        Parameters
        ----------
        parquet_path : str
            Path (relative to project root) to val_predictions.parquet.
        """
        path = self.root / parquet_path
        df = pd.read_parquet(path)
        logger.info(
            f"[ConformalPredictor] Calibrating from {parquet_path} "
            f"({len(df):,} rows)"
        )
        return self.calibrate(
            y_cal=df["actual"].astype(float),
            y_pred_cal=df["prediction"].values.astype(float),
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _require_calibration(self) -> None:
        if not self._is_calibrated:
            raise RuntimeError(
                "[ConformalPredictor] Not calibrated. Call calibrate() first."
            )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate point predictions using the Stage 3 model.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray of shape (n,) — non-negative point predictions.
        """
        self._require_model()
        dtest = self._prepare_dmatrix(X)
        preds = self._model.predict(dtest)
        return np.maximum(preds, 0.0)

    def predict_interval(
        self,
        X: pd.DataFrame,
        alpha: Optional[float] = None,
        use_asymmetric: bool = False,
    ) -> pd.DataFrame:
        """
        Construct (1-alpha) conformal prediction intervals.

        Parameters
        ----------
        X : pd.DataFrame
            Test features.
        alpha : float, optional
            Override the instance-level alpha.
        use_asymmetric : bool
            If True, return asymmetric intervals (signed residual variant).

        Returns
        -------
        pd.DataFrame with columns:
          ``lower`` — lower prediction bound (clipped at 0)
          ``point`` — Stage 3 point prediction
          ``upper`` — upper prediction bound
          ``width`` — interval width (upper - lower)
          ``interval_type`` — 'symmetric' or 'asymmetric'
        """
        self._require_calibration()
        self._require_model()

        eff_alpha = alpha if alpha is not None else self.alpha

        # Re-compute thresholds if a different alpha is requested
        if eff_alpha != self.alpha and self._abs_scores is not None:
            n = self._n_calibration
            level = self._corrected_quantile_level(n, eff_alpha) if self.finite_sample_correction else (1.0 - eff_alpha)
            q_sym = float(np.quantile(self._abs_scores, level))
            if self.compute_asymmetric:
                lev_lo = min(math.ceil((n + 1) * (eff_alpha / 2)) / n, 1.0) if self.finite_sample_correction else eff_alpha / 2
                lev_hi = self._corrected_quantile_level(n, eff_alpha / 2) if self.finite_sample_correction else 1.0 - eff_alpha / 2
                q_lo = float(np.quantile(self._signed_scores, lev_lo))
                q_hi = float(np.quantile(self._signed_scores, lev_hi))
            else:
                q_lo = q_hi = None
        else:
            q_sym = self._q_hat_sym
            q_lo = self._q_hat_lo
            q_hi = self._q_hat_hi

        point_preds = self.predict(X)

        if use_asymmetric and self.compute_asymmetric:
            lower = np.maximum(point_preds + q_lo, 0.0)
            upper = np.maximum(point_preds + q_hi, 0.0)
            interval_type = "asymmetric"
        else:
            lower = np.maximum(point_preds - q_sym, 0.0)
            upper = point_preds + q_sym
            interval_type = "symmetric"

        return pd.DataFrame(
            {
                "lower": lower,
                "point": point_preds,
                "upper": upper,
                "width": upper - lower,
                "interval_type": interval_type,
            }
        )

    def predict_both_intervals(
        self,
        X: pd.DataFrame,
        alpha: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Return both symmetric and asymmetric intervals in a single DataFrame.

        Columns: ``lower_sym``, ``point``, ``upper_sym``,
                 ``lower_asym``, ``upper_asym``.
        """
        self._require_calibration()
        sym_df = self.predict_interval(X, alpha=alpha, use_asymmetric=False)
        asym_df = self.predict_interval(X, alpha=alpha, use_asymmetric=True)

        return pd.DataFrame(
            {
                "lower_sym": sym_df["lower"].values,
                "point": sym_df["point"].values,
                "upper_sym": sym_df["upper"].values,
                "width_sym": sym_df["width"].values,
                "lower_asym": asym_df["lower"].values,
                "upper_asym": asym_df["upper"].values,
                "width_asym": asym_df["width"].values,
            }
        )

    # ------------------------------------------------------------------
    # Calibration summary (for Stage 5 & reporting)
    # ------------------------------------------------------------------

    def get_calibration_summary(self) -> Dict[str, Any]:
        """
        Return a structured summary of calibration state.

        This dict is written verbatim into stage4_interval_metadata.json
        and consumed by Stage 5 for coverage analysis.

        Returns
        -------
        dict with keys:
          n_calibration, alpha, q_hat_symmetric, q_hat_lo, q_hat_hi,
          finite_sample_correction, score_type,
          score_stats (min/p25/median/p75/max/mean/std of abs scores)
        """
        self._require_calibration()

        score_stats: Dict[str, float] = {}
        if self._abs_scores is not None:
            score_stats = {
                "min": float(np.min(self._abs_scores)),
                "p25": float(np.percentile(self._abs_scores, 25)),
                "median": float(np.median(self._abs_scores)),
                "p75": float(np.percentile(self._abs_scores, 75)),
                "max": float(np.max(self._abs_scores)),
                "mean": float(np.mean(self._abs_scores)),
                "std": float(np.std(self._abs_scores)),
            }

        return {
            "n_calibration": self._n_calibration,
            "alpha": self.alpha,
            "coverage_target": 1.0 - self.alpha,
            "q_hat_symmetric": self._q_hat_sym,
            "q_hat_lo_asymmetric": self._q_hat_lo,
            "q_hat_hi_asymmetric": self._q_hat_hi,
            "finite_sample_correction": self.finite_sample_correction,
            "score_type": self.cp_config.get(
                "score_type", "absolute_residual"
            ),
            "theoretical_coverage_lower_bound": 1.0 - self.alpha,
            "theoretical_coverage_upper_bound": min(
                1.0,
                1.0 - self.alpha + 1.0 / (self._n_calibration + 1),
            ),
            "conformity_score_stats": score_stats,
            # Expose raw scores for Stage 5 calibration curve
            "conformity_scores": (
                self._abs_scores.tolist()
                if self._abs_scores is not None
                else []
            ),
            "signed_conformity_scores": (
                self._signed_scores.tolist()
                if self._signed_scores is not None
                else []
            ),
        }
