"""
==============================================================================
Quantile Predictor — Stage 4 Uncertainty Quantification
==============================================================================
Implements quantile regression using XGBoost's native ``reg:quantilereg``
objective (available since XGBoost 2.0).

Mathematical Foundation
-----------------------
Quantile regression estimates the conditional quantile function Q(q | x)
by minimising the asymmetric pinball (check) loss:

    L_q(y, ŷ) = q · (y − ŷ)        if y ≥ ŷ   (under-prediction)
              = (q − 1) · (y − ŷ)   if y < ŷ   (over-prediction)

For a 90% prediction interval we train three separate models:
  • q = 0.05  →  lower bound
  • q = 0.50  →  median forecast
  • q = 0.95  →  upper bound

Each model is trained with the *same structural hyperparameters* as the Stage
3 best model (learning rate, depth, regularisation) so that the interval
reflects uncertainty under the same inductive bias.  Only the loss function
differs.

Advantages
----------
  • Asymmetric intervals — naturally handles right-skewed demand distributions
  • No distributional assumption — purely data-driven
  • Zero-inflated robustness — pinball loss penalises sparse-regime errors
    proportionally to the quantile level

Limitations
-----------
  • No finite-sample coverage guarantee (unlike conformal prediction)
  • Quantile crossing is possible when models are trained independently;
    we enforce monotonicity post-hoc via isotonic regression
  • Three separate training runs increase compute cost
  • The 5th/95th percentile models may underfit extreme events

Training Protocol
-----------------
  1. Load Stage 3 best hyperparameters (structure)
  2. Override ``objective`` to "reg:quantilereg" and set ``quantile_alpha``
  3. Train on combined train + val features (calibration split not needed)
  4. Predict on test split
  5. Enforce q05 ≤ q50 ≤ q95 via post-hoc isotonic monotonicity correction
==============================================================================
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from src.utils.helpers import setup_logger, get_project_root

logger = setup_logger()


class QuantilePredictor:
    """
    XGBoost Quantile Regression predictor for probabilistic forecasting.

    Trains one XGBoost model per target quantile using the ``reg:quantilereg``
    objective, then optionally enforces crossing-free monotonicity.

    Parameters
    ----------
    config : dict
        Full uncertainty config (loaded from config/uncertainty.yaml).
    stage3_params_path : str
        Path (relative to project root) to ``optimized_params.json``.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        stage3_params_path: str = "models/optimized_params.json",
    ) -> None:
        self.config = config
        self.root = get_project_root()
        self.qr_config = config.get("quantile_regression", {})
        self.pipeline_config = config.get("pipeline", {})

        self.quantiles: List[float] = self.qr_config.get(
            "quantiles", [0.05, 0.50, 0.95]
        )
        self.seed: int = self.pipeline_config.get("seed", 42)

        # Models keyed by quantile value
        self._models: Dict[float, xgb.Booster] = {}
        self._feature_names: Optional[List[str]] = None
        self._is_fitted: bool = False

        # Load Quantile-specific hyperparameters
        self._hyperparameters: Dict[str, Any] = self.qr_config.get(
            "hyperparameters", {}
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------



    def _build_xgb_params(self, quantile: float) -> Dict[str, Any]:
        """
        Construct the full XGBoost parameter dict for a given quantile.

        Merges:
          1. Stage 3 structural params (learning_rate, max_depth, ...)
          2. Static model settings (tree_method, enable_categorical)
          3. Quantile-specific overrides (objective, quantile_alpha)
        """
        # Base structural params tuned for Quantile Regression
        params: Dict[str, Any] = dict(self._hyperparameters)

        # Static structural settings matching Stage 3 training environment
        params.update(
            {
                "tree_method": "hist",
                "enable_categorical": True,
                "objective": "reg:quantileerror",
                "quantile_alpha": quantile,
                "eval_metric": "quantile",
                "seed": self.seed,
                # n_jobs handled via nthread in native XGBoost API
                "nthread": -1,
            }
        )

        # Remove keys that belong to train() call, not params dict
        for key in ["n_estimators", "early_stopping_rounds", "n_jobs", "random_state", "eval_metric_period"]:
            params.pop(key, None)

        return params

    def _prepare_dmatrix(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> xgb.DMatrix:
        """Convert DataFrame to DMatrix, preserving categorical dtypes."""
        return xgb.DMatrix(
            data=X,
            label=y,
            enable_categorical=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "QuantilePredictor":
        """
        Train one XGBoost quantile model per target quantile.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features (may be combined train + val).
        y_train : pd.Series
            Training targets.
        X_val : pd.DataFrame, optional
            Validation features for early stopping.
        y_val : pd.Series, optional
            Validation targets for early stopping.

        Returns
        -------
        self
        """
        self._feature_names = X_train.columns.tolist()

        n_estimators: int = self.qr_config.get("n_estimators", 300)
        early_stopping_rounds: int = self.qr_config.get(
            "early_stopping_rounds", 30
        )

        dtrain = self._prepare_dmatrix(X_train, y_train)

        has_val = X_val is not None and y_val is not None
        if has_val:
            dval = self._prepare_dmatrix(X_val, y_val)

        for q in self.quantiles:
            logger.info(
                f"[QuantilePredictor] Training quantile q={q:.2f} model "
                f"(n_estimators={n_estimators})..."
            )
            params = self._build_xgb_params(q)

            evals = [(dtrain, "train")]
            if has_val:
                evals.append((dval, "val"))

            callbacks = []
            if has_val:
                callbacks.append(
                    xgb.callback.EarlyStopping(
                        rounds=early_stopping_rounds,
                        metric_name="quantile",
                        data_name="val",
                        save_best=True,
                    )
                )

            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=n_estimators,
                evals=evals,
                callbacks=callbacks if callbacks else None,
                verbose_eval=False,
            )

            self._models[q] = model
            try:
                best_iter = model.best_iteration
            except AttributeError:
                # best_iteration only set when early stopping is active
                best_iter = n_estimators
            logger.info(
                f"[QuantilePredictor] q={q:.2f} trained. "
                f"Best iteration: {best_iter}"
            )

        self._is_fitted = True
        return self

    def _enforce_monotonicity(self, q05: np.ndarray, q50: np.ndarray, q95: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Post-hoc isotonic correction to prevent quantile crossing.

        Stacks predictions as rows and applies isotonic regression per
        observation so that q05 ≤ q50 ≤ q95 is guaranteed.
        """
        n = len(q05)
        # Stack: shape (n, 3)
        stacked = np.column_stack([q05, q50, q95])
        iso = IsotonicRegression(increasing=True)
        corrected = np.zeros_like(stacked)
        for i in range(n):
            corrected[i] = iso.fit_transform([0.05, 0.50, 0.95], stacked[i])
        return corrected[:, 0], corrected[:, 1], corrected[:, 2]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return median (q=0.50) point forecast.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray of shape (n,)
        """
        if not self._is_fitted:
            raise RuntimeError(
                "[QuantilePredictor] Model not fitted. Call fit() first."
            )
        q50 = 0.50
        if q50 not in self._models:
            raise KeyError(
                f"[QuantilePredictor] q=0.50 model not found. "
                f"Available quantiles: {list(self._models.keys())}"
            )
        dtest = self._prepare_dmatrix(X)
        preds = self._models[q50].predict(dtest)
        return np.maximum(preds, 0.0)

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        quantiles: Optional[List[float]] = None,
        enforce_monotonicity: bool = True,
    ) -> pd.DataFrame:
        """
        Return predictions for all trained quantiles.

        Parameters
        ----------
        X : pd.DataFrame
        quantiles : list of float, optional
            Subset of quantiles to predict. Defaults to all trained quantiles.
        enforce_monotonicity : bool
            If True, applies isotonic correction to prevent crossing.

        Returns
        -------
        pd.DataFrame with columns ``q{int(q*100):02d}`` for each quantile,
        e.g. ``q05``, ``q50``, ``q95``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "[QuantilePredictor] Model not fitted. Call fit() first."
            )

        qs = quantiles if quantiles is not None else self.quantiles
        dtest = self._prepare_dmatrix(X)

        raw: Dict[float, np.ndarray] = {}
        for q in qs:
            if q not in self._models:
                raise KeyError(
                    f"[QuantilePredictor] Quantile q={q} model not found."
                )
            preds = self._models[q].predict(dtest)
            raw[q] = np.maximum(preds, 0.0)

        # Enforce monotonicity if standard 3-quantile set
        if (
            enforce_monotonicity
            and 0.05 in raw
            and 0.50 in raw
            and 0.95 in raw
        ):
            raw[0.05], raw[0.50], raw[0.95] = self._enforce_monotonicity(
                raw[0.05], raw[0.50], raw[0.95]
            )
            logger.info(
                "[QuantilePredictor] Monotonicity correction applied."
            )

        result = pd.DataFrame(
            {f"q{int(q * 100):02d}": raw[q] for q in qs}
        )
        return result

    def predict_interval(
        self,
        X: pd.DataFrame,
        alpha: float = 0.10,
    ) -> pd.DataFrame:
        """
        Return a (1-alpha) prediction interval.

        For alpha=0.10 → uses q=0.05 (lower) and q=0.95 (upper).

        Parameters
        ----------
        X : pd.DataFrame
        alpha : float
            Miscoverage level (default 0.10 → 90% interval).

        Returns
        -------
        pd.DataFrame with columns ``lower``, ``median``, ``upper``.
        """
        lower_q = round(alpha / 2, 4)
        upper_q = round(1.0 - alpha / 2, 4)

        # Use closest available quantile if exact not trained
        available = sorted(self._models.keys())

        def _find_closest(target: float) -> float:
            return min(available, key=lambda q: abs(q - target))

        lq = _find_closest(lower_q)
        uq = _find_closest(upper_q)

        df_q = self.predict_quantiles(X, quantiles=[lq, 0.50, uq])

        col_lower = f"q{int(lq * 100):02d}"
        col_median = "q50"
        col_upper = f"q{int(uq * 100):02d}"

        return pd.DataFrame(
            {
                "lower": df_q[col_lower].values,
                "median": df_q[col_median].values,
                "upper": df_q[col_upper].values,
            }
        )

    def save_models(self, out_dir: str = "models/quantile/") -> None:
        """Save all trained quantile models to disk."""
        if not self._is_fitted:
            raise RuntimeError("[QuantilePredictor] No fitted models to save.")
        out_path = self.root / out_dir
        out_path.mkdir(parents=True, exist_ok=True)
        for q, model in self._models.items():
            fname = out_path / f"xgb_quantile_q{int(q * 100):02d}.json"
            model.save_model(str(fname))
            logger.info(f"[QuantilePredictor] Saved q={q:.2f} model → {fname}")

    def load_models(self, out_dir: str = "models/quantile/") -> None:
        """Load previously saved quantile models from disk."""
        out_path = self.root / out_dir
        for q in self.quantiles:
            fname = out_path / f"xgb_quantile_q{int(q * 100):02d}.json"
            if not fname.exists():
                raise FileNotFoundError(
                    f"[QuantilePredictor] Model file not found: {fname}"
                )
            model = xgb.Booster()
            model.load_model(str(fname))
            self._models[q] = model
        self._is_fitted = True
        logger.info(
            f"[QuantilePredictor] Loaded {len(self._models)} quantile models from {out_path}"
        )
