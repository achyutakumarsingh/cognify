"""
==============================================================================
Stage 4 — Probabilistic Uncertainty Quantification Engine
==============================================================================
Package exposing the four core uncertainty quantification components:

  QuantilePredictor    — XGBoost quantile regression (reg:quantilereg)
  ConformalPredictor   — Split conformal prediction (distribution-free)
  IntervalEvaluator    — Proper scoring rules & coverage metrics
  UncertaintyVisualizer — Diagnostic plots for interval analysis
==============================================================================
"""

from src.uncertainty.quantile_predictor import QuantilePredictor
from src.uncertainty.conformal_predictor import ConformalPredictor
from src.uncertainty.interval_evaluator import IntervalEvaluator
from src.uncertainty.uncertainty_visualizer import UncertaintyVisualizer

__all__ = [
    "QuantilePredictor",
    "ConformalPredictor",
    "IntervalEvaluator",
    "UncertaintyVisualizer",
]
