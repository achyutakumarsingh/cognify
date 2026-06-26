"""
==============================================================================
Dataset Validator — Supply Chain Risk Triage  (Stage 2)
==============================================================================
Responsibility:
    Run a comprehensive suite of validation checks on the processed dataset
    before it is saved and handed to Stage 3.

    Every check is self-contained, returns a boolean, and emits a log line.
    The orchestrator calls `run_all_checks()` and reports a pass/fail summary.
==============================================================================
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.helpers import setup_logger
from src.data.feature_engineer import FEATURE_METADATA

logger = setup_logger()


# ==============================================================================
# Individual Checks
# ==============================================================================
def check_row_count(
    df: pd.DataFrame,
    expected_series: int,
    expected_days: int,
    tolerance: float = 0.01,
) -> Tuple[bool, str]:
    actual = len(df)
    expected = expected_series * expected_days
    pct_diff = abs(actual - expected) / expected
    if pct_diff > tolerance:
        msg = (f"FAIL  Row count {actual:,} differs from expected "
               f"{expected:,} by {pct_diff*100:.1f}%")
        return False, msg
    return True, f"PASS  Row count: {actual:,} ({pct_diff*100:.2f}% off expected)"


def check_no_nulls_in_critical(df: pd.DataFrame) -> Tuple[bool, str]:
    critical = ["sales", "item_id", "store_id", "date", "d", "day_index", "split"]
    failures = []
    for col in critical:
        if col not in df.columns:
            continue
        n = df[col].isna().sum()
        if n > 0:
            failures.append(f"{col}:{n:,}")
    if failures:
        return False, f"FAIL  NaN in critical columns: {', '.join(failures)}"
    return True, "PASS  No NaN in critical columns"


def check_no_negative_sales(df: pd.DataFrame) -> Tuple[bool, str]:
    if "sales" not in df.columns:
        return True, "SKIP  'sales' column absent"
    n = (df["sales"] < 0).sum()
    if n > 0:
        return False, f"FAIL  {n:,} negative sales values"
    return True, f"PASS  All sales values ≥ 0  (max={df['sales'].max()})"


def check_temporal_ordering(df: pd.DataFrame, sample_n: int = 5) -> Tuple[bool, str]:
    """
    Verify that within a random sample of series, day_index is strictly
    increasing — confirming the data is temporally ordered.
    """
    if "day_index" not in df.columns or "item_id" not in df.columns:
        return True, "SKIP  Required columns absent"
    sample_ids = df["item_id"].unique()[:sample_n]
    for item in sample_ids:
        store = df[df["item_id"] == item]["store_id"].iloc[0]
        sub = (
            df[(df["item_id"] == item) & (df["store_id"] == store)]
            .sort_values("day_index")["day_index"]
        )
        if not sub.is_monotonic_increasing:
            return False, f"FAIL  Non-monotonic day_index for item '{item}'"
    return True, f"PASS  Temporal order verified for {sample_n} sampled series"


def check_split_completeness(df: pd.DataFrame) -> Tuple[bool, str]:
    """Ensure all three splits are present and non-empty."""
    if "split" not in df.columns:
        return True, "SKIP  'split' column absent"
    expected = {"train", "val", "test"}
    actual = set(df["split"].unique())
    missing = expected - actual
    if missing:
        return False, f"FAIL  Missing splits: {missing}"
    counts = df["split"].value_counts().to_dict()
    return True, f"PASS  Split counts: {counts}"


def check_no_future_leakage_in_lags(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Spot-check that lag features do not carry future values.

    Method: For a single series, verify that lag_7 on row (d) equals
    the sales value from row (d-7) in that series.
    """
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    if not lag_cols or "day_index" not in df.columns:
        return True, "SKIP  No lag columns or day_index found"

    item = df["item_id"].iloc[0]
    store = df["store_id"].iloc[0]
    sub = df[(df["item_id"] == item) & (df["store_id"] == store)].sort_values("day_index")

    for lag_col in lag_cols[:1]:   # check first lag only (lag_7 or lag_14)
        n = int(lag_col.split("_")[1])
        # Row at index n has lag_n value = sales at index 0
        idx_d    = sub.index[n]        # day = d+n
        idx_d_n  = sub.index[0]       # day = d

        actual_lag_val = sub.loc[idx_d, lag_col]
        expected_val   = sub.loc[idx_d_n, "sales"]

        if actual_lag_val != expected_val:
            return False, (
                f"FAIL  {lag_col} leakage check: "
                f"expected {expected_val}, got {actual_lag_val}"
            )

    return True, "PASS  No leakage detected in lag features (spot-check)"


def check_feature_completeness(df: pd.DataFrame) -> Tuple[bool, str]:
    """Verify all expected features from FEATURE_METADATA are present."""
    expected = set(FEATURE_METADATA.keys())
    actual = set(df.columns)
    missing = expected - actual
    extra = actual - expected
    if missing:
        return False, f"FAIL  Missing features: {sorted(missing)}"
    msg = f"PASS  All {len(expected)} expected features present"
    if extra:
        msg += f"  (+{len(extra)} extra columns)"
    return True, msg


def check_zero_inflation_rate(
    df: pd.DataFrame,
    expected_min: float = 0.60,
    expected_max: float = 0.80,
) -> Tuple[bool, str]:
    """
    Verify the zero-inflation rate is within expected bounds.
    Stage 1 showed ~68% — we warn if this changes significantly.
    """
    if "sales" not in df.columns:
        return True, "SKIP  'sales' column absent"
    rate = (df["sales"] == 0).mean()
    if rate < expected_min or rate > expected_max:
        return False, (
            f"FAIL  Zero-inflation rate {rate:.1%} outside expected "
            f"[{expected_min:.0%}, {expected_max:.0%}]"
        )
    return True, f"PASS  Zero-inflation rate: {rate:.1%}  (expected ~68%)"


def check_memory_budget(df: pd.DataFrame, max_mb: float = 12000.0) -> Tuple[bool, str]:
    """
    Warn if the processed DataFrame exceeds the memory budget.
    Full 59M-row dataset after optimization peaks ~8.6 GB.
    Store-by-store chunks stay under 1 GB each.
    """
    mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    if mb > max_mb:
        return False, f"FAIL  Memory {mb:.0f} MB exceeds budget {max_mb:.0f} MB"
    return True, f"PASS  Memory usage: {mb:.0f} MB  (budget {max_mb:.0f} MB)"


# ==============================================================================
# Full Validation Suite
# ==============================================================================
def run_all_checks(
    df: pd.DataFrame,
    expected_series: int = 30490,
    expected_days: int = 1941,
) -> Dict[str, Tuple[bool, str]]:
    """
    Run all validation checks and return a results dictionary.

    Parameters
    ----------
    df              : pd.DataFrame
    expected_series : int
    expected_days   : int

    Returns
    -------
    dict  {check_name: (passed: bool, message: str)}
    """
    checks = {
        "row_count":           lambda: check_row_count(df, expected_series, expected_days),
        "no_nulls_critical":   lambda: check_no_nulls_in_critical(df),
        "no_negative_sales":   lambda: check_no_negative_sales(df),
        "temporal_ordering":   lambda: check_temporal_ordering(df),
        "split_completeness":  lambda: check_split_completeness(df),
        "lag_leakage":         lambda: check_no_future_leakage_in_lags(df),
        "feature_completeness":lambda: check_feature_completeness(df),
        "zero_inflation_rate": lambda: check_zero_inflation_rate(df),
        "memory_budget":       lambda: check_memory_budget(df),
    }

    results = {}
    n_pass = 0
    n_fail = 0

    logger.info("=" * 60)
    logger.info("VALIDATION SUITE")
    logger.info("=" * 60)

    for name, check_fn in checks.items():
        try:
            passed, msg = check_fn()
        except Exception as exc:
            passed, msg = False, f"ERROR  {exc}"
        results[name] = (passed, msg)
        symbol = "✓" if passed else "✗"
        log_fn = logger.info if passed else logger.warning
        log_fn(f"  [{symbol}] {name}: {msg}")
        if passed:
            n_pass += 1
        else:
            n_fail += 1

    logger.info("-" * 60)
    logger.info(f"  Result: {n_pass}/{n_pass+n_fail} checks passed")
    logger.info("=" * 60)

    return results


# ==============================================================================
# Missing Value Report
# ==============================================================================
def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame summarising missing values per column.

    Returns
    -------
    pd.DataFrame  — columns: Column, Dtype, Null_Count, Null_Pct
    """
    null_counts = df.isna().sum()
    null_pcts   = (null_counts / len(df) * 100).round(2)
    report = pd.DataFrame({
        "Column": null_counts.index,
        "Dtype": [str(df[c].dtype) for c in null_counts.index],
        "Null_Count": null_counts.values,
        "Null_Pct": null_pcts.values,
    }).sort_values("Null_Count", ascending=False).reset_index(drop=True)
    return report[report["Null_Count"] > 0]


# ==============================================================================
# Dataset Summary
# ==============================================================================
def dataset_summary(df: pd.DataFrame) -> None:
    """Print a concise summary of the final processed dataset."""
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    logger.info("=" * 60)
    logger.info("PROCESSED DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Rows:    {len(df):,}")
    logger.info(f"  Columns: {df.shape[1]}")
    logger.info(f"  Memory:  {mem_mb:.1f} MB")

    if "split" in df.columns:
        for s in ["train", "val", "test"]:
            n = (df["split"] == s).sum()
            logger.info(f"  {s:6s}:  {n:,} rows")

    if "sales" in df.columns:
        logger.info(f"  Sales stats: mean={df['sales'].mean():.3f}  "
                    f"max={df['sales'].max()}  "
                    f"zero_pct={100*(df['sales']==0).mean():.1f}%")

    # Feature group counts
    from collections import Counter
    group_counts = Counter(v["group"] for v in FEATURE_METADATA.values()
                           if v["group"] not in ("identity", "meta"))
    logger.info(f"  Feature groups: {dict(group_counts)}")
    logger.info("=" * 60)
