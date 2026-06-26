"""
==============================================================================
Memory Optimizer — Supply Chain Risk Triage  (Stage 2)
==============================================================================
Responsibility:
    Downcast DataFrame dtypes to the smallest safe representation, minimising
    RAM consumption while preserving all information.

Why this matters:
    - sales_train_evaluation (wide):   ~462 MB as int64
    - sell_prices:                     ~853 MB as object/float64
    After optimisation the merged long-format dataset must fit comfortably
    in memory when processed store-by-store.

Decisions documented inside each function.
==============================================================================
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.helpers import setup_logger, load_config

logger = setup_logger()


# ==============================================================================
# 1. Column-Level Downcasting
# ==============================================================================
def downcast_integers(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Downcast listed integer columns to the smallest safe signed integer type.

    Strategy: We use pd.to_numeric with downcast='integer'.  pandas will pick
    int8 / int16 / int32 / int64 automatically based on the actual value range.
    We do NOT blindly cast everything to int8 — we trust pandas to be safe.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
        Columns to attempt downcasting. Missing columns are silently skipped.

    Returns
    -------
    pd.DataFrame  (modified in-place, also returned for chaining)
    """
    for col in columns:
        if col not in df.columns:
            continue
        original_dtype = df[col].dtype
        df[col] = pd.to_numeric(df[col], downcast="integer")
        new_dtype = df[col].dtype
        if original_dtype != new_dtype:
            logger.debug(f"  [downcast_int] {col}: {original_dtype} → {new_dtype}")
    return df


def downcast_floats(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Downcast listed float columns from float64 → float32.

    Rationale: sell_price values range $0.01 – $107.32.
    float32 has ~7 significant decimal digits — more than enough for USD prices.
    Rolling/lag features are also safe in float32.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]

    Returns
    -------
    pd.DataFrame
    """
    for col in columns:
        if col not in df.columns:
            continue
        if pd.api.types.is_float_dtype(df[col]):
            original_dtype = df[col].dtype
            df[col] = df[col].astype("float32")
            logger.debug(f"  [downcast_float] {col}: {original_dtype} → float32")
    return df


def encode_categoricals(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Convert low-cardinality string/object columns to pandas Categorical.

    Why Categorical instead of label-encoding (int)?
    - Preserves human-readable labels for debugging and dashboards.
    - pandas stores categories in an index + int16 code array → large savings.
    - Later stages can use pd.get_dummies or sklearn LabelEncoder directly.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]

    Returns
    -------
    pd.DataFrame
    """
    for col in columns:
        if col not in df.columns:
            continue
        if df[col].dtype != "category":
            df[col] = df[col].astype("category")
            logger.debug(f"  [categorise] {col} → category "
                         f"({df[col].cat.categories.size} levels)")
    return df


# ==============================================================================
# 2. Full Optimisation Pass
# ==============================================================================
def optimise_memory(
    df: pd.DataFrame,
    config: Optional[Dict] = None,
    label: str = "DataFrame"
) -> Tuple[pd.DataFrame, float, float]:
    """
    Apply all memory optimisations in one pass and report savings.

    Optimisation order matters:
        1. Integers first (sales column must be int before we categorise others).
        2. Floats second.
        3. Categoricals last (some columns may have been created as object after
           integer operations).

    Parameters
    ----------
    df : pd.DataFrame
    config : dict, optional
        preprocessing.yaml['preprocessing'] section.  Loaded automatically
        if not supplied.
    label : str
        Display name for log messages.

    Returns
    -------
    (df_optimised, before_mb, after_mb)
    """
    if config is None:
        cfg = load_config("preprocessing.yaml")["preprocessing"]
    else:
        cfg = config

    before_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logger.info(f"[Memory] {label}: {before_mb:.1f} MB before optimisation")

    # Step 1 – integers
    int16_cols = [c for c in cfg.get("int16_columns", []) if c in df.columns]
    int8_cols  = [c for c in cfg.get("int8_columns",  []) if c in df.columns]
    df = downcast_integers(df, int16_cols + int8_cols)

    # Step 2 – floats
    float32_cols = [c for c in cfg.get("float32_columns", []) if c in df.columns]
    # Also downcast any float columns produced by feature engineering
    extra_float_cols = [
        c for c in df.columns
        if df[c].dtype == "float64"
        and c not in float32_cols
    ]
    df = downcast_floats(df, float32_cols + extra_float_cols)

    # Step 3 – categoricals
    cat_cols = [c for c in cfg.get("category_columns", []) if c in df.columns]
    df = encode_categoricals(df, cat_cols)

    after_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    reduction_pct = 100 * (1 - after_mb / before_mb)
    logger.info(
        f"[Memory] {label}: {after_mb:.1f} MB after optimisation "
        f"(↓ {reduction_pct:.1f}%)"
    )
    return df, before_mb, after_mb


# ==============================================================================
# 3. Dtype Report
# ==============================================================================
def dtype_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a per-column report of dtype, cardinality, and memory.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame  — columns: Column, Dtype, Unique_Values, Memory_MB
    """
    mem = df.memory_usage(deep=True)
    records = []
    for col in df.columns:
        records.append({
            "Column": col,
            "Dtype": str(df[col].dtype),
            "Unique_Values": df[col].nunique(),
            "Memory_MB": round(mem.get(col, 0) / (1024 ** 2), 4),
        })
    return pd.DataFrame(records)
