"""
==============================================================================
Data Merger — Supply Chain Risk Triage  (Stage 2)
==============================================================================
Responsibility:
    Merge the three M5 datasets into a single long-format analytic table
    that is ready for feature engineering.

Merge Strategy  (see docstring of `build_long_format`):
    Step 1 — MELT  :  sales (wide)  →  long  (one row per item-store-day)
    Step 2 — JOIN  :  + calendar    via   `d`
    Step 3 — JOIN  :  + sell_prices via   (store_id, item_id, wm_yr_wk)

Why long-format?
    - Feature engineering (lags, rolling stats) requires temporal ordering
      of rows *within each time series*.  Long format makes this natural.
    - Every ML framework expects tidy tabular data (one sample per row).
    - Wide format would require transposing anyway — better to do it once.

Why store-by-store processing?
    Full melt of all 30,490 series × 1,941 days = ~59 M rows.
    Doing this at once peaks at ~6-8 GB RAM.  Processing one store at a
    time reduces peak usage to ~600 MB per batch, then we append/stream
    to Parquet.  The 10 stores are independent — no information leaks.
==============================================================================
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.helpers import setup_logger, load_config

logger = setup_logger()

# Metadata columns in the sales wide-format DataFrame
SALES_META_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


# ==============================================================================
# 1. Melt (Wide → Long)
# ==============================================================================
def melt_sales(sales: pd.DataFrame, store_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert the sales DataFrame from wide format to long (tidy) format.

    Wide:  30,490 rows  × 1,947 columns  (6 metadata + 1,941 day columns)
    Long:  ~59.2 M rows × 8 columns

    Parameters
    ----------
    sales : pd.DataFrame
        Raw sales DataFrame as loaded from sales_train_evaluation.csv.
    store_ids : list[str], optional
        If provided, only melt rows for the given stores.  Used by the
        store-by-store processing mode to control memory.

    Returns
    -------
    pd.DataFrame
        Columns: id, item_id, dept_id, cat_id, store_id, state_id, d, sales
    """
    if store_ids is not None:
        sales = sales[sales["store_id"].isin(store_ids)].copy()
        logger.info(f"[Melt] Filtered to stores {store_ids}: {len(sales):,} series")

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    logger.info(f"[Melt] Melting {len(sales):,} series × {len(day_cols)} days "
                f"→ expected {len(sales) * len(day_cols):,} rows")

    long_df = sales.melt(
        id_vars=SALES_META_COLS,
        value_vars=day_cols,
        var_name="d",           # e.g. "d_1", "d_7", …
        value_name="sales",
    )

    logger.info(f"[Melt] Result: {len(long_df):,} rows × {long_df.shape[1]} cols")
    return long_df


# ==============================================================================
# 2. Join Calendar
# ==============================================================================
def join_calendar(long_df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join the long sales DataFrame with calendar on the `d` column.

    Why LEFT join?
        Every `d` value in the sales data must have a matching calendar row.
        A left join preserves all sales rows and will show NaN for any `d`
        without a calendar entry (which we validate for).

    Columns added from calendar:
        date, wm_yr_wk, weekday, wday, month, year,
        event_name_1, event_type_1, event_name_2, event_type_2,
        snap_CA, snap_TX, snap_WI

    Parameters
    ----------
    long_df   : pd.DataFrame  (long-format sales)
    calendar  : pd.DataFrame  (raw calendar table)

    Returns
    -------
    pd.DataFrame
    """
    # Drop 'date' separately — we want date as a proper datetime column
    # keep only the columns we need from calendar (drop redundant ones)
    cal_cols = [
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]
    cal_subset = calendar[cal_cols].copy()

    before = len(long_df)
    merged = long_df.merge(cal_subset, on="d", how="left")
    after = len(merged)

    if before != after:
        logger.warning(
            f"[Calendar join] Row count changed: {before:,} → {after:,}. "
            f"Investigate duplicates in calendar.d!"
        )
    else:
        logger.info(f"[Calendar join] ✓ {after:,} rows preserved")

    # Validate: no NaN in key calendar columns
    n_missing = merged["date"].isna().sum()
    if n_missing > 0:
        logger.warning(f"[Calendar join] {n_missing:,} rows have no calendar match on 'd'")

    return merged


# ==============================================================================
# 3. Join Sell Prices
# ==============================================================================
def join_sell_prices(merged: pd.DataFrame, sell_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join the merged DataFrame with sell_prices on (store_id, item_id, wm_yr_wk).

    Why LEFT join (not inner)?
        Some items have no price in certain weeks (not yet on shelf, or delisted).
        An inner join would silently drop those rows, biasing the training set.
        We keep all rows and handle the missing prices in cleaning.

    Why weekly price (wm_yr_wk) instead of daily?
        Walmart only updates prices at the fiscal-week level.  Prices are
        constant within a week, so joining on week is exact and correct.

    Parameters
    ----------
    merged      : pd.DataFrame  (long sales + calendar)
    sell_prices : pd.DataFrame  (raw sell_prices table)

    Returns
    -------
    pd.DataFrame
    """
    before = len(merged)
    result = merged.merge(
        sell_prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )
    after = len(result)

    if before != after:
        logger.warning(
            f"[Price join] Row count changed: {before:,} → {after:,}. "
            f"Possible duplicate (store_id, item_id, wm_yr_wk) in sell_prices!"
        )
    else:
        logger.info(f"[Price join] ✓ {after:,} rows preserved")

    n_missing_price = result["sell_price"].isna().sum()
    pct = 100 * n_missing_price / after
    logger.info(
        f"[Price join] Missing sell_price: {n_missing_price:,} rows ({pct:.1f}%)"
    )

    return result


# ==============================================================================
# 4. Handle Missing Values after Merge
# ==============================================================================
def fill_missing_values(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Impute missing values introduced by the LEFT joins.

    Decisions:
        event columns  →  fill NaN with 'no_event' (string sentinel).
                          Reason: NaN in event_name means no event occurred,
                          not that data is missing.  Using a sentinel lets
                          tree models split on "event vs no_event" directly.

        sell_price     →  forward-fill within each (item_id, store_id) group,
                          then backward-fill to handle leading NaN.
                          Reason: price is a slow-changing variable.  The most
                          recent known price is the best proxy for an unknown
                          week (e.g., item newly listed but price not yet in db).
                          Remaining NaN (no price ever) → fill with 0.0 as a
                          sentinel (downstream models can learn this pattern).

    Parameters
    ----------
    df     : pd.DataFrame
    config : dict, optional  (preprocessing.yaml['preprocessing'])

    Returns
    -------
    pd.DataFrame
    """
    if config is None:
        config = load_config("preprocessing.yaml")["preprocessing"]

    fill_val = config.get("event_fill_value", "no_event")

    # --- Event columns ---
    for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        if col in df.columns:
            df[col] = df[col].fillna(fill_val)

    # --- Sell price: ffill within series, then bfill, then 0 ---
    if "sell_price" in df.columns:
        df = df.sort_values(["item_id", "store_id", "d"])
        df["sell_price"] = (
            df.groupby(["item_id", "store_id"], observed=True)["sell_price"]
            .transform(lambda s: s.ffill().bfill())
        )
        remaining = df["sell_price"].isna().sum()
        if remaining > 0:
            logger.info(
                f"[Fill] {remaining:,} sell_price still NaN after ffill/bfill "
                f"(items with no price history) → filling with 0.0"
            )
            df["sell_price"] = df["sell_price"].fillna(0.0)

    logger.info("[Fill] Missing value imputation complete")
    return df


# ==============================================================================
# 5. Merge Validation
# ==============================================================================
def validate_merge(
    df: pd.DataFrame,
    expected_series: int = 30490,
    expected_days: int = 1941,
) -> bool:
    """
    Sanity-check the merged long-format DataFrame.

    Checks:
        1. Row count ≈ expected_series × expected_days
        2. No NaN in critical columns (sales, item_id, store_id, date, d)
        3. Sales column is non-negative

    Parameters
    ----------
    df              : pd.DataFrame
    expected_series : int  (default 30,490 — full M5)
    expected_days   : int  (default 1,941 — evaluation dataset)

    Returns
    -------
    bool  — True if all checks pass
    """
    logger.info("[Validate merge] Running post-merge validation checks …")
    ok = True
    expected_rows = expected_series * expected_days

    # Check 1: row count
    actual_rows = len(df)
    tol = 0.01  # allow ±1% due to store-based partial runs
    if abs(actual_rows - expected_rows) / expected_rows > tol:
        logger.warning(
            f"[Validate] Row count {actual_rows:,} differs from expected "
            f"{expected_rows:,} by >{tol*100:.0f}%"
        )
        ok = False
    else:
        logger.info(f"[Validate] ✓ Row count: {actual_rows:,}")

    # Check 2: no NaN in critical columns
    critical = ["sales", "item_id", "store_id", "d"]
    for col in critical:
        if col in df.columns:
            n_null = df[col].isna().sum()
            if n_null > 0:
                logger.warning(f"[Validate] ✗ {col} has {n_null:,} NaN values")
                ok = False
            else:
                logger.info(f"[Validate] ✓ {col}: no NaN")

    # Check 3: sales non-negative
    if "sales" in df.columns:
        n_neg = (df["sales"] < 0).sum()
        if n_neg > 0:
            logger.warning(f"[Validate] ✗ {n_neg:,} negative sales values found")
            ok = False
        else:
            logger.info("[Validate] ✓ All sales values ≥ 0")

    return ok
