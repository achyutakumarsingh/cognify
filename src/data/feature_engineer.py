"""
==============================================================================
Feature Engineer — Supply Chain Risk Triage  (Stage 2)
==============================================================================
Responsibility:
    Transform the merged long-format DataFrame into a rich feature matrix
    ready for machine learning models.

Feature Groups (in order of computation):
    1. Date / Calendar features        (deterministic, no lag risk)
    2. Zero-inflation indicator        (demand intermittency flag)
    3. Price features                  (derived from sell_price)
    4. Event / Calendar event features (binary flags + categorical)
    5. SNAP interaction features       (SNAP × state match)
    6. Lag features                    (MUST be computed per series)
    7. Rolling statistics              (MUST be computed per series)

CRITICAL DESIGN RULE — Temporal Integrity:
    Lag and rolling features look into the PAST of each time series.
    They must ONLY be computed AFTER the data is sorted chronologically
    within each (item_id, store_id) group.
    Any leakage of future values would invalidate all forecasting results.
    We enforce this through explicit sorting before each transformation.
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


# ==============================================================================
# Helper: day-column → integer index
# ==============================================================================
def _d_to_int(d_col: pd.Series) -> pd.Series:
    """
    Convert 'd_1', 'd_7', … strings to integers 1, 7, …

    We store the numeric day index as `day_index` for sorting and
    for computing absolute position in the series (useful as a trend feature).
    """
    return d_col.str.replace("d_", "", regex=False).astype("int16")


# ==============================================================================
# 1. Date Features
# ==============================================================================
def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive standard calendar/date features from the `date` column.

    Features created:
        year         — Calendar year (2011-2016).  Captures long-term trend.
        month        — Month of year (1-12).  Captures monthly seasonality.
        week         — ISO week of year (1-52).  Captures weekly seasonality.
        day_of_month — Day within month (1-31).  Useful for end-of-month effects.
        day_of_week  — Day within week (0=Monday…6=Sunday per ISO).
        is_weekend   — Binary flag: 1 if Saturday or Sunday.
        quarter      — Quarter (1-4).  Captures seasonal macro patterns.
        day_index    — Integer day number (1-1941).  Encodes trend/recency.

    Why not use `wday` from calendar directly?
        `wday` in M5 is 1=Saturday, 7=Friday — non-standard.  We derive
        day_of_week from the datetime object to get ISO-standard 0=Monday.

    Parameters
    ----------
    df : pd.DataFrame  (must have 'date' column as datetime64)

    Returns
    -------
    pd.DataFrame
    """
    assert "date" in df.columns, "'date' column required for date features"
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    df["year"]         = df["date"].dt.year.astype("int16")
    df["month"]        = df["date"].dt.month.astype("int8")
    df["week"]         = df["date"].dt.isocalendar().week.astype("int8")
    df["day_of_month"] = df["date"].dt.day.astype("int8")
    df["day_of_week"]  = df["date"].dt.dayofweek.astype("int8")   # 0=Mon, 6=Sun
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype("int8")
    df["quarter"]      = df["date"].dt.quarter.astype("int8")
    df["day_index"]    = _d_to_int(df["d"])   # already int16

    logger.info("[Features] Date features added: year, month, week, day_of_month, "
                "day_of_week, is_weekend, quarter, day_index")
    return df


# ==============================================================================
# 2. Zero-Inflation / Intermittency Feature
# ==============================================================================
def add_zero_inflation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features that explicitly model demand intermittency.

    68% of M5 observations are zero (Stage 1).  Standard regression models
    treat zeros as just another value.  We expose the intermittency pattern
    as explicit features so tree models can learn:
    (a) whether demand is active at all, and
    (b) the historical intermittency rate of each series.

    Features created:
        is_zero_sales  — Binary: 1 if sales == 0 on this day.
                         Used as a diagnostic feature and for calibration.

    Note:
        We deliberately do NOT create look-ahead features here (e.g.,
        "fraction_zero_last_28_days") because those are rolling features
        that must be computed in the lag section to respect temporal order.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df["is_zero_sales"] = (df["sales"] == 0).astype("int8")
    logger.info("[Features] Zero-inflation indicator added: is_zero_sales")
    return df


# ==============================================================================
# 3. Price Features
# ==============================================================================
def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive price-change features from sell_price.

    Price dynamics are strong demand signals:
    - A price cut often triggers a short-term demand spike.
    - A large price increase can suppress demand.
    - Relative price vs. the item's own mean captures whether the current
      price is "cheap" or "expensive" for this product.

    Features created:
        price_lag1_within_series  — price from the previous available day
                                    (used internally, then dropped)
        price_diff                — sell_price − price of previous week
                                    (absolute change in USD)
        price_pct_change          — % change from previous week's price
        price_rel_to_mean         — sell_price / mean(sell_price for this item-store)
                                    Values < 1 → below-average price (likely promotion)
        has_price                 — 1 if sell_price > 0 (item is listed/active)

    IMPORTANT: price_diff and price_pct_change are computed per (item_id, store_id)
    group sorted by day_index to ensure we look at the *previous price for this series*,
    not the price from a different series.

    Parameters
    ----------
    df : pd.DataFrame  (must have 'sell_price' and 'day_index' columns)

    Returns
    -------
    pd.DataFrame
    """
    assert "sell_price" in df.columns, "'sell_price' column required"
    assert "day_index" in df.columns, "run add_date_features() before add_price_features()"

    # Sort within each series by day (essential for shift operations)
    df = df.sort_values(["item_id", "store_id", "day_index"])

    grp = df.groupby(["item_id", "store_id"], observed=True)["sell_price"]

    # Previous-day price (shift(1) within the series)
    prev_price = grp.transform(lambda s: s.shift(1))

    df["price_diff"]        = (df["sell_price"] - prev_price).astype("float32")
    df["price_pct_change"]  = (
        ((df["sell_price"] - prev_price) / (prev_price + 1e-6)) * 100
    ).astype("float32")

    # Relative price vs. series mean
    series_mean_price = grp.transform("mean")
    df["price_rel_to_mean"] = (df["sell_price"] / (series_mean_price + 1e-6)).astype("float32")

    # Is the item listed (has a positive price)?
    df["has_price"] = (df["sell_price"] > 0).astype("int8")

    # Fill NaN in price_diff / price_pct_change at series start
    df["price_diff"]       = df["price_diff"].fillna(0.0).astype("float32")
    df["price_pct_change"] = df["price_pct_change"].fillna(0.0).astype("float32")

    logger.info("[Features] Price features added: price_diff, price_pct_change, "
                "price_rel_to_mean, has_price")
    return df


# ==============================================================================
# 4. Event / Calendar Event Features
# ==============================================================================
def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode calendar events as model-ready features.

    The M5 calendar has two event slots (event_name_1, event_type_1,
    event_name_2, event_type_2).  After filling NaN → 'no_event' in the
    merge step, we create:

    Features created:
        has_event_1    — Binary: 1 if event_name_1 != 'no_event'
        has_event_2    — Binary: 1 if event_name_2 != 'no_event'
        has_any_event  — Binary: 1 if any event today
        is_sporting    — Binary: event type is Sporting
        is_cultural    — Binary: event type is Cultural
        is_national    — Binary: event type is National
        is_religious   — Binary: event type is Religious

    Why binary flags instead of label-encoding event names?
        There are 162 primary events with many unique names.  One-hot
        encoding would create 162 sparse columns.  Grouped type flags
        (Sporting/Cultural/National/Religious) capture the systematic
        demand patterns (e.g., sporting events always lift FOODS, etc.)
        without exploding dimensionality.  Full event names are preserved
        as category columns for optional one-hot encoding later.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    no_event = "no_event"

    if "event_name_1" in df.columns:
        df["has_event_1"] = (df["event_name_1"] != no_event).astype("int8")
    if "event_name_2" in df.columns:
        df["has_event_2"] = (df["event_name_2"] != no_event).astype("int8")

    df["has_any_event"] = (
        df.get("has_event_1", pd.Series(0, index=df.index))
        | df.get("has_event_2", pd.Series(0, index=df.index))
    ).astype("int8")

    # Event type flags (check both event slots)
    for etype in ["Sporting", "Cultural", "National", "Religious"]:
        col_name = f"is_{etype.lower()}"
        flag = pd.Series(0, index=df.index, dtype="int8")
        for et_col in ["event_type_1", "event_type_2"]:
            if et_col in df.columns:
                flag = flag | (df[et_col] == etype).astype("int8")
        df[col_name] = flag

    logger.info("[Features] Event features added: has_event_1/2, has_any_event, "
                "is_sporting, is_cultural, is_national, is_religious")
    return df


# ==============================================================================
# 5. SNAP Features
# ==============================================================================
def add_snap_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create SNAP interaction features.

    SNAP (Supplemental Nutrition Assistance Program) benefits are issued on
    specific days per state.  When SNAP is active, FOODS sales spike
    significantly.  The direct snap_CA/TX/WI flags from calendar tell us
    whether SNAP is active in a given state on a given day.  But each
    series is tied to a specific store in a specific state — so the
    relevant flag is the one for *that series's state*.

    Feature created:
        snap_active  — 1 if SNAP is active in THIS series's state on this day.
                       (Matches snap_CA/TX/WI to state_id column.)

    Why not use all three raw snap columns?
        Including all three for every series would create multicollinearity and
        would be factually wrong (a CA store is unaffected by TX SNAP days).
        snap_active is the correct, series-specific interaction.

    Parameters
    ----------
    df : pd.DataFrame  (must have state_id, snap_CA, snap_TX, snap_WI)

    Returns
    -------
    pd.DataFrame
    """
    state_to_snap = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}

    # Build snap_active using np.select to avoid pandas 3.0 dtype-strict
    # boolean-mask assignment issues.
    conditions = []
    values = []
    for state, col in state_to_snap.items():
        if col in df.columns:
            conditions.append(df["state_id"] == state)
            values.append(df[col].astype("int8"))

    if conditions:
        snap_active = np.zeros(len(df), dtype="int8")
        for cond, val in zip(conditions, values):
            snap_active = np.where(cond.values, val.values, snap_active).astype("int8")
        df["snap_active"] = snap_active
    else:
        df["snap_active"] = np.int8(0)

    logger.info("[Features] SNAP feature added: snap_active (state-matched)")
    return df


# ==============================================================================
# 6. Lag Features
# ==============================================================================
def add_lag_features(
    df: pd.DataFrame,
    lag_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Add lagged sales values as features.

    WHY LAGS?
        In time-series forecasting, the most predictive features are often
        past values of the target itself.  The specific lag windows are chosen
        to capture:
          lag_7  → Same day of week 1 week ago (strong weekly seasonality)
          lag_14 → Same day of week 2 weeks ago
          lag_28 → Same day of week 4 weeks ago  (M5 standard benchmark lag)

    TEMPORAL INTEGRITY GUARANTEE:
        `shift(n)` moves values DOWN by n positions within each group.
        Row for day d gets the value from day d-n.  This is safe because
        we are only looking at past values — no future data leakage.

        The first `n` days of each series will have NaN for lag_n.
        These are filled with 0 (neutral / "assume no recent demand").
        Alternative: fill with the series mean.  We use 0 because it makes
        no assumption about historical patterns for new series.

    Parameters
    ----------
    df          : pd.DataFrame  (must be sorted by item_id, store_id, day_index)
    lag_windows : list[int]  (default from preprocessing.yaml)

    Returns
    -------
    pd.DataFrame
    """
    if lag_windows is None:
        cfg = load_config("preprocessing.yaml")["preprocessing"]
        lag_windows = cfg.get("lag_windows", [7, 14, 28])

    # Ensure temporal ordering within each series
    df = df.sort_values(["item_id", "store_id", "day_index"])

    grp = df.groupby(["item_id", "store_id"], observed=True)["sales"]

    for lag in lag_windows:
        col_name = f"lag_{lag}"
        df[col_name] = grp.transform(lambda s: s.shift(lag)).fillna(0).astype("float32")
        logger.debug(f"[Features] Added lag_{lag}")

    logger.info(f"[Features] Lag features added: {['lag_'+str(w) for w in lag_windows]}")
    return df


# ==============================================================================
# 7. Rolling Statistics
# ==============================================================================
def add_rolling_features(
    df: pd.DataFrame,
    rolling_windows: Optional[List[int]] = None,
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Add rolling mean and rolling standard deviation features.

    WHY ROLLING STATS?
        Rolling statistics smooth out day-to-day noise and capture the
        local trend / volatility of each time series.

        rolling_mean_7  → Short-term demand level  (weekly average)
        rolling_mean_14 → Medium-term demand level
        rolling_mean_28 → Long-term demand level    (monthly average)

        rolling_std_7   → Short-term demand volatility
        rolling_std_28  → Long-term demand volatility

        Volatility (std) is especially important for our project because
        Stage 4 will use it as a prior on uncertainty:
        high rolling_std → wider prediction intervals are expected.

    SHIFT-BEFORE-ROLLING (preventing leakage):
        We shift(1) BEFORE applying the rolling window.
        Without this shift, rolling_mean_7 on day d would include day d's
        own sales value → data leakage.
        With shift(1), the window covers [d-7, d-1] — purely historical.

    min_periods=1:
        Allow the rolling window to produce values even when the series is
        shorter than the window (first few days).  This prevents excessive
        NaN at series starts.

    Implementation note (pandas 3.0 compatible):
        We use groupby(...).transform() on the ALREADY-shifted series.
        To avoid a nested groupby issue, we create a composite group key
        as a new column, transform on that, then drop it.

    Parameters
    ----------
    df             : pd.DataFrame
    rolling_windows: list[int]
    min_periods    : int

    Returns
    -------
    pd.DataFrame
    """
    if rolling_windows is None:
        cfg = load_config("preprocessing.yaml")["preprocessing"]
        rolling_windows = cfg.get("rolling_windows", [7, 14, 28])
        min_periods = cfg.get("rolling_min_periods", 1)

    df = df.sort_values(["item_id", "store_id", "day_index"])

    # Create a composite group key for rolling computations
    df["_series_key"] = df["item_id"].astype(str) + "_" + df["store_id"].astype(str)

    # Shift sales by 1 within each series to prevent same-day leakage
    df["_sales_shifted"] = (
        df.groupby("_series_key", observed=True)["sales"]
        .transform(lambda s: s.shift(1))
    )

    for window in rolling_windows:
        # Rolling mean
        mean_col = f"rolling_mean_{window}"
        df[mean_col] = (
            df.groupby("_series_key", observed=True)["_sales_shifted"]
            .transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
            .fillna(0.0)
            .astype("float32")
        )

        # Rolling std (only for windows 7 and 28 to keep feature count manageable)
        if window in [7, 28]:
            std_col = f"rolling_std_{window}"
            df[std_col] = (
                df.groupby("_series_key", observed=True)["_sales_shifted"]
                .transform(lambda s: s.rolling(window, min_periods=min_periods).std())
                .fillna(0.0)
                .astype("float32")
            )
            logger.debug(f"[Features] Added {mean_col} + {std_col}")
        else:
            logger.debug(f"[Features] Added {mean_col}")

    # Clean up temporary columns
    df.drop(columns=["_series_key", "_sales_shifted"], inplace=True)

    mean_names = [f"rolling_mean_{w}" for w in rolling_windows]
    std_names  = [f"rolling_std_{w}" for w in [7, 28]]
    logger.info(f"[Features] Rolling features added: {mean_names + std_names}")
    return df


# ==============================================================================
# 8. Temporal Split Column
# ==============================================================================
def add_split_column(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Add a `split` column marking each row as 'train', 'val', or 'test'.

    Split boundaries (from preprocessing.yaml):
        train : day_index ≤ 1885
        val   : 1886 ≤ day_index ≤ 1913
        test  : 1914 ≤ day_index ≤ 1941

    Why 28-day val/test windows?
        The M5 competition requires 28-day ahead forecasts.  Evaluating on
        a 28-day window matches the real forecasting task exactly.

    WHY NOT RANDOM SPLIT?
        Time-series data has temporal dependencies.  Random splitting
        would cause future data to leak into training, inflating metrics.
        Walk-forward (temporal) split is the only valid strategy.

    Parameters
    ----------
    df     : pd.DataFrame
    config : dict  (preprocessing.yaml['preprocessing'])

    Returns
    -------
    pd.DataFrame
    """
    if config is None:
        config = load_config("preprocessing.yaml")["preprocessing"]

    train_end = config["train_end_day"]   # 1885
    val_end   = config["val_end_day"]     # 1913
    test_end  = config["test_end_day"]    # 1941

    assert "day_index" in df.columns, "run add_date_features() before add_split_column()"

    conditions = [
        df["day_index"] <= train_end,
        (df["day_index"] > train_end) & (df["day_index"] <= val_end),
        df["day_index"] > val_end,
    ]
    choices = ["train", "val", "test"]
    df["split"] = pd.Categorical(
        np.select(conditions, choices, default="train"),
        categories=["train", "val", "test"]
    )

    split_counts = df["split"].value_counts().to_dict()
    logger.info(f"[Features] Temporal split added: {split_counts}")
    return df


# ==============================================================================
# 9. Feature Metadata Catalogue
# ==============================================================================
FEATURE_METADATA = {
    # --- Identity ---
    "id":             {"group": "identity",  "dtype": "category",  "desc": "Unique series ID (item_id + store_id)"},
    "item_id":        {"group": "identity",  "dtype": "category",  "desc": "Product identifier"},
    "dept_id":        {"group": "identity",  "dtype": "category",  "desc": "Department (7 levels)"},
    "cat_id":         {"group": "identity",  "dtype": "category",  "desc": "Category (3 levels)"},
    "store_id":       {"group": "identity",  "dtype": "category",  "desc": "Store identifier (10 stores)"},
    "state_id":       {"group": "identity",  "dtype": "category",  "desc": "State (CA / TX / WI)"},
    "d":              {"group": "identity",  "dtype": "str",       "desc": "Day string identifier (d_1 ... d_1941)"},
    "date":           {"group": "identity",  "dtype": "datetime",  "desc": "Calendar date"},
    "wm_yr_wk":       {"group": "identity",  "dtype": "int",       "desc": "Walmart fiscal year-week"},
    # --- Target ---
    "sales":          {"group": "target",    "dtype": "int16",     "desc": "Daily unit sales (TARGET VARIABLE)"},
    # --- Date features ---
    "year":           {"group": "date",      "dtype": "int16",     "desc": "Calendar year (long-term trend)"},
    "month":          {"group": "date",      "dtype": "int8",      "desc": "Month of year (seasonal)"},
    "week":           {"group": "date",      "dtype": "int8",      "desc": "ISO week of year"},
    "day_of_month":   {"group": "date",      "dtype": "int8",      "desc": "Day within month"},
    "day_of_week":    {"group": "date",      "dtype": "int8",      "desc": "Day of week (0=Mon, 6=Sun)"},
    "is_weekend":     {"group": "date",      "dtype": "int8",      "desc": "Weekend flag (Sat/Sun)"},
    "quarter":        {"group": "date",      "dtype": "int8",      "desc": "Calendar quarter"},
    "day_index":      {"group": "date",      "dtype": "int16",     "desc": "Absolute day number (1-1941, trend proxy)"},
    "wday":           {"group": "date",      "dtype": "int8",      "desc": "M5 weekday (1=Sat per Walmart convention)"},
    "weekday":        {"group": "date",      "dtype": "category",  "desc": "Weekday name (string)"},
    # --- Zero-inflation ---
    "is_zero_sales":  {"group": "zero_inflation", "dtype": "int8", "desc": "1 if sales==0 (intermittency indicator)"},
    # --- Price features ---
    "sell_price":          {"group": "price", "dtype": "float32", "desc": "Weekly sell price (USD)"},
    "price_diff":          {"group": "price", "dtype": "float32", "desc": "Price change from previous week (USD)"},
    "price_pct_change":    {"group": "price", "dtype": "float32", "desc": "Percentage price change from previous week"},
    "price_rel_to_mean":   {"group": "price", "dtype": "float32", "desc": "Current price / series mean price (promotion indicator)"},
    "has_price":           {"group": "price", "dtype": "int8",    "desc": "1 if sell_price > 0 (item is active/listed)"},
    # --- Event features ---
    "event_name_1":    {"group": "event", "dtype": "category", "desc": "Primary event name (or 'no_event')"},
    "event_type_1":    {"group": "event", "dtype": "category", "desc": "Primary event type"},
    "event_name_2":    {"group": "event", "dtype": "category", "desc": "Secondary event name"},
    "event_type_2":    {"group": "event", "dtype": "category", "desc": "Secondary event type"},
    "has_event_1":     {"group": "event", "dtype": "int8",     "desc": "1 if any primary event today"},
    "has_event_2":     {"group": "event", "dtype": "int8",     "desc": "1 if any secondary event today"},
    "has_any_event":   {"group": "event", "dtype": "int8",     "desc": "1 if any event (primary or secondary)"},
    "is_sporting":     {"group": "event", "dtype": "int8",     "desc": "1 if event type is Sporting"},
    "is_cultural":     {"group": "event", "dtype": "int8",     "desc": "1 if event type is Cultural"},
    "is_national":     {"group": "event", "dtype": "int8",     "desc": "1 if event type is National"},
    "is_religious":    {"group": "event", "dtype": "int8",     "desc": "1 if event type is Religious"},
    # --- SNAP features ---
    "snap_CA":         {"group": "snap", "dtype": "int8", "desc": "SNAP active in California"},
    "snap_TX":         {"group": "snap", "dtype": "int8", "desc": "SNAP active in Texas"},
    "snap_WI":         {"group": "snap", "dtype": "int8", "desc": "SNAP active in Wisconsin"},
    "snap_active":     {"group": "snap", "dtype": "int8", "desc": "SNAP active for THIS series's state (interaction feature)"},
    # --- Lag features ---
    "lag_7":           {"group": "lag",     "dtype": "float32", "desc": "Sales 7 days ago (same day-of-week, weekly cycle)"},
    "lag_14":          {"group": "lag",     "dtype": "float32", "desc": "Sales 14 days ago"},
    "lag_28":          {"group": "lag",     "dtype": "float32", "desc": "Sales 28 days ago (M5 standard lag)"},
    # --- Rolling features ---
    "rolling_mean_7":  {"group": "rolling", "dtype": "float32", "desc": "7-day rolling mean sales (short-term trend)"},
    "rolling_mean_14": {"group": "rolling", "dtype": "float32", "desc": "14-day rolling mean sales"},
    "rolling_mean_28": {"group": "rolling", "dtype": "float32", "desc": "28-day rolling mean sales (monthly trend)"},
    "rolling_std_7":   {"group": "rolling", "dtype": "float32", "desc": "7-day rolling std (short-term volatility → uncertainty prior)"},
    "rolling_std_28":  {"group": "rolling", "dtype": "float32", "desc": "28-day rolling std (long-term volatility → uncertainty prior)"},
    # --- Split ---
    "split":           {"group": "meta",    "dtype": "category", "desc": "Data split: train / val / test (temporal)"},
}


def get_feature_columns(exclude_groups: Optional[List[str]] = None) -> List[str]:
    """
    Return the list of feature column names for modelling.

    Parameters
    ----------
    exclude_groups : list[str], optional
        Groups to exclude (e.g., ['identity', 'target', 'meta']).

    Returns
    -------
    list[str]
    """
    excl = set(exclude_groups or ["identity", "target", "meta"])
    return [col for col, meta in FEATURE_METADATA.items()
            if meta["group"] not in excl]
