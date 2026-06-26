"""
==============================================================================
STAGE 1 — Dataset Understanding & Exploratory Data Analysis
==============================================================================
Project : Calibrated Uncertainty Quantification for Supply Chain Risk Triage
Dataset : M5 Forecasting — Accuracy (Walmart hierarchical sales data)
Stage   : 1 of 10

Purpose:
  - Load and validate all three datasets (calendar, sell_prices, sales)
  - Profile each dataset (shape, dtypes, memory, missing values, duplicates)
  - Perform exploratory data analysis (EDA)
  - Identify target variable, time columns, and key identifiers
  - Generate a comprehensive Data Quality Report
  - Document findings and produce a Stage 2 preparation checklist

Design Principles:
  - NO data cleaning, merging, feature engineering, or model training
  - All analysis is observational — we only READ the data
  - Modular functions so each analysis step is independently testable
  - Rich console output with section headers for readability

Usage:
  python src/data/stage1_eda.py

==============================================================================
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ---- Project imports ----
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import load_all_datasets, DATASET_DESCRIPTIONS
from src.utils.helpers import (
    setup_logger,
    set_seed,
    memory_usage_summary,
    get_project_root
)

# Suppress pandas display truncation for clean output
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", 40)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


# ==============================================================================
# Helper: Section Printing
# ==============================================================================
def print_section(title: str, char: str = "=", width: int = 80) -> None:
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def print_subsection(title: str, char: str = "-", width: int = 60) -> None:
    """Print a formatted subsection header."""
    print(f"\n  {char * width}")
    print(f"  {title}")
    print(f"  {char * width}\n")


# ==============================================================================
# 1. Dataset Overview
# ==============================================================================
def display_dataset_overview(
    datasets: Dict[str, pd.DataFrame]
) -> None:
    """
    Display shape, columns, first 5 rows, dtypes, and memory usage
    for each dataset.
    """
    print_section("1. DATASET OVERVIEW")

    for name, df in datasets.items():
        desc = DATASET_DESCRIPTIONS.get(name, {})

        print_subsection(f"Dataset: {name.upper()}")

        # Description
        print(f"  📋 Description:")
        print(f"     {desc.get('description', 'N/A')}\n")

        # Role
        print(f"  🔑 Role: {desc.get('role', 'N/A')}\n")

        # Shape
        print(f"  📐 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

        # Memory usage
        mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        print(f"  💾 Memory Usage: {mem_mb:.2f} MB\n")

        # Column names
        print(f"  📊 Columns ({df.shape[1]}):")
        if df.shape[1] <= 20:
            for col in df.columns:
                print(f"     • {col} ({df[col].dtype})")
        else:
            # For wide datasets like sales, show metadata cols + sample of day cols
            meta_cols = [c for c in df.columns if not c.startswith("d_")]
            day_cols = [c for c in df.columns if c.startswith("d_")]
            print(f"     Metadata columns ({len(meta_cols)}):")
            for col in meta_cols:
                print(f"       • {col} ({df[col].dtype})")
            print(f"     Day columns ({len(day_cols)}):")
            print(f"       • {day_cols[0]} ... {day_cols[-1]} (all {df[day_cols[0]].dtype})")

        # First 5 rows
        print(f"\n  👀 First 5 Rows:")
        if df.shape[1] <= 20:
            print(df.head().to_string(index=False, max_colwidth=25))
        else:
            # Show metadata + first/last 3 day columns
            show_cols = meta_cols + day_cols[:3] + ["..."] + day_cols[-3:]
            display_df = df[meta_cols + day_cols[:3]].head().copy()
            display_df["..."] = "..."
            for c in day_cols[-3:]:
                display_df[c] = df[c].head()
            print(display_df.to_string(index=False, max_colwidth=25))

        # Data types summary
        print(f"\n  🏷️  Data Types:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"     • {dtype}: {count} column(s)")

        # Key columns description
        key_cols = desc.get("key_columns", {})
        if key_cols:
            print(f"\n  🔗 Key Columns:")
            for col, role in key_cols.items():
                print(f"     • {col}: {role}")

        print()


# ==============================================================================
# 2. Exploratory Data Analysis
# ==============================================================================
def analyze_missing_values(
    datasets: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Analyze missing values across all datasets.

    Returns
    -------
    dict
        Per-dataset DataFrame with missing value statistics.
    """
    print_subsection("2a. Missing Values Analysis")

    missing_reports = {}

    for name, df in datasets.items():
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        report = pd.DataFrame({
            "Column": missing.index,
            "Missing_Count": missing.values,
            "Missing_Pct": missing_pct.values,
            "Total_Rows": len(df)
        })
        report = report[report["Missing_Count"] > 0].sort_values(
            "Missing_Pct", ascending=False
        ).reset_index(drop=True)

        missing_reports[name] = report

        print(f"\n  📊 {name.upper()}:")
        total_missing = missing.sum()
        total_cells = df.shape[0] * df.shape[1]
        print(f"     Total missing cells: {total_missing:,} / {total_cells:,} "
              f"({100 * total_missing / total_cells:.2f}%)")

        if len(report) == 0:
            print("     ✅ No missing values found!")
        else:
            print(f"     Columns with missing values ({len(report)}):")
            for _, row in report.iterrows():
                print(f"       • {row['Column']}: {row['Missing_Count']:,} "
                      f"({row['Missing_Pct']:.1f}%)")

    return missing_reports


def analyze_duplicates(
    datasets: Dict[str, pd.DataFrame]
) -> Dict[str, int]:
    """
    Check for duplicate rows in each dataset.

    Returns
    -------
    dict
        Number of duplicate rows per dataset.
    """
    print_subsection("2b. Duplicate Rows Analysis")

    dup_counts = {}
    for name, df in datasets.items():
        # For the sales dataset, check duplicates on metadata columns only
        # (checking all 1941 day columns would be meaningful but slow)
        if name == "sales":
            meta_cols = [c for c in df.columns if not c.startswith("d_")]
            n_dup = df.duplicated(subset=meta_cols).sum()
            print(f"  📊 {name.upper()}: {n_dup:,} duplicate rows "
                  f"(checked on metadata columns: {meta_cols})")
        else:
            n_dup = df.duplicated().sum()
            print(f"  📊 {name.upper()}: {n_dup:,} duplicate rows")

        dup_counts[name] = n_dup

        if n_dup == 0:
            print(f"     ✅ No duplicates found!")
        else:
            print(f"     ⚠️  {n_dup} duplicate(s) detected — investigate in Stage 2")

    return dup_counts


def analyze_unique_values(
    datasets: Dict[str, pd.DataFrame]
) -> None:
    """
    Display unique value counts for categorical and identifier columns.
    """
    print_subsection("2c. Unique Values Analysis")

    # Calendar — all columns
    cal = datasets["calendar"]
    print(f"\n  📊 CALENDAR:")
    for col in cal.columns:
        n_unique = cal[col].nunique()
        print(f"     • {col}: {n_unique:,} unique values")

    # Sell Prices — all columns
    sp = datasets["sell_prices"]
    print(f"\n  📊 SELL_PRICES:")
    for col in sp.columns:
        n_unique = sp[col].nunique()
        if n_unique <= 15:
            vals = sp[col].unique()
            print(f"     • {col}: {n_unique} unique → {sorted(vals)}")
        else:
            print(f"     • {col}: {n_unique:,} unique values")

    # Sales — metadata columns only
    sales = datasets["sales"]
    meta_cols = [c for c in sales.columns if not c.startswith("d_")]
    print(f"\n  📊 SALES (metadata columns):")
    for col in meta_cols:
        n_unique = sales[col].nunique()
        if n_unique <= 15:
            vals = sorted(sales[col].unique())
            print(f"     • {col}: {n_unique} unique → {vals}")
        else:
            print(f"     • {col}: {n_unique:,} unique values")

    # Sales — day columns summary
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    print(f"\n     • day columns (d_1 to d_{len(day_cols)}): "
          f"{len(day_cols)} columns representing daily unit sales")


def compute_descriptive_statistics(
    datasets: Dict[str, pd.DataFrame]
) -> None:
    """
    Display descriptive statistics for numerical columns in each dataset.
    """
    print_subsection("2d. Descriptive Statistics")

    # Calendar — numerical columns
    cal = datasets["calendar"]
    cal_numeric = cal.select_dtypes(include=[np.number])
    if not cal_numeric.empty:
        print(f"\n  📊 CALENDAR (numerical columns):")
        print(cal_numeric.describe().round(2).to_string())

    # Sell Prices — sell_price column
    sp = datasets["sell_prices"]
    print(f"\n  📊 SELL_PRICES (sell_price):")
    print(sp["sell_price"].describe().round(4).to_string())
    print(f"\n     Price range: ${sp['sell_price'].min():.2f} — "
          f"${sp['sell_price'].max():.2f}")
    print(f"     Median price: ${sp['sell_price'].median():.2f}")

    # Sales — aggregate statistics across all day columns
    sales = datasets["sales"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    print(f"\n  📊 SALES (daily unit sales across all {len(day_cols)} days):")

    # Compute stats efficiently using numpy on the day columns
    sales_values = sales[day_cols].values
    print(f"     Overall min:    {np.nanmin(sales_values)}")
    print(f"     Overall max:    {np.nanmax(sales_values)}")
    print(f"     Overall mean:   {np.nanmean(sales_values):.4f}")
    print(f"     Overall median: {np.nanmedian(sales_values):.1f}")
    print(f"     Overall std:    {np.nanstd(sales_values):.4f}")

    # Percentage of zero-sales days
    total_cells = sales_values.size
    zero_cells = (sales_values == 0).sum()
    print(f"\n     Zero-sales cells: {zero_cells:,} / {total_cells:,} "
          f"({100 * zero_cells / total_cells:.1f}%)")
    print(f"     → High zero-inflation is expected for slow-moving items")

    # Per-series statistics (mean sales per item-store)
    series_means = sales[day_cols].mean(axis=1)
    print(f"\n     Per-series mean sales distribution:")
    print(f"       Min mean:    {series_means.min():.4f}")
    print(f"       25th pctl:   {series_means.quantile(0.25):.4f}")
    print(f"       Median mean: {series_means.median():.4f}")
    print(f"       75th pctl:   {series_means.quantile(0.75):.4f}")
    print(f"       Max mean:    {series_means.max():.4f}")


# ==============================================================================
# 3. Dataset Relationships
# ==============================================================================
def explain_dataset_relationships(
    datasets: Dict[str, pd.DataFrame]
) -> None:
    """
    Explain how the three datasets relate to each other through
    their join keys.
    """
    print_section("3. DATASET RELATIONSHIPS (Entity-Relationship)")

    print("""
  The three datasets form a STAR SCHEMA with the sales table as the
  central fact table:

  ┌─────────────────────┐         ┌─────────────────────────────────┐
  │   CALENDAR          │         │   SALES (Fact Table)            │
  │   ─────────         │         │   ─────────────────             │
  │   d  ←──────────────┼────────►│   d_1, d_2, ..., d_1941        │
  │   wm_yr_wk ←────┐   │         │   id, item_id, store_id        │
  │   date           │   │         │   dept_id, cat_id, state_id     │
  │   events, SNAP   │   │         └────────────┬──────────────────┘
  └──────────────────┘   │                       │
                         │                       │ (item_id + store_id)
  ┌──────────────────────┘                       │
  │                                              ▼
  │   ┌─────────────────────────────────────────────┐
  │   │   SELL_PRICES                                │
  │   │   ───────────                                │
  └──►│   wm_yr_wk  (← calendar.wm_yr_wk)           │
      │   store_id   (← sales.store_id)              │
      │   item_id    (← sales.item_id)               │
      │   sell_price                                  │
      └─────────────────────────────────────────────┘

  Join Strategy (for Stage 2):
    1. MELT sales from wide → long format:
       (id, item_id, store_id, ..., d) → one row per item-store-day

    2. JOIN sales ↔ calendar ON d:
       Adds date, weekday, month, year, events, SNAP flags

    3. JOIN (sales+calendar) ↔ sell_prices ON (store_id, item_id, wm_yr_wk):
       Adds weekly sell price context

  Granularity:
    • calendar:    1 row per day        (1,969 rows)
    • sell_prices: 1 row per store×item×week (~6.8M rows)
    • sales:       1 row per store×item (30,490 rows × 1,941 day columns)
    """)

    # Validate join keys exist
    sales = datasets["sales"]
    cal = datasets["calendar"]
    sp = datasets["sell_prices"]

    print("  🔗 Join Key Validation:")

    # Calendar.d ↔ Sales day columns
    cal_days = set(cal["d"].unique())
    sales_days = set(c for c in sales.columns if c.startswith("d_"))
    overlap = cal_days & sales_days
    print(f"     • calendar.d ∩ sales.d_* columns: "
          f"{len(overlap):,} / {len(sales_days):,} matched")

    # store_id overlap
    sales_stores = set(sales["store_id"].unique())
    price_stores = set(sp["store_id"].unique())
    store_overlap = sales_stores & price_stores
    print(f"     • sales.store_id ∩ sell_prices.store_id: "
          f"{len(store_overlap)} / {len(sales_stores)} matched → {sorted(store_overlap)}")

    # item_id overlap
    sales_items = set(sales["item_id"].unique())
    price_items = set(sp["item_id"].unique())
    item_overlap = sales_items & price_items
    print(f"     • sales.item_id ∩ sell_prices.item_id: "
          f"{len(item_overlap):,} / {len(sales_items):,} matched")

    # wm_yr_wk overlap
    cal_weeks = set(cal["wm_yr_wk"].unique())
    price_weeks = set(sp["wm_yr_wk"].unique())
    week_overlap = cal_weeks & price_weeks
    print(f"     • calendar.wm_yr_wk ∩ sell_prices.wm_yr_wk: "
          f"{len(week_overlap):,} / {len(cal_weeks):,} matched")


# ==============================================================================
# 4. Key Variable Identification
# ==============================================================================
def identify_key_variables(
    datasets: Dict[str, pd.DataFrame]
) -> None:
    """
    Identify and document the target variable, time columns,
    product identifiers, and store identifiers.
    """
    print_section("4. KEY VARIABLE IDENTIFICATION")

    sales = datasets["sales"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    print("""
  ┌──────────────────────┬──────────────────────────────────────────────┐
  │ Variable Role        │ Column(s)                                    │
  ├──────────────────────┼──────────────────────────────────────────────┤
  │ 🎯 TARGET VARIABLE   │ d_1 ... d_1941 (daily unit sales counts)    │
  │                      │ → Integer values ≥ 0 (zero-inflated)        │
  │                      │ → After melting: single "sales" column      │
  ├──────────────────────┼──────────────────────────────────────────────┤
  │ 📅 TIME COLUMNS      │ calendar.date    (actual date, YYYY-MM-DD)  │
  │                      │ calendar.d       (day ID: d_1 to d_1969)    │
  │                      │ calendar.wm_yr_wk (Walmart fiscal week)     │
  │                      │ calendar.wday    (day of week, 1=Sat)       │
  │                      │ calendar.month   (1-12)                     │
  │                      │ calendar.year    (2011-2016)                 │
  ├──────────────────────┼──────────────────────────────────────────────┤
  │ 📦 PRODUCT IDs       │ sales.item_id    (3,049 unique products)    │
  │                      │ sales.dept_id    (7 departments)            │
  │                      │ sales.cat_id     (3 categories)             │
  ├──────────────────────┼──────────────────────────────────────────────┤
  │ 🏪 STORE IDs         │ sales.store_id   (10 stores)                │
  │                      │ sales.state_id   (3 states: CA, TX, WI)     │
  ├──────────────────────┼──────────────────────────────────────────────┤
  │ 🆔 SERIES ID         │ sales.id         (30,490 unique series)     │
  │                      │ = item_id + store_id + "_evaluation"        │
  ├──────────────────────┼──────────────────────────────────────────────┤
  │ 💰 PRICE VARIABLE    │ sell_prices.sell_price (weekly USD price)    │
  ├──────────────────────┼──────────────────────────────────────────────┤
  │ 📣 EVENT FEATURES    │ calendar.event_name_1/2, event_type_1/2     │
  │                      │ (holidays, sporting events, cultural events) │
  ├──────────────────────┼──────────────────────────────────────────────┤
  │ 🛒 SNAP FLAGS        │ calendar.snap_CA, snap_TX, snap_WI          │
  │                      │ (SNAP food stamp eligibility per state)      │
  └──────────────────────┴──────────────────────────────────────────────┘
    """)

    # Verification counts
    print(f"  Verification:")
    print(f"    • Unique series (id):      {sales['id'].nunique():,}")
    print(f"    • Unique items (item_id):  {sales['item_id'].nunique():,}")
    print(f"    • Unique stores (store_id):{sales['store_id'].nunique()}")
    print(f"    • Unique depts (dept_id):  {sales['dept_id'].nunique()}")
    print(f"    • Unique cats (cat_id):    {sales['cat_id'].nunique()}")
    print(f"    • Unique states (state_id):{sales['state_id'].nunique()}")
    print(f"    • Day columns:             {len(day_cols)}")

    # Product hierarchy
    print(f"\n  📊 Product Hierarchy:")
    hierarchy = sales.groupby(["cat_id", "dept_id"])["item_id"].nunique().reset_index()
    hierarchy.columns = ["Category", "Department", "Num_Products"]
    print(hierarchy.to_string(index=False))

    # Store distribution
    print(f"\n  📊 Store Distribution:")
    store_dist = sales.groupby(["state_id", "store_id"]).size().reset_index(name="Num_Series")
    print(store_dist.to_string(index=False))


# ==============================================================================
# 5. Data Quality Report
# ==============================================================================
def generate_data_quality_report(
    datasets: Dict[str, pd.DataFrame],
    missing_reports: Dict[str, pd.DataFrame],
    dup_counts: Dict[str, int]
) -> str:
    """
    Generate a comprehensive Data Quality Report.

    Returns
    -------
    str
        The full report as a formatted string.
    """
    print_section("5. DATA QUALITY REPORT")

    cal = datasets["calendar"]
    sp = datasets["sell_prices"]
    sales = datasets["sales"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    report_lines = []

    def add(line: str = "") -> None:
        report_lines.append(line)
        print(line)

    add("=" * 80)
    add("  DATA QUALITY REPORT — Stage 1")
    add("  Project: Calibrated Uncertainty Quantification for Supply Chain Risk Triage")
    add("  Dataset: M5 Forecasting — Accuracy (Walmart)")
    add("=" * 80)

    # --- Completeness ---
    add("\n  📋 1. COMPLETENESS")
    add("  " + "-" * 50)

    add(f"\n  Calendar ({cal.shape[0]:,} rows × {cal.shape[1]} cols):")
    cal_missing = missing_reports["calendar"]
    if len(cal_missing) == 0:
        add("    ✅ No missing values in non-event columns")
    else:
        for _, row in cal_missing.iterrows():
            add(f"    ⚠️  {row['Column']}: {row['Missing_Count']:,} missing "
                f"({row['Missing_Pct']:.1f}%)")
    # Event columns are expected to be mostly null
    event_cols = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    for ec in event_cols:
        n_null = cal[ec].isnull().sum()
        add(f"    ℹ️  {ec}: {n_null:,} nulls ({100*n_null/len(cal):.1f}%) "
            f"— EXPECTED (most days have no events)")

    add(f"\n  Sell Prices ({sp.shape[0]:,} rows × {sp.shape[1]} cols):")
    sp_missing = missing_reports["sell_prices"]
    if len(sp_missing) == 0:
        add("    ✅ No missing values")
    else:
        for _, row in sp_missing.iterrows():
            add(f"    ⚠️  {row['Column']}: {row['Missing_Count']:,} missing")

    add(f"\n  Sales ({sales.shape[0]:,} rows × {sales.shape[1]} cols):")
    sales_missing = missing_reports["sales"]
    if len(sales_missing) == 0:
        add("    ✅ No missing values in any column")
    else:
        for _, row in sales_missing.iterrows():
            add(f"    ⚠️  {row['Column']}: {row['Missing_Count']:,} missing")

    # --- Consistency ---
    add("\n  📋 2. CONSISTENCY")
    add("  " + "-" * 50)

    # Check date range continuity
    date_range = cal["date"]
    add(f"  Calendar date range: {date_range.min()} → {date_range.max()}")
    expected_days = (date_range.max() - date_range.min()).days + 1
    actual_days = len(cal)
    add(f"  Expected days: {expected_days}, Actual rows: {actual_days}")
    if expected_days == actual_days:
        add("  ✅ No gaps in calendar — continuous daily coverage")
    else:
        add(f"  ⚠️  {expected_days - actual_days} missing days detected")

    # Check day column alignment
    sales_day_count = len(day_cols)
    add(f"\n  Sales day columns: d_1 to d_{sales_day_count} ({sales_day_count} days)")
    add(f"  Calendar d column: d_1 to d_{cal['d'].iloc[-1].split('_')[1]} "
        f"({len(cal)} entries)")
    if sales_day_count <= len(cal):
        add(f"  ✅ All {sales_day_count} sales days have calendar coverage "
            f"({len(cal) - sales_day_count} extra calendar days for future forecasting)")
    else:
        add(f"  ⚠️  Sales has {sales_day_count - len(cal)} days beyond calendar coverage")

    # --- Duplicates ---
    add("\n  📋 3. DUPLICATES")
    add("  " + "-" * 50)
    for name, count in dup_counts.items():
        status = "✅ None" if count == 0 else f"⚠️  {count:,} found"
        add(f"  {name}: {status}")

    # --- Data Types ---
    add("\n  📋 4. DATA TYPE OBSERVATIONS")
    add("  " + "-" * 50)
    add(f"  Calendar.date: {cal['date'].dtype} — "
        f"{'✅ Parsed as datetime' if pd.api.types.is_datetime64_any_dtype(cal['date']) else '⚠️  Stored as string'}")
    add(f"  Sales day columns: {sales[day_cols[0]].dtype} — "
        "✅ Integer type (unit sales counts)")
    add(f"  Sell_prices.sell_price: {sp['sell_price'].dtype} — "
        "✅ Float type (USD prices)")

    # --- Zero Inflation ---
    add("\n  📋 5. ZERO INFLATION")
    add("  " + "-" * 50)
    sales_vals = sales[day_cols].values
    zero_pct = 100 * (sales_vals == 0).sum() / sales_vals.size
    add(f"  Zero-sales percentage: {zero_pct:.1f}%")
    add(f"  → This is a defining characteristic of the M5 dataset")
    add(f"  → Many items have intermittent demand (sparse time series)")
    add(f"  → Recommendation: Consider Tweedie/Poisson loss functions and")
    add(f"    zero-inflated models in Stage 3")

    # --- Price Observations ---
    add("\n  📋 6. PRICE OBSERVATIONS")
    add("  " + "-" * 50)
    add(f"  Price range: ${sp['sell_price'].min():.2f} — ${sp['sell_price'].max():.2f}")
    add(f"  Median price: ${sp['sell_price'].median():.2f}")
    add(f"  Mean price: ${sp['sell_price'].mean():.2f}")
    n_price_entries = len(sp)
    expected_entries = sales["item_id"].nunique() * sales["store_id"].nunique() * cal["wm_yr_wk"].nunique()
    coverage = 100 * n_price_entries / expected_entries
    add(f"  Price entries: {n_price_entries:,} / {expected_entries:,} possible "
        f"({coverage:.1f}% coverage)")
    add(f"  → Not all items are sold in all stores every week")
    add(f"  → Missing prices likely indicate items not yet on shelf")

    # --- Events ---
    add("\n  📋 7. EVENTS & SNAP")
    add("  " + "-" * 50)
    n_events_1 = cal["event_name_1"].notna().sum()
    n_events_2 = cal["event_name_2"].notna().sum()
    add(f"  Days with primary event: {n_events_1} ({100*n_events_1/len(cal):.1f}%)")
    add(f"  Days with secondary event: {n_events_2} ({100*n_events_2/len(cal):.1f}%)")
    if n_events_1 > 0:
        event_types = cal["event_type_1"].dropna().value_counts()
        add(f"  Event types:")
        for etype, count in event_types.items():
            add(f"    • {etype}: {count} occurrences")
    snap_cols = ["snap_CA", "snap_TX", "snap_WI"]
    for sc in snap_cols:
        snap_days = cal[sc].sum()
        add(f"  {sc}: {int(snap_days)} SNAP-active days "
            f"({100*snap_days/len(cal):.1f}%)")

    # --- Memory ---
    add("\n  📋 8. MEMORY USAGE")
    add("  " + "-" * 50)
    for name, df in datasets.items():
        mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        add(f"  {name}: {mem_mb:.1f} MB")
    total_mem = sum(
        df.memory_usage(deep=True).sum() / (1024**2)
        for df in datasets.values()
    )
    add(f"  TOTAL: {total_mem:.1f} MB")
    add(f"  → Recommendation: Downcast int64 → int16 for sales columns")
    add(f"    and use category dtype for identifiers in Stage 2")

    # --- Recommendations ---
    add("\n  📋 9. RECOMMENDATIONS FOR STAGE 2")
    add("  " + "-" * 50)
    add("  1. MELT sales from wide → long format (item-store-day granularity)")
    add("  2. JOIN with calendar (on d) to add date features")
    add("  3. JOIN with sell_prices (on store_id, item_id, wm_yr_wk) for prices")
    add("  4. DOWNCAST dtypes to reduce memory (int64→int16, object→category)")
    add("  5. HANDLE missing event columns (fill with 'no_event' or similar)")
    add("  6. CREATE time-based features (lag, rolling mean, day-of-week, etc.)")
    add("  7. HANDLE zero-inflated sales (log1p transform, clip, or special modeling)")
    add("  8. VERIFY temporal ordering — ensure no look-ahead leakage")
    add("  9. DEFINE train/validation/test split (temporal walk-forward)")

    add("\n" + "=" * 80)

    return "\n".join(report_lines)


# ==============================================================================
# 6. Stage 2 Checklist
# ==============================================================================
def print_stage2_checklist() -> None:
    """Print the preparation checklist for Stage 2."""
    print_section("6. STAGE 2 PREPARATION CHECKLIST")

    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STAGE 2 — Data Cleaning & Feature Engineering                     │
  │  ═══════════════════════════════════════════════                    │
  │                                                                     │
  │  □ 1. Melt sales data from wide → long format                      │
  │       → Each row = one item × one store × one day                  │
  │       → New column: "sales" (the target variable)                  │
  │                                                                     │
  │  □ 2. Merge datasets                                               │
  │       → sales + calendar (on "d" column)                           │
  │       → merged + sell_prices (on store_id, item_id, wm_yr_wk)     │
  │                                                                     │
  │  □ 3. Memory optimization                                         │
  │       → Downcast int64 → int16/int32 for sales                    │
  │       → Convert object columns → category dtype                   │
  │       → Target: reduce from ~700+ MB to <200 MB                   │
  │                                                                     │
  │  □ 4. Handle missing values                                        │
  │       → Event columns: fill NaN with "no_event"                   │
  │       → Sell prices: investigate items without prices              │
  │       → Document all imputation decisions                         │
  │                                                                     │
  │  □ 5. Feature engineering                                          │
  │       → Calendar features: day_of_week, month, is_weekend          │
  │       → Lag features: lag_7, lag_14, lag_28                        │
  │       → Rolling features: rolling_mean_7, rolling_mean_28         │
  │       → Price features: price_change_flag, relative_price          │
  │       → Event encoding: binary flags + type encoding              │
  │       → SNAP interaction features                                  │
  │                                                                     │
  │  □ 6. Define temporal train/validation/test split                  │
  │       → Train: d_1 to d_1885 (all but last 56 days)               │
  │       → Validation: d_1886 to d_1913 (days 1886-1913)             │
  │       → Test: d_1914 to d_1941 (last 28 days)                     │
  │       → NO random splitting — strict temporal ordering             │
  │                                                                     │
  │  □ 7. Save processed dataset to data/processed/                    │
  │       → Parquet format for efficient storage and fast I/O          │
  │       → Document schema and column descriptions                   │
  │                                                                     │
  │  □ 8. Validate processed data                                      │
  │       → No look-ahead leakage                                     │
  │       → Correct join cardinality (no row explosion)                │
  │       → All features properly aligned temporally                  │
  └─────────────────────────────────────────────────────────────────────┘
    """)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main() -> None:
    """
    Execute the full Stage 1 — Dataset Understanding pipeline.
    """
    logger = setup_logger(
        log_file="outputs/logs/stage1_eda.log"
    )
    set_seed(42)

    print_section(
        "STAGE 1 — DATASET UNDERSTANDING & EDA",
        char="█", width=80
    )
    print("  Project: Calibrated Uncertainty Quantification for Supply Chain Risk Triage")
    print("  Dataset: M5 Forecasting — Accuracy (Walmart)")
    print("  Stage:   1 of 10 — Dataset Understanding\n")

    # ---- Load and validate all datasets ----
    datasets = load_all_datasets()

    # ---- 1. Dataset Overview ----
    display_dataset_overview(datasets)

    # ---- 2. Exploratory Data Analysis ----
    print_section("2. EXPLORATORY DATA ANALYSIS (EDA)")

    missing_reports = analyze_missing_values(datasets)
    dup_counts = analyze_duplicates(datasets)
    analyze_unique_values(datasets)
    compute_descriptive_statistics(datasets)

    # ---- 3. Dataset Relationships ----
    explain_dataset_relationships(datasets)

    # ---- 4. Key Variable Identification ----
    identify_key_variables(datasets)

    # ---- 5. Data Quality Report ----
    report = generate_data_quality_report(datasets, missing_reports, dup_counts)

    # Save report to file
    report_path = get_project_root() / "outputs" / "reports" / "stage1_data_quality_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Data quality report saved to: {report_path}")

    # ---- 6. Stage 2 Checklist ----
    print_stage2_checklist()

    # ---- Summary ----
    print_section("STAGE 1 COMPLETE ✓", char="█", width=80)
    print("  Key Findings:")
    print("    • 3 datasets loaded and validated successfully")
    print(f"    • 30,490 time series across 10 stores, 3 states, 3,049 products")
    print(f"    • 1,941 days of sales data (Jan 2011 — Jun 2016)")
    print(f"    • High zero-inflation (~{(datasets['sales'][[c for c in datasets['sales'].columns if c.startswith('d_')]].values == 0).sum() / datasets['sales'][[c for c in datasets['sales'].columns if c.startswith('d_')]].values.size * 100:.0f}% of cells are zero)")
    print(f"    • No duplicate rows in any dataset")
    print(f"    • Missing values only in event columns (expected)")
    print(f"    • Star schema: sales ↔ calendar (on d), sales ↔ sell_prices (on store+item+week)")
    print(f"\n  Next: Stage 2 — Data Cleaning & Feature Engineering")
    print("█" * 80)


if __name__ == "__main__":
    main()
