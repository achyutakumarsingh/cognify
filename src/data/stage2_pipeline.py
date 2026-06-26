"""
==============================================================================
STAGE 2 — Production Preprocessing Pipeline  (Orchestrator)
==============================================================================
Project : Calibrated Uncertainty Quantification for Supply Chain Risk Triage
Stage   : 2 of 10  — Data Cleaning & Feature Engineering

This is the main entry point for Stage 2.  It orchestrates the following
pipeline in strict order:

    ┌─────────────────────────────────────────────────────────────┐
    │  RAW DATA  (calendar + sell_prices + sales wide-format)     │
    └─────────────────┬───────────────────────────────────────────┘
                      │ 1. Load & validate files
                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  MELT  (wide → long, 30,490 rows → ~59M rows)              │
    └─────────────────┬───────────────────────────────────────────┘
                      │ 2. Join calendar (on d)
                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  MERGE  (long + calendar + sell_prices)                     │
    └─────────────────┬───────────────────────────────────────────┘
                      │ 3. Impute missing values
                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  CLEAN  (fill events, ffill prices)                         │
    └─────────────────┬───────────────────────────────────────────┘
                      │ 4. Feature engineering (all groups)
                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  FEATURES  (date, price, event, SNAP, lag, rolling, split)  │
    └─────────────────┬───────────────────────────────────────────┘
                      │ 5. Memory optimisation
                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  OPTIMISE  (downcast dtypes, categoricals)                  │
    └─────────────────┬───────────────────────────────────────────┘
                      │ 6. Validate
                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  VALIDATE  (9-check suite)                                  │
    └─────────────────┬───────────────────────────────────────────┘
                      │ 7. Save artifacts
                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  ARTIFACTS  (Parquet + JSON metadata + config snapshot)     │
    └─────────────────────────────────────────────────────────────┘

Memory strategy:
    The full melted dataset is ~59 M rows.  To avoid OOM on typical
    developer machines, we process ONE STORE AT A TIME (10 stores),
    writing each batch to Parquet and concatenating at the end.
    Each store slice is ~5.9 M rows, comfortably under 1 GB.

Usage:
    python src/data/stage2_pipeline.py

    # Or for quick testing on 2 stores:
    python src/data/stage2_pipeline.py --stores CA_1 TX_1

    # Or on all series but limit to first N days (dev mode):
    python src/data/stage2_pipeline.py --max-days 200
==============================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Project imports ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.helpers import (
    get_project_root,
    load_config,
    setup_logger,
    set_seed,
)
from src.data.loader import load_all_datasets
from src.data.merger import (
    melt_sales,
    join_calendar,
    join_sell_prices,
    fill_missing_values,
)
from src.data.feature_engineer import (
    add_date_features,
    add_zero_inflation_features,
    add_price_features,
    add_event_features,
    add_snap_features,
    add_lag_features,
    add_rolling_features,
    add_split_column,
    FEATURE_METADATA,
    get_feature_columns,
)
from src.data.memory_optimizer import optimise_memory, dtype_report
from src.data.validator import (
    run_all_checks,
    missing_value_report,
    dataset_summary,
)
from src.data.artifact_saver import save_all_artifacts

logger = setup_logger(log_file="outputs/logs/stage2_pipeline.log")


# ═══════════════════════════════════════════════════════════════════════════════
# PreprocessingPipeline class
# ═══════════════════════════════════════════════════════════════════════════════
class PreprocessingPipeline:
    """
    Orchestrates the full Stage 2 preprocessing pipeline.

    Attributes
    ----------
    cfg_prep   : dict  — preprocessing.yaml['preprocessing']
    cfg_out    : dict  — preprocessing.yaml['outputs']
    stores     : list[str]  — store IDs to process (None = all)
    max_days   : int   — limit day columns for dev/testing (None = all)
    """

    def __init__(
        self,
        stores: Optional[List[str]] = None,
        max_days: Optional[int] = None,
    ) -> None:
        cfg = load_config("preprocessing.yaml")
        self.cfg_prep = cfg["preprocessing"]
        self.cfg_out  = cfg["outputs"]
        self.stores   = stores          # None → process all 10 stores
        self.max_days = max_days        # None → use all 1,941 days

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    def load(self) -> Dict[str, pd.DataFrame]:
        logger.info("━" * 60)
        logger.info("STEP 1 │ Loading raw datasets")
        logger.info("━" * 60)
        datasets = load_all_datasets()

        if self.max_days is not None:
            # Limit the number of day columns (dev/test mode)
            sales = datasets["sales"]
            day_cols = [c for c in sales.columns if c.startswith("d_")]
            keep_days = [f"d_{i}" for i in range(1, self.max_days + 1)
                         if f"d_{i}" in sales.columns]
            meta_cols = [c for c in sales.columns if not c.startswith("d_")]
            datasets["sales"] = sales[meta_cols + keep_days]
            logger.info(f"[DEV MODE] Limiting to {self.max_days} days "
                        f"({len(keep_days)} day columns)")

        return datasets

    # ── Step 2: Melt + Merge ─────────────────────────────────────────────────
    def merge(
        self,
        datasets: Dict[str, pd.DataFrame],
        store_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Melt (wide→long) and join all three datasets for one store.
        """
        logger.info(f"[Merge] Processing store: {store_id or 'ALL'}")

        # 2a. Melt
        store_filter = [store_id] if store_id else None
        long_df = melt_sales(datasets["sales"], store_ids=store_filter)

        # 2b. Join calendar
        long_df = join_calendar(long_df, datasets["calendar"])

        # 2c. Join sell prices
        long_df = join_sell_prices(long_df, datasets["sell_prices"])

        # 2d. Fill missing values
        long_df = fill_missing_values(long_df, self.cfg_prep)

        return long_df

    # ── Step 3: Feature Engineering ──────────────────────────────────────────
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations in the correct order.

        Order matters:
            1. Date features first → creates day_index (needed by price/lag)
            2. Zero-inflation → simple boolean on sales
            3. Price features → need day_index for sorting
            4. Event features → pure calendar-based, no ordering needed
            5. SNAP features  → needs state_id, existing snap_* columns
            6. Lag features   → MUST come after sort; need sales column intact
            7. Rolling stats  → same temporal requirement as lags
            8. Split column   → labels each row; must come last
        """
        logger.info("[Features] Starting feature engineering …")

        df = add_date_features(df)           # Step 1
        df = add_zero_inflation_features(df) # Step 2
        df = add_price_features(df)          # Step 3
        df = add_event_features(df)          # Step 4
        df = add_snap_features(df)           # Step 5
        df = add_lag_features(                # Step 6
            df,
            lag_windows=self.cfg_prep.get("lag_windows", [7, 14, 28])
        )
        df = add_rolling_features(            # Step 7
            df,
            rolling_windows=self.cfg_prep.get("rolling_windows", [7, 14, 28]),
            min_periods=self.cfg_prep.get("rolling_min_periods", 1),
        )
        df = add_split_column(df, self.cfg_prep)  # Step 8

        logger.info(f"[Features] Done. Total columns: {df.shape[1]}")
        return df

    # ── Step 4: Memory Optimisation ──────────────────────────────────────────
    def optimise(
        self, df: pd.DataFrame
    ):
        """Apply dtype downcasting and return (df, before_mb, after_mb)."""
        logger.info("━" * 60)
        logger.info("STEP 4 │ Memory optimisation")
        logger.info("━" * 60)
        df, before_mb, after_mb = optimise_memory(
            df, config=self.cfg_prep, label="merged_features"
        )
        return df, before_mb, after_mb

    # ── Step 5: Validate ─────────────────────────────────────────────────────
    def validate(
        self,
        df: pd.DataFrame,
        expected_series: int,
        expected_days: int,
    ) -> Dict:
        logger.info("━" * 60)
        logger.info("STEP 5 │ Validation suite")
        logger.info("━" * 60)
        return run_all_checks(df, expected_series, expected_days)

    # ── Step 6: Save ─────────────────────────────────────────────────────────
    def save(
        self,
        df: pd.DataFrame,
        check_results: Dict,
        before_mb: float,
        after_mb: float,
    ) -> Dict[str, Path]:
        logger.info("━" * 60)
        logger.info("STEP 6 │ Saving artifacts")
        logger.info("━" * 60)
        return save_all_artifacts(df, check_results, before_mb, after_mb)

    # ── Full pipeline ─────────────────────────────────────────────────────────
    def run(self) -> pd.DataFrame:
        """
        Execute the complete Stage 2 pipeline and return the processed DataFrame.
        """
        t0 = time.time()
        set_seed(42)

        logger.info("█" * 60)
        logger.info("  STAGE 2 — Production Preprocessing Pipeline")
        logger.info("█" * 60)

        # ── Step 1: Load ────────────────────────────────────────────────────
        datasets = self.load()
        sales = datasets["sales"]
        all_stores = sorted(sales["store_id"].unique())
        stores_to_process = self.stores if self.stores else all_stores
        day_cols = [c for c in sales.columns if c.startswith("d_")]
        n_days = len(day_cols)
        n_series = len(sales[sales["store_id"].isin(stores_to_process)])

        logger.info(f"Stores to process: {stores_to_process} ({len(stores_to_process)} total)")
        logger.info(f"Days: {n_days}  |  Series: {n_series:,}")

        # ── Step 2+3: Melt + Merge + Feature Engineering (per-store) ────────
        logger.info("━" * 60)
        logger.info("STEP 2+3 │ Melt → Merge → Feature Engineering (per-store)")
        logger.info("━" * 60)

        store_chunks: List[pd.DataFrame] = []

        for store_id in stores_to_process:
            t_store = time.time()
            logger.info(f"\n  ── Store: {store_id} ──")

            # 2a-d: melt + join + fill
            store_df = self.merge(datasets, store_id=store_id)

            # 3: feature engineering
            store_df = self.engineer_features(store_df)

            elapsed = time.time() - t_store
            logger.info(f"  Store {store_id}: {len(store_df):,} rows | "
                        f"{store_df.shape[1]} cols | {elapsed:.1f}s")

            store_chunks.append(store_df)

        # ── Concatenate all stores ────────────────────────────────────────────
        logger.info("\nConcatenating all store chunks …")
        df = pd.concat(store_chunks, ignore_index=True)
        del store_chunks
        logger.info(f"Concatenated: {len(df):,} rows × {df.shape[1]} cols")

        # ── Step 4: Memory Optimisation ────────────────────────────────────
        df, before_mb, after_mb = self.optimise(df)

        # ── Dataset summary ────────────────────────────────────────────────
        dataset_summary(df)

        # ── Missing value report ───────────────────────────────────────────
        mvr = missing_value_report(df)
        if len(mvr) > 0:
            logger.warning(f"[Missing values] {len(mvr)} columns have NaN:\n{mvr}")
        else:
            logger.info("[Missing values] ✓ No missing values in processed dataset")

        # ── Dtype report ───────────────────────────────────────────────────
        dr = dtype_report(df)
        logger.info(f"\n[Dtype report]\n{dr.to_string(index=False)}")

        # ── Step 5: Validate ────────────────────────────────────────────────
        expected_series = n_series
        expected_days   = n_days
        check_results = self.validate(df, expected_series, expected_days)

        # ── Step 6: Save ────────────────────────────────────────────────────
        saved_paths = self.save(df, check_results, before_mb, after_mb)

        # ── Summary ─────────────────────────────────────────────────────────
        elapsed_total = time.time() - t0
        self._print_summary(df, check_results, before_mb, after_mb,
                            saved_paths, elapsed_total)

        return df

    # ── Summary printer ───────────────────────────────────────────────────────
    def _print_summary(
        self,
        df: pd.DataFrame,
        check_results: Dict,
        before_mb: float,
        after_mb: float,
        saved_paths: Dict,
        elapsed: float,
    ) -> None:
        n_pass = sum(1 for p, _ in check_results.values() if p)
        n_total = len(check_results)

        logger.info("\n" + "█" * 60)
        logger.info("  STAGE 2 COMPLETE ✓")
        logger.info("█" * 60)
        logger.info(f"  Total time:      {elapsed:.1f}s ({elapsed/60:.1f} min)")
        logger.info(f"  Dataset shape:   {len(df):,} rows × {df.shape[1]} cols")
        logger.info(f"  Memory:          {before_mb:.0f} MB → {after_mb:.0f} MB "
                    f"(↓ {100*(1-after_mb/before_mb):.0f}%)")
        logger.info(f"  Validation:      {n_pass}/{n_total} checks passed")
        logger.info(f"  Feature count:   {len(get_feature_columns())} model features")
        logger.info(f"  Saved artifacts:")
        for name, path in saved_paths.items():
            logger.info(f"    • {name}: {path}")
        logger.info("█" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# Zero-Inflation Analysis Report
# ═══════════════════════════════════════════════════════════════════════════════
def analyze_zero_inflation(df: pd.DataFrame) -> None:
    """
    Detailed analysis of zero-sales distribution.

    Prints a breakdown of zero rates:
    - Overall
    - By category
    - By store
    - By split (train/val/test)

    Also explains how later models should handle this.
    """
    logger.info("\n" + "=" * 60)
    logger.info("  ZERO-INFLATION ANALYSIS")
    logger.info("=" * 60)

    overall = (df["sales"] == 0).mean()
    logger.info(f"\n  Overall zero rate: {overall:.1%}")

    if "cat_id" in df.columns:
        logger.info("\n  Zero rate by category:")
        by_cat = df.groupby("cat_id", observed=True)["sales"].apply(
            lambda s: (s == 0).mean()
        ).reset_index()
        for _, row in by_cat.iterrows():
            logger.info(f"    {str(row['cat_id']):20s}: {row['sales']:.1%}")

    if "store_id" in df.columns:
        logger.info("\n  Zero rate by store:")
        by_store = df.groupby("store_id", observed=True)["sales"].apply(
            lambda s: (s == 0).mean()
        ).reset_index()
        for _, row in by_store.iterrows():
            logger.info(f"    {str(row['store_id']):10s}: {row['sales']:.1%}")

    if "split" in df.columns:
        logger.info("\n  Zero rate by split:")
        by_split = df.groupby("split", observed=True)["sales"].apply(
            lambda s: (s == 0).mean()
        ).reset_index()
        for _, row in by_split.iterrows():
            logger.info(f"    {str(row['split']):6s}: {row['sales']:.1%}")

    logger.info("""
  ── Modelling recommendations for Stage 3 ──────────────────
  1. LOSS FUNCTION: Use Tweedie or Poisson loss instead of MSE.
     Tweedie handles zero-inflated count data natively and is
     the default in winning M5 solutions.

  2. EVALUATION: WRMSSE (weighted RMSSE) — the M5 metric —
     down-weights zeros naturally.  Do NOT use plain RMSE.

  3. FEATURES: rolling_std_7 and rolling_std_28 encode the
     historical intermittency of each series.  Series with high
     rolling std are candidates for wider prediction intervals
     (Stage 4 Uncertainty Estimation).

  4. DO NOT REMOVE ZEROS: They represent real market behaviour
     (product not sold / out of stock / not yet listed).  Removing
     them would introduce sampling bias and inflate forecast accuracy.

  5. INDICATOR FEATURE: is_zero_sales (created in Stage 2) can
     be used as a response variable in a two-stage model:
       Stage A: classify "will there be any demand?" (binary)
       Stage B: forecast sales volume given demand > 0
  """)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: Production Preprocessing Pipeline"
    )
    parser.add_argument(
        "--stores", nargs="*",
        default=None,
        help="Specific store IDs to process (default: all 10 stores). "
             "Example: --stores CA_1 TX_1"
    )
    parser.add_argument(
        "--max-days", type=int, default=None,
        help="Limit to first N day columns (for dev/testing). "
             "Example: --max-days 200"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  Stage 2 Configuration")
    logger.info("=" * 60)
    logger.info(f"  Stores:   {args.stores or 'ALL'}")
    logger.info(f"  Max days: {args.max_days or 'ALL (1941)'}")
    logger.info("=" * 60)

    pipeline = PreprocessingPipeline(
        stores=args.stores,
        max_days=args.max_days,
    )
    df = pipeline.run()

    # Zero-inflation detailed analysis
    analyze_zero_inflation(df)

    logger.info("\n✓ Stage 2 complete. Ready for Stage 3 — Demand Forecasting.")


if __name__ == "__main__":
    main()
