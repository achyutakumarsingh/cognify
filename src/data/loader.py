"""
==============================================================================
Data Loader — Supply Chain Risk Triage
==============================================================================
Responsible for loading and validating all three M5 datasets:
  1. calendar.csv      — Date metadata, events, and SNAP flags
  2. sell_prices.csv   — Weekly sell prices per store × item
  3. sales_train_evaluation.csv — Daily unit sales in wide format

Design Decisions:
  - This module only LOADS and VALIDATES. No cleaning or transformation.
  - Uses the centralized paths.yaml configuration.
  - Returns a dictionary of DataFrames for downstream consumption.
  - Validates file existence before attempting to load.
==============================================================================
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Add project root to path so we can import utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.helpers import (
    load_config,
    load_csv,
    setup_logger,
    get_project_root
)


# ==============================================================================
# Dataset Loader
# ==============================================================================
def validate_file_exists(relative_path: str) -> bool:
    """
    Check whether a data file exists at the given relative path.

    Parameters
    ----------
    relative_path : str
        Path relative to the project root.

    Returns
    -------
    bool
        True if the file exists, False otherwise.
    """
    full_path = get_project_root() / relative_path
    return full_path.exists()


def load_all_datasets(
    nrows_sales: Optional[int] = None,
    nrows_prices: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load and validate all three M5 datasets.

    Parameters
    ----------
    nrows_sales : int, optional
        Limit rows for the sales dataset (for quick iteration).
    nrows_prices : int, optional
        Limit rows for the sell_prices dataset.

    Returns
    -------
    dict
        Dictionary with keys: 'calendar', 'sell_prices', 'sales'.
        Each value is a pandas DataFrame.

    Raises
    ------
    FileNotFoundError
        If any of the three required files is missing.
    """
    logger = setup_logger()
    config = load_config("paths.yaml")
    raw_paths = config["data"]["raw"]

    # ---- Step 1: Validate all files exist BEFORE loading any ----
    logger.info("=" * 60)
    logger.info("DATASET VALIDATION")
    logger.info("=" * 60)

    all_exist = True
    for name, rel_path in raw_paths.items():
        exists = validate_file_exists(rel_path)
        status = "✓ FOUND" if exists else "✗ MISSING"
        logger.info(f"  {status}: {rel_path}")
        if not exists:
            all_exist = False

    if not all_exist:
        raise FileNotFoundError(
            "One or more required data files are missing. "
            "Please ensure all CSV files are in the data/ directory."
        )

    logger.info("All required files validated successfully.\n")

    # ---- Step 2: Load datasets ----
    logger.info("=" * 60)
    logger.info("LOADING DATASETS")
    logger.info("=" * 60)

    datasets = {}

    # Calendar — small dataset, load fully with date parsing
    datasets["calendar"] = load_csv(
        raw_paths["calendar"],
        parse_dates=["date"]
    )

    # Sell Prices — large dataset (~6.8M rows), optionally limit
    datasets["sell_prices"] = load_csv(
        raw_paths["sell_prices"],
        nrows=nrows_prices
    )

    # Sales — wide-format dataset (30,490 rows × 1,947 cols)
    datasets["sales"] = load_csv(
        raw_paths["sales"],
        nrows=nrows_sales
    )

    logger.info("\nAll datasets loaded successfully.")
    logger.info("=" * 60)

    return datasets


# ==============================================================================
# Dataset Descriptions
# ==============================================================================
DATASET_DESCRIPTIONS = {
    "calendar": {
        "description": (
            "Calendar metadata for each day in the dataset (d_1 through d_1969). "
            "Contains the actual date, day of week, month, year, Walmart fiscal "
            "week (wm_yr_wk), event names/types (e.g., SuperBowl, Christmas), "
            "and SNAP purchase indicator flags for CA, TX, and WI."
        ),
        "key_columns": {
            "d": "Day identifier (d_1 to d_1969) — join key to sales data",
            "date": "Calendar date (YYYY-MM-DD)",
            "wm_yr_wk": "Walmart fiscal year-week — join key to sell_prices",
            "event_name_1/2": "Named events (holidays, sporting, cultural)",
            "event_type_1/2": "Event category (Sporting, Cultural, National, Religious)",
            "snap_CA/TX/WI": "SNAP benefit eligibility flag per state (0 or 1)",
        },
        "role": "Time dimension table — links day IDs to calendar features",
    },
    "sell_prices": {
        "description": (
            "Weekly sell prices for each product at each store. Prices change "
            "at the Walmart fiscal-week level (wm_yr_wk). This is a fact table "
            "with ~6.8 million rows covering all store × item × week combinations "
            "where the item was available for sale."
        ),
        "key_columns": {
            "store_id": "Store identifier (e.g., CA_1, TX_2, WI_3)",
            "item_id": "Product identifier (e.g., HOBBIES_1_001)",
            "wm_yr_wk": "Walmart fiscal year-week — join key to calendar",
            "sell_price": "Weekly unit sell price in USD",
        },
        "role": "Price dimension table — provides pricing context for each item-store-week",
    },
    "sales": {
        "description": (
            "Daily unit sales for 30,490 time series (unique item × store "
            "combinations). Data is in WIDE format: each row is a single "
            "product-store combination, and columns d_1 through d_1941 "
            "represent daily unit sales. The evaluation dataset extends "
            "28 days beyond the validation dataset."
        ),
        "key_columns": {
            "id": "Unique series identifier (e.g., HOBBIES_1_001_CA_1_evaluation)",
            "item_id": "Product identifier — join key to sell_prices",
            "dept_id": "Department (e.g., HOBBIES_1, FOODS_3)",
            "cat_id": "Category (HOBBIES, HOUSEHOLD, FOODS)",
            "store_id": "Store identifier — join key to sell_prices",
            "state_id": "State (CA, TX, WI)",
            "d_1 ... d_1941": "Daily unit sales (the TARGET variable)",
        },
        "role": "Core fact table — contains the time series to forecast",
    },
}
