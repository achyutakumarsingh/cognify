"""
==============================================================================
Utility Helpers — Supply Chain Risk Triage
==============================================================================
Shared helper functions used across all pipeline stages:
  - YAML config loading
  - Logging setup
  - Reproducibility seeding
  - Memory-efficient CSV loading
==============================================================================
"""

import os
import sys
import yaml
import random
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional


# ==============================================================================
# 1. Project Root Resolution
# ==============================================================================
def get_project_root() -> Path:
    """
    Resolve the project root directory.

    Strategy: Walk upward from this file's location until we find the
    'config/' directory (our project marker). This makes the code
    independent of the working directory used to invoke scripts.

    Returns
    -------
    Path
        Absolute path to the project root directory.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "config").is_dir():
            return current
        current = current.parent
    # Fallback: assume CWD is the project root
    return Path.cwd()


# ==============================================================================
# 2. Configuration Loading
# ==============================================================================
def load_config(config_name: str = "paths.yaml") -> Dict[str, Any]:
    """
    Load a YAML configuration file from the config/ directory.

    Parameters
    ----------
    config_name : str
        Name of the YAML file inside config/ (default: 'paths.yaml').

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    """
    config_path = get_project_root() / "config" / config_name
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ==============================================================================
# 3. Logging Setup
# ==============================================================================
def setup_logger(
    name: str = "supply_chain",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configure and return a logger with console and optional file output.

    Parameters
    ----------
    name : str
        Logger name (default: 'supply_chain').
    log_file : str, optional
        Path to a log file. If None, only console output is used.
    level : int
        Logging level (default: logging.INFO).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = get_project_root() / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ==============================================================================
# 4. Reproducibility
# ==============================================================================
def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and (if
    available) PyTorch / TensorFlow.

    Parameters
    ----------
    seed : int
        Random seed value (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Optional: PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ==============================================================================
# 5. Data Loading Utilities
# ==============================================================================
def load_csv(
    relative_path: str,
    nrows: Optional[int] = None,
    dtype: Optional[Dict[str, Any]] = None,
    parse_dates: Optional[list] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load a CSV file using a path relative to the project root.

    Includes validation, memory reporting, and optional row limiting
    for rapid iteration during development.

    Parameters
    ----------
    relative_path : str
        Path relative to the project root (e.g., 'data/calendar.csv').
    nrows : int, optional
        Limit the number of rows read (useful for quick testing).
    dtype : dict, optional
        Column-specific dtype overrides for memory optimization.
    parse_dates : list, optional
        Columns to parse as datetime.
    verbose : bool
        If True, log shape and memory usage after loading.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the resolved file path does not exist.
    """
    logger = setup_logger()
    full_path = get_project_root() / relative_path

    if not full_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {full_path}\n"
            f"  (resolved from relative path: '{relative_path}')"
        )

    logger.info(f"Loading: {relative_path}")
    df = pd.read_csv(
        full_path,
        nrows=nrows,
        dtype=dtype,
        parse_dates=parse_dates
    )

    if verbose:
        mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        logger.info(
            f"  → Shape: {df.shape[0]:,} rows × {df.shape[1]} cols "
            f"| Memory: {mem_mb:.1f} MB"
        )

    return df


# ==============================================================================
# 6. Memory Usage Summary
# ==============================================================================
def memory_usage_summary(df: pd.DataFrame, name: str = "DataFrame") -> pd.DataFrame:
    """
    Generate a per-column memory usage report.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to profile.
    name : str
        Display name for the DataFrame.

    Returns
    -------
    pd.DataFrame
        Summary with columns: Column, Dtype, Memory_MB, Pct_of_Total.
    """
    mem = df.memory_usage(deep=True)
    total = mem.sum()

    records = []
    for col in df.columns:
        col_mem = mem[col]
        records.append({
            "Column": col,
            "Dtype": str(df[col].dtype),
            "Memory_MB": round(col_mem / (1024 ** 2), 3),
            "Pct_of_Total": round(100 * col_mem / total, 1)
        })

    summary = pd.DataFrame(records).sort_values(
        "Memory_MB", ascending=False
    ).reset_index(drop=True)

    logger = setup_logger()
    logger.info(
        f"Memory profile for '{name}': "
        f"{total / (1024**2):.1f} MB total"
    )
    return summary
