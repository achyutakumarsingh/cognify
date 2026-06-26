"""
==============================================================================
Artifact Saver — Supply Chain Risk Triage  (Stage 2)
==============================================================================
Responsibility:
    Persist all preprocessing outputs to disk in formats optimised for
    downstream consumption by Stage 3+.

Formats chosen:
    Parquet  — columnar, compressed, schema-preserving.  10-100× faster
               to read than CSV for analytical workloads.  Preserves dtypes
               (including Categorical) natively via PyArrow.

    JSON     — human-readable, lightweight.  Used for metadata, configs,
               and split indices that are read programmatically.

    YAML     — snapshot of the preprocessing configuration used, so any
               stage can reproduce the exact same setup.
==============================================================================
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.helpers import setup_logger, load_config, get_project_root
from src.data.feature_engineer import FEATURE_METADATA

logger = setup_logger()


# ==============================================================================
# 1. Save Feature-Engineered Dataset
# ==============================================================================
def save_features_parquet(
    df: pd.DataFrame,
    relative_path: Optional[str] = None,
    partition_by: Optional[str] = None,
) -> Path:
    """
    Save the processed feature DataFrame to Parquet.

    Why Parquet over CSV?
        - Preserves dtypes (int8, float32, Categorical) — no parsing on load.
        - Columnar storage: Stage 3 can load only the columns it needs.
        - Snappy compression reduces disk use by ~5-10× vs uncompressed CSV.
        - Read speed: ~50-100× faster than CSV for large files.

    Parameters
    ----------
    df             : pd.DataFrame
    relative_path  : str, optional  (default from preprocessing.yaml)
    partition_by   : str, optional  (e.g. 'store_id' for partitioned Parquet)

    Returns
    -------
    Path  — absolute path to saved file / directory
    """
    if relative_path is None:
        cfg = load_config("preprocessing.yaml")
        relative_path = cfg["outputs"]["features_parquet"]

    out_path = get_project_root() / relative_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # pandas Categorical columns must be converted to string-backed categories
    # before writing to Parquet (PyArrow supports dictionary-encoded strings).
    # We do NOT convert — PyArrow handles Categorical natively since v5.0.

    if partition_by and partition_by in df.columns:
        # Write partitioned Parquet (one sub-file per partition value)
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(out_path),
            partition_cols=[partition_by],
            compression="snappy",
            use_legacy_dataset=False,
        )
        logger.info(f"[Save] Partitioned Parquet saved to {out_path}/ "
                    f"(partition by {partition_by})")
    else:
        df.to_parquet(str(out_path), index=False, compression="snappy", engine="pyarrow")
        size_mb = out_path.stat().st_size / (1024 ** 2)
        logger.info(f"[Save] Features Parquet saved: {out_path} ({size_mb:.1f} MB)")

    return out_path


# ==============================================================================
# 2. Save Feature Metadata
# ==============================================================================
def save_feature_metadata(relative_path: Optional[str] = None) -> Path:
    """
    Save the FEATURE_METADATA catalogue to JSON.

    This file serves as the schema contract between Stage 2 and all later
    stages.  Any stage can read it to know what features exist, their group,
    expected dtype, and description.

    Returns
    -------
    Path
    """
    if relative_path is None:
        cfg = load_config("preprocessing.yaml")
        relative_path = cfg["outputs"]["feature_metadata"]

    out_path = get_project_root() / relative_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(FEATURE_METADATA, f, indent=2)

    logger.info(f"[Save] Feature metadata saved: {out_path} "
                f"({len(FEATURE_METADATA)} features documented)")
    return out_path


# ==============================================================================
# 3. Save Temporal Split Index
# ==============================================================================
def save_split_index(
    df: pd.DataFrame,
    relative_path: Optional[str] = None,
) -> Path:
    """
    Save a JSON file mapping split name → day_index range.

    Stage 3+ can use this to correctly slice the Parquet file without
    hardcoding day boundaries.

    Returns
    -------
    Path
    """
    if relative_path is None:
        cfg = load_config("preprocessing.yaml")
        relative_path = cfg["outputs"]["split_index"]

    out_path = get_project_root() / relative_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    split_info: Dict[str, Any] = {}
    if "split" in df.columns and "day_index" in df.columns:
        for s in ["train", "val", "test"]:
            mask = df["split"] == s
            if mask.any():
                split_info[s] = {
                    "day_index_min": int(df.loc[mask, "day_index"].min()),
                    "day_index_max": int(df.loc[mask, "day_index"].max()),
                    "n_days": int(df.loc[mask, "day_index"].nunique()),
                    "n_rows": int(mask.sum()),
                }
        if "date" in df.columns:
            for s in ["train", "val", "test"]:
                mask = df["split"] == s
                if mask.any():
                    split_info[s]["date_start"] = str(df.loc[mask, "date"].min().date())
                    split_info[s]["date_end"]   = str(df.loc[mask, "date"].max().date())

    with open(out_path, "w") as f:
        json.dump(split_info, f, indent=2)

    logger.info(f"[Save] Temporal split index saved: {out_path}")
    return out_path


# ==============================================================================
# 4. Save Preprocessing Config Snapshot
# ==============================================================================
def save_config_snapshot(relative_path: Optional[str] = None) -> Path:
    """
    Copy the current preprocessing.yaml into the processed/ directory.

    Rationale: if the config changes in a future run, the snapshot tells us
    exactly which parameters were used to produce a given artefact.

    Returns
    -------
    Path
    """
    if relative_path is None:
        cfg = load_config("preprocessing.yaml")
        relative_path = cfg["outputs"]["preprocessing_config_snapshot"]

    out_path = get_project_root() / relative_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src_path = get_project_root() / "config" / "preprocessing.yaml"
    shutil.copy2(src_path, out_path)

    logger.info(f"[Save] Config snapshot saved: {out_path}")
    return out_path


# ==============================================================================
# 5. Save Validation Report
# ==============================================================================
def save_validation_report(
    check_results: Dict,
    memory_before_mb: float,
    memory_after_mb: float,
    df: pd.DataFrame,
    relative_path: str = "outputs/reports/stage2_validation_report.json",
) -> Path:
    """
    Persist the validation results and memory stats to JSON.

    Returns
    -------
    Path
    """
    out_path = get_project_root() / relative_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "dataset_shape": {"rows": len(df), "columns": df.shape[1]},
        "memory": {
            "before_optimization_mb": round(memory_before_mb, 1),
            "after_optimization_mb": round(memory_after_mb, 1),
            "reduction_pct": round(100 * (1 - memory_after_mb / memory_before_mb), 1)
            if memory_before_mb > 0 else 0,
        },
        "validation_checks": {
            name: {"passed": passed, "message": msg}
            for name, (passed, msg) in check_results.items()
        },
        "n_checks_passed": sum(1 for p, _ in check_results.values() if p),
        "n_checks_total": len(check_results),
        "all_passed": all(p for p, _ in check_results.values()),
    }

    if "split" in df.columns:
        report["split_counts"] = df["split"].value_counts().to_dict()

    if "sales" in df.columns:
        report["target_stats"] = {
            "mean": round(float(df["sales"].mean()), 4),
            "std": round(float(df["sales"].std()), 4),
            "max": int(df["sales"].max()),
            "zero_pct": round(float((df["sales"] == 0).mean()) * 100, 1),
        }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"[Save] Validation report saved: {out_path}")
    return out_path


# ==============================================================================
# 6. Save All Artifacts at Once
# ==============================================================================
def save_all_artifacts(
    df: pd.DataFrame,
    check_results: Dict,
    memory_before_mb: float,
    memory_after_mb: float,
) -> Dict[str, Path]:
    """
    Convenience wrapper: save all Stage 2 outputs in one call.

    Returns
    -------
    dict  {artifact_name: path}
    """
    paths = {}
    paths["features_parquet"]  = save_features_parquet(df)
    paths["feature_metadata"]  = save_feature_metadata()
    paths["split_index"]       = save_split_index(df)
    paths["config_snapshot"]   = save_config_snapshot()
    paths["validation_report"] = save_validation_report(
        check_results, memory_before_mb, memory_after_mb, df
    )
    return paths
