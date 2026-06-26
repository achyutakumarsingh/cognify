"""
==============================================================================
Forecast Data Loader — Supply Chain Risk Triage
==============================================================================
Responsibility:
    Safely load the processed features.parquet based on the temporal_split_index
    and feature_metadata.
    
    Provides methods to yield the train, val, and test splits as distinct
    X, y pandas dataframes without data leakage. Memory efficiency is key,
    so we utilize PyArrow's pushdown filtering if needed.
==============================================================================
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import pyarrow.parquet as pq

from src.utils.helpers import setup_logger, get_project_root

logger = setup_logger()

class ForecastDataLoader:
    def __init__(
        self,
        features_path: str = "data/processed/features.parquet",
        metadata_path: str = "data/processed/feature_metadata.json",
        split_index_path: str = "data/processed/temporal_split_index.json",
        target_col: str = "sales"
    ):
        self.root = get_project_root()
        self.features_path = self.root / features_path
        self.metadata_path = self.root / metadata_path
        self.split_index_path = self.root / split_index_path
        self.target_col = target_col
        
        # Load schemas
        self.metadata = self._load_json(self.metadata_path)
        self.split_index = self._load_json(self.split_index_path)
        
        # Determine features to load
        self.feature_cols = [
            f for f, meta in self.metadata.items()
            if meta.get("group") not in ["identity", "meta", "target"]
        ]
        # Include item_id and store_id if categorical representation is enabled
        self.cols_to_load = self.feature_cols + [self.target_col, "split", "day_index", "item_id", "store_id"]
        
        logger.info(f"[ForecastDataLoader] Initialized. Expecting {len(self.feature_cols)} features.")

    def _load_json(self, path: Path) -> Dict:
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")
        with open(path, "r") as f:
            return json.load(f)

    def load_split(self, split_name: str, dev_mode: bool = False, max_series: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Loads a specific temporal split (train, val, or test).
        
        Parameters
        ----------
        split_name : str
            One of 'train', 'val', 'test'.
        dev_mode : bool
            If true, samples the dataset to speed up development pipelines.
        max_series : int
            Optionally limit the number of series loaded if in dev mode.
            
        Returns
        -------
        X : pd.DataFrame
        y : pd.Series
        """
        if split_name not in ["train", "val", "test"]:
            raise ValueError("split_name must be train, val, or test.")
            
        logger.info(f"[ForecastDataLoader] Loading {split_name} split...")
        
        # Using PyArrow filters for memory-efficient pushdown filtering
        filters = [("split", "==", split_name)]
        
        df = pd.read_parquet(
            self.features_path,
            columns=self.cols_to_load,
            filters=filters,
            engine="pyarrow"
        )
        
        if dev_mode or max_series is not None:
            # Subsample series
            series_limit = max_series if max_series else 20
            unique_items = df["item_id"].unique()[:series_limit]
            df = df[df["item_id"].isin(unique_items)]
            logger.info(f"[ForecastDataLoader] DEV MODE: limited {split_name} to {len(unique_items)} items ({len(df):,} rows)")
            
        X = df.drop(columns=[self.target_col, "split", "day_index"])
        
        # Set categorical dtypes explicitly for XGBoost if not already set by parquet
        for col in X.columns:
            if X[col].dtype.name in ["category", "object"]:
                X[col] = X[col].astype("category")
                
        y = df[self.target_col]
        
        logger.info(f"[ForecastDataLoader] Loaded {split_name}: X shape {X.shape}, y shape {y.shape}")
        return X, y

    def load_all_splits(self, dev_mode: bool = False) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Convenience method to load train, val, and test splits into memory.
        WARNING: Will consume significant memory if dev_mode is False.
        """
        splits = {}
        for s in ["train", "val", "test"]:
            try:
                splits[s] = self.load_split(s, dev_mode=dev_mode)
            except Exception as e:
                logger.warning(f"[ForecastDataLoader] Could not load split {s}: {e}")
        return splits
        
    def get_feature_names(self) -> List[str]:
        return self.feature_cols
