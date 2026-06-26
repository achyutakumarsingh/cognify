"""
dashboard/data_loader.py
Centralised data loader and cache for the Streamlit dashboard.
All DataFrames and reports are loaded once and cached via st.cache_data.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent   # project root


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _parquet(rel: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / rel)

def _json(rel: str) -> dict:
    with open(ROOT / rel) as f:
        return json.load(f)

def _csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


# ──────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_business_data() -> pd.DataFrame:
    """Stage 7 enriched dataset: forecasts + risk + simulation costs."""
    df = _parquet("outputs/simulation/stage7_business_impact_data.parquet")
    # Ensure string types for categorical filters
    df["item_id"]   = df["item_id"].astype(str)
    df["store_id"]  = df["store_id"].astype(str)
    df["cat_id"]    = df["item_id"].str.split("_").str[0]
    df["dept_id"]   = df["item_id"].str.split("_").str[:2].str.join("_")
    return df

@st.cache_data(show_spinner=False)
def load_conformal_predictions() -> pd.DataFrame:
    """Stage 4 conformal prediction intervals."""
    df = _parquet("outputs/predictions/conformal_predictions.parquet")
    df["item_id"]  = df["item_id"].astype(str)
    df["store_id"] = df["store_id"].astype(str)
    return df

@st.cache_data(show_spinner=False)
def load_quantile_predictions() -> pd.DataFrame:
    """Stage 4 quantile regression prediction intervals."""
    df = _parquet("outputs/predictions/quantile_predictions.parquet")
    df["item_id"]  = df["item_id"].astype(str)
    df["store_id"] = df["store_id"].astype(str)
    return df

@st.cache_data(show_spinner=False)
def load_financial_summary() -> dict:
    return _json("outputs/reports/stage7_financial_summary.json")

@st.cache_data(show_spinner=False)
def load_calibration_report() -> dict:
    return _json("outputs/reports/stage5_calibration_report.json")

@st.cache_data(show_spinner=False)
def load_segment_analysis() -> pd.DataFrame:
    return _csv("outputs/reports/stage5_segment_analysis.csv")

@st.cache_data(show_spinner=False)
def load_sensitivity_summary() -> dict:
    return _json("outputs/reports/stage7_sensitivity_summary.json")

@st.cache_data(show_spinner=False)
def load_stage3_evaluation() -> dict:
    return _json("outputs/reports/stage3_evaluation_report.json")

@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    return _csv("outputs/reports/feature_importance.csv")

@st.cache_data(show_spinner=False)
def load_classification_report() -> dict:
    return _json("outputs/reports/stage6_classification_report.json")
