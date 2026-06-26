"""
app.py – Supply Chain Decision Intelligence Dashboard
Entry point: streamlit run app.py
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure project root is on sys.path so dashboard.* imports work
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# ─── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Decision Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  /* ── Base typography ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
  }

  /* ── App background ── */
  .stApp {
    background: linear-gradient(160deg, #0f0c29 0%, #16123a 50%, #0f0c29 100%);
    color: #e0e7ff;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #0f0c29 100%);
    border-right: 1px solid #4f46e544;
  }
  section[data-testid="stSidebar"] * {
    color: #c7d2fe !important;
  }

  /* ── Radio buttons in sidebar (nav) ── */
  div[role="radiogroup"] label {
    padding: 8px 12px !important;
    border-radius: 8px !important;
    margin: 2px 0 !important;
    transition: background 0.2s;
    display: block;
  }
  div[role="radiogroup"] label:hover {
    background: #4f46e522 !important;
  }

  /* ── Metrics ── */
  [data-testid="metric-container"] {
    background: #1e1b4b !important;
    border: 1px solid #4f46e540;
    border-radius: 10px;
    padding: 12px !important;
  }
  [data-testid="stMetricValue"] { color: #818cf8 !important; font-weight: 700 !important; }
  [data-testid="stMetricDelta"] { color: #4ade80 !important; }

  /* ── Dataframe ── */
  .dataframe { background: #1e1b4b !important; color: #e0e7ff !important; }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.88; }

  /* ── Selectbox / slider ── */
  .stSelectbox > div, .stSlider > div {
    background: #1e1b4b !important;
    border-color: #4f46e544 !important;
  }

  /* ── Headers ── */
  h1, h2, h3, h4, h5 { color: #e0e7ff !important; }

  /* ── Divider ── */
  hr { border-color: #4f46e533 !important; }

  /* ── Progress bar ── */
  .stProgress > div > div { background: #4f46e5 !important; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab"] {
    background: #1e1b4b;
    color: #818cf8;
    border-radius: 6px 6px 0 0;
  }
  .stTabs [aria-selected="true"] {
    background: #4f46e5 !important;
    color: #fff !important;
  }

  /* ── Hide Streamlit branding ── */
  #MainMenu, footer { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar Navigation ───────────────────────────────────────────────────────
PAGES = {
    "🏢 Executive Overview":     "p1_executive_overview",
    "📈 Demand Forecasting":     "p2_demand_forecasting",
    "🎯 Uncertainty Analysis":   "p3_uncertainty_analysis",
    "📐 Calibration Dashboard":  "p4_calibration_dashboard",
    "⚠️ Risk Intelligence":      "p5_risk_intelligence",
    "💰 Business Impact":        "p6_business_impact",
    "🔬 Scenario Simulator":     "p7_scenario_simulator",
    "🔍 Product Drill-Down":     "p8_product_drilldown",
    "💬 Explainability":         "p9_explainability",
    "▶️ Demo Mode":              "p10_demo_mode",
}

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 24px">
      <div style="font-size:2.2rem">🔮</div>
      <h2 style="color:#818cf8;margin:4px 0;font-size:1.05rem;letter-spacing:0.05em">
        SUPPLY CHAIN AI
      </h2>
      <p style="color:#64748b;font-size:0.72rem;margin:0">Decision Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    selected = st.radio("", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")

    st.markdown("""
    <div style="font-size:0.7rem;color:#475569;text-align:center;padding:8px 0">
      <b>Stages Complete:</b> 1 → 7<br>
      M5 Forecasting Competition Dataset<br>
      XGBoost · Conformal Prediction<br>
      Risk-Aware Inventory Optimization
    </div>
    """, unsafe_allow_html=True)


# ─── Dynamic page loading ─────────────────────────────────────────────────────
module_name = PAGES[selected]
import importlib
page_module = importlib.import_module(f"dashboard.pages.{module_name}")
page_module.render()
