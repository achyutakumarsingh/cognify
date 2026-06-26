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
    "🔴 Risk & Operations":     "p2_risk_operations",
    "⚙️ Scenario Simulator":     "p3_scenario_simulator",
    "🧠 Technical Engine":       "p4_technical_engine",
}

# Initialize session state for demo
if "demo_active" not in st.session_state:
    st.session_state["demo_active"] = False
if "demo_step" not in st.session_state:
    st.session_state["demo_step"] = 0

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
    
    # Guided Demo checkbox
    demo_active = st.checkbox("🚀 Guided Demo Mode", value=st.session_state["demo_active"])
    st.session_state["demo_active"] = demo_active
    
    if demo_active:
        st.markdown("### 🧭 Demo Progress")
        steps = list(PAGES.keys())
        selected_index = st.session_state["demo_step"]
        # Ensure index is valid
        if selected_index >= len(steps):
            selected_index = 0
            st.session_state["demo_step"] = 0
        
        # Display list of steps
        for idx, step_name in enumerate(steps):
            if idx == selected_index:
                st.markdown(f"**👉 {step_name}**")
            else:
                st.markdown(f"<span style='color:#64748b'>{step_name}</span>", unsafe_allow_html=True)
                
        selected = steps[selected_index]
    else:
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
# If demo is active, show the Guided Demo banner at the top of the main area
if st.session_state["demo_active"]:
    steps = list(PAGES.keys())
    current_step = st.session_state["demo_step"]
    
    demo_tips = [
        "**Executive View:** Understand our business impact. We show a 15.6% cost reduction and an 8.9% increase in service levels.",
        "**Risk & Operations:** See the core Decision tool. We isolate High-Risk SKUs and offer one-click automated ordering workflows.",
        "**Scenario Simulator:** Drag sliders (Holding cost, Volatility) to see how safety stock requirements dynamically adapt to costs.",
        "**Technical Engine:** For technical judges. Inspect our Tweedie XGBoost residuals, conformal coverage, and feature gains."
    ]
    
    col_banner, col_btn = st.columns([4, 1])
    with col_banner:
        st.markdown(
            f"""<div style="background:linear-gradient(135deg,#312e81,#1e1b4b);
            border-left:4px solid #818cf8;border-radius:6px;padding:12px 16px;color:#e2e8f0;font-size:0.9rem">
            <b>🚀 DEMO STEP {current_step+1} OF 4:</b> {demo_tips[current_step]}
            </div>""",
            unsafe_allow_html=True,
        )
    with col_btn:
        if current_step < len(steps) - 1:
            if st.button("Next Step ➡️", use_container_width=True):
                st.session_state["demo_step"] += 1
                st.rerun()
        else:
            if st.button("Finish Demo 🔁", use_container_width=True):
                st.session_state["demo_step"] = 0
                st.session_state["demo_active"] = False
                st.rerun()

module_name = PAGES[selected]
import importlib
page_module = importlib.import_module(f"dashboard.pages.{module_name}")
page_module.render()
