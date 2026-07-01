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
    initial_sidebar_state="collapsed",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  /* Brand */
  --cognify-blue: #4F6BED;
  --cognify-blue-light: #7B93F5;
  --cognify-blue-dim: rgba(79, 107, 237, 0.15);

  /* Surfaces */
  --surface-0: #0F1117;    /* page background */
  --surface-1: #1A1D2E;    /* cards */
  --surface-2: #222640;    /* elevated cards, modals */
  --surface-3: #2A2F52;    /* hover states, selected */

  /* Semantic Status */
  --red-alert:   #FF4D6A;  /* stockout risk, critical */
  --red-dim:     rgba(255, 77, 106, 0.12);
  --amber-warn:  #FFB547;  /* medium risk, warning */
  --amber-dim:   rgba(255, 181, 71, 0.12);
  --green-safe:  #2DD4A7;  /* healthy stock, safe */
  --green-dim:   rgba(45, 212, 167, 0.12);

  /* Text hierarchy */
  --text-primary:   #E8EAF0;
  --text-secondary: #9CA3C4;
  --text-muted:     #5C6180;

  /* Borders */
  --border:        rgba(255,255,255,0.07);
  --border-strong: rgba(255,255,255,0.14);
  --border-accent: rgba(79, 107, 237, 0.4);
}

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
}

/* Page title — used once per page */
.cog-title {
  font-size: 28px;
  font-weight: 600;
  color: var(--text-primary);
  letter-spacing: -0.5px;
  line-height: 1.2;
  margin-bottom: 4px;
}

/* Section header */
.cog-section {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 1.2px;
  margin: 28px 0 12px 0;
}

/* KPI number — the big hero stat */
.cog-kpi-value {
  font-size: 36px;
  font-weight: 600;
  color: var(--text-primary);
  line-height: 1;
  font-variant-numeric: tabular-nums;
}

/* KPI label */
.cog-kpi-label {
  font-size: 12px;
  font-weight: 500;
  color: var(--text-secondary);
  margin-top: 4px;
}

/* Delta — positive/negative change indicator */
.cog-delta-up   { color: var(--green-safe); font-size: 13px; margin-top: 6px; font-weight: 500; }
.cog-delta-down { color: var(--red-alert);  font-size: 13px; margin-top: 6px; font-weight: 500; }

/* Monospace for numbers in tables */
.cog-mono {
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
}

/* Body insight text — the plain-English translation */
.cog-insight {
  font-size: 14px;
  color: var(--text-secondary);
  line-height: 1.6;
  border-left: 3px solid var(--cognify-blue);
  padding-left: 14px;
  margin: 12px 0;
}

/* ── Hide Sidebar Completely ── */
[data-testid="stSidebar"], section[data-testid="stSidebar"] {
  display: none !important;
}
[data-testid="collapsedControl"] {
  display: none !important;
}

.cog-logo {
  padding: 24px 20px 20px 20px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 8px;
}
.cog-logo-text {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
  letter-spacing: -0.3px;
}
.cog-logo-sub {
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 2px;
}

/* Radio buttons in sidebar (nav) acting as cog-nav-item */
div[role="radiogroup"] label {
  display: flex !important;
  align-items: center !important;
  gap: 10px !important;
  padding: 10px 20px !important;
  border-radius: 8px !important;
  margin: 2px 8px !important;
  font-size: 14px !important;
  color: var(--text-secondary) !important;
  cursor: pointer !important;
  transition: all 0.15s !important;
}
div[role="radiogroup"] label:hover {
  background: var(--cognify-blue-dim) !important;
  color: var(--cognify-blue-light) !important;
}
div[role="radiogroup"] label[data-checked="true"] {
  background: var(--cognify-blue-dim) !important;
  color: var(--cognify-blue-light) !important;
}

/* Status pill at bottom of sidebar */
.cog-sidebar-status {
  margin: 20px 12px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 16px;
  font-size: 12px;
  color: var(--text-muted);
}

/* ── Main Content Area ── */
.main .block-container {
  padding-top: 2rem !important;
  padding-left: 2.5rem !important;
  padding-right: 2.5rem !important;
  max-width: 1200px !important;
}

/* Every st.dataframe gets dark themed */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  overflow: hidden !important;
}

/* Streamlit selectbox / input dark styling */
[data-testid="stSelectbox"] > div > div {
  background: var(--surface-2) !important;
  border-color: var(--border-strong) !important;
  border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
  background: var(--cognify-blue) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 500 !important;
  padding: 10px 20px !important;
  font-size: 14px !important;
  transition: all 0.15s !important;
}
.stButton > button:hover {
  background: var(--cognify-blue-light) !important;
  transform: translateY(-1px);
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer { visibility: hidden !important; }
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { visibility: hidden !important; }
[data-testid="manage-app-button"] { display: none !important; }
[class^="viewerBadge"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar Navigation ───────────────────────────────────────────────────────
PAGES = {
    "🏢 Executive Overview":     "p1_executive_overview",
    "🔴 Risk & Operations":     "p2_risk_operations",
    "⚙️ Scenario Simulator":     "p3_scenario_simulator",
    "🧠 Technical Engine":       "p4_technical_engine",
}

# Initialize session state for demo and navigation
if "demo_active" not in st.session_state:
    st.session_state["demo_active"] = False
if "demo_toggle_widget" not in st.session_state:
    st.session_state["demo_toggle_widget"] = False
if "demo_step" not in st.session_state:
    st.session_state["demo_step"] = 0
if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = list(PAGES.keys())[0]
if "topnav_nav" not in st.session_state:
    st.session_state["topnav_nav"] = list(PAGES.keys())[0]
if "show_nav" not in st.session_state:
    st.session_state["show_nav"] = True

# Callbacks to sync topnav
def on_topnav_change():
    page = st.session_state["topnav_nav"]
    st.session_state["selected_page"] = page
    if st.session_state["demo_active"]:
        st.session_state["demo_step"] = list(PAGES.keys()).index(page)

def on_demo_toggle():
    active = st.session_state["demo_toggle_widget"]
    st.session_state["demo_active"] = active
    if active:
        st.session_state["demo_step"] = 0
        first_page = list(PAGES.keys())[0]
        st.session_state["selected_page"] = first_page
        st.session_state["topnav_nav"] = first_page




# ─── Top Navigation Bar ───────────────────────────────────────────────────────
# Sync topnav toggle widget state
st.session_state["demo_toggle_widget"] = st.session_state["demo_active"]

# We adjust columns to include the Demo Mode toggle on the right
col_logo, col_nav, col_demo = st.columns([1.5, 3.5, 1.2])

with col_logo:
    logo_text = "⬡ CognifyAI (Show Menu)" if not st.session_state["show_nav"] else "⬡ CognifyAI (Hide Menu)"
    if st.button(logo_text, key="logo_toggle", use_container_width=True):
        st.session_state["show_nav"] = not st.session_state["show_nav"]
        st.rerun()

if st.session_state["show_nav"]:
    with col_nav:
        st.radio(
            "Top Navigation",
            list(PAGES.keys()),
            key="topnav_nav",
            on_change=on_topnav_change,
            horizontal=True,
            label_visibility="collapsed"
        )

with col_demo:
    # Render the Demo Mode toggle switch on the right side of the navbar
    st.toggle(
        "🚀 Demo Mode",
        key="demo_toggle_widget",
        on_change=on_demo_toggle
    )

st.markdown("---")


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
                next_page = steps[st.session_state["demo_step"]]
                st.session_state["selected_page"] = next_page
                st.session_state["topnav_nav"] = next_page
                st.rerun()
        else:
            if st.button("Finish Demo 🔁", use_container_width=True):
                st.session_state["demo_step"] = 0
                st.session_state["demo_active"] = False
                first_page = steps[0]
                st.session_state["selected_page"] = first_page
                st.session_state["topnav_nav"] = first_page
                st.rerun()

selected = st.session_state["selected_page"]
module_name = PAGES[selected]
import importlib
page_module = importlib.import_module(f"dashboard.pages.{module_name}")
page_module.render()
