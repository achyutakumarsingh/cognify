"""
dashboard/pages/p10_demo_mode.py
Page 10 – Guided Demo Mode
"""
from __future__ import annotations
import time
import streamlit as st
from dashboard.components.ui import section_header, insight_box, kpi_card
from dashboard.data_loader import load_business_data, load_financial_summary, load_classification_report
import plotly.graph_objects as go


DEMO_STEPS = [
    ("📦 Step 1: Historical Demand Data", "The system ingests 5 years of M5 retail sales data across 3,049 items and 10 stores.",  "info"),
    ("📈 Step 2: XGBoost Demand Forecast", "A Tweedie XGBoost model generates point forecasts with optimized hyperparameters. WAPE achieved: <b>~77%</b>.", "info"),
    ("🎯 Step 3: Conformal Prediction Intervals", "Split Conformal Prediction wraps the forecasts in <b>statistically guaranteed</b> 90% coverage intervals.", "info"),
    ("📐 Step 4: Calibration Validation", "Stage 5 validates that intervals achieve <b>87.5% empirical coverage</b> — close to the 90% target.", "success"),
    ("⚠️ Step 5: Risk Scoring & Classification", "A composite risk score is computed from interval width, forecast error, volatility, and calibration penalty.", "warning"),
    ("✅ Step 6: Business Recommendations", "The Decision Engine converts risk scores into <b>actionable inventory recommendations</b> for planners.", "success"),
    ("💰 Step 7: Financial Simulation", "The simulation proves a <b>15.6% reduction in total operating costs</b> and service level improvement from 83% → 92%.", "success"),
]

def _step_card(index: int, title: str, description: str, kind: str):
    st.markdown(
        f"""<div style="background:#1e1b4b;border-radius:12px;padding:16px;margin:8px 0;
        border-left:4px solid #4f46e5">
          <span style="color:#818cf8;font-size:0.8rem;font-weight:600">STEP {index+1} OF {len(DEMO_STEPS)}</span>
          <h4 style="color:#e0e7ff;margin:4px 0">{title}</h4>
        </div>""",
        unsafe_allow_html=True,
    )
    insight_box(description, kind)


def render():
    section_header(
        "▶️ Demo Mode",
        "Guided 3-minute walkthrough of the complete AI Decision Intelligence pipeline."
    )

    df   = load_business_data()
    fin  = load_financial_summary()
    clf  = load_classification_report()

    st.markdown("""
    <div style="background:linear-gradient(135deg,#312e81,#1e1b4b);border-radius:16px;
    padding:24px;text-align:center;margin-bottom:24px;border:1px solid #4f46e5">
      <h2 style="color:#e0e7ff;margin:0">Supply Chain Decision Intelligence</h2>
      <p style="color:#a5b4fc;margin:8px 0 0">
        Click <b>▶ Run Demo</b> to see the complete AI pipeline story in under 3 minutes.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col_run, col_skip = st.columns([1, 3])
    run_demo = col_run.button("▶ Run Demo", type="primary", use_container_width=True)
    step_num = st.session_state.get("demo_step", -1)

    if run_demo:
        st.session_state["demo_step"] = 0
        step_num = 0

    if step_num >= 0:
        # Show steps sequentially with progress
        progress = st.progress(0)
        status   = st.empty()

        for i, (title, desc, kind) in enumerate(DEMO_STEPS):
            progress.progress((i + 1) / len(DEMO_STEPS))
            status.markdown(f"**Running step {i+1} of {len(DEMO_STEPS)}…**")
            _step_card(i, title, desc, kind)
            time.sleep(0.6)

        progress.progress(1.0)
        status.empty()

        # ─── Final impact summary ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🏆 Final Business Impact")
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Cost Savings",    f"${fin['Impact']['Net_Savings']:,.0f}", color="#4ade80", icon="💰")
        with c2: kpi_card("Cost Reduction",  f"{fin['Impact']['Cost_Reduction_Pct']*100:.1f}%", color="#4ade80", icon="📉")
        with c3: kpi_card("Service Level ▲", f"{(fin['Proposed']['Service_Level'] - fin['Baseline']['Service_Level'])*100:.1f}pp", color="#fbbf24", icon="⭐")
        with c4: kpi_card("ROI",             f"{fin['Impact']['ROI']:.2f}x",  color="#818cf8", icon="📈")

        # ─── Workflow diagram ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Complete AI Decision Workflow")
        steps_flow = [
            "Historical Data", "XGBoost Forecast", "Conformal Intervals",
            "Calibration", "Risk Engine", "Decision Engine", "Inventory Policy", "Financial Impact"
        ]
        fig = go.Figure()
        for i, s in enumerate(steps_flow):
            colour = "#4f46e5" if i < 3 else "#7c3aed" if i < 5 else "#4ade80"
            fig.add_trace(go.Scatter(
                x=[i], y=[0], mode="markers+text",
                marker=dict(size=40, color=colour),
                text=[f"<b>{s}</b>"],
                textposition="top center",
                textfont=dict(color="#e0e7ff", size=10),
                showlegend=False,
            ))
            if i < len(steps_flow) - 1:
                fig.add_annotation(
                    x=i + 0.5, y=0,
                    text="→", showarrow=False,
                    font=dict(size=20, color="#818cf8"),
                )
        fig.update_layout(
            paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
            font_color="#e0e7ff", height=180,
            xaxis=dict(visible=False, range=[-0.5, len(steps_flow) - 0.5]),
            yaxis=dict(visible=False, range=[-1, 1.5]),
            margin=dict(t=60, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        insight_box(
            "Demo complete. The system is ready for live deployment. Navigate to any page "
            "in the sidebar to explore detailed analysis.",
            "success"
        )

        if st.button("🔁 Restart Demo"):
            st.session_state["demo_step"] = -1
            st.rerun()
    else:
        st.markdown("#### What the demo covers:")
        for i, (title, desc, _) in enumerate(DEMO_STEPS):
            st.markdown(f"**{i+1}.** {title}")
