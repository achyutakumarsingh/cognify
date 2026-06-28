"""
dashboard/pages/p1_executive_overview.py
Page 1 – Executive Overview
"""
from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dashboard.components.ui import kpi_card, section_header, insight_box, apply_cognify_theme
from dashboard.data_loader import load_business_data, load_financial_summary, load_classification_report


def render():
    df   = load_business_data()
    fin  = load_financial_summary()
    clf  = load_classification_report()

    total_products     = df["item_id"].nunique()
    high_risk_products = clf["risk_level_counts"].get("High", 0)
    net_savings        = fin["Impact"]["Net_Savings"]
    stockout_reduction = fin["Impact"]["Stockout_Reduction_Units"]
    svc_baseline       = fin["Baseline"]["Service_Level"] * 100
    svc_proposed       = fin["Proposed"]["Service_Level"] * 100
    svc_delta          = svc_proposed - svc_baseline
    roi                = fin["Impact"]["ROI"]

    st.markdown(f"""
    <div style="
      background: linear-gradient(135deg, #1A1D2E 0%, #0F1117 100%);
      border: 1px solid rgba(79,107,237,0.3);
      border-radius: 16px;
      padding: 32px 36px;
      margin-bottom: 32px;
    ">
      <div style="font-size:12px;color:#4F6BED;font-weight:600;
        text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px">
        ⬡ Live Situation Report
      </div>
      <div style="font-size:32px;font-weight:600;color:#E8EAF0;line-height:1.2">
        {high_risk_products} SKUs at critical stockout risk
      </div>
      <div style="font-size:16px;color:#FF4D6A;margin-top:8px;font-weight:500">
        Potential losses detected in current evaluation window
      </div>
      <div style="font-size:14px;color:#9CA3C4;margin-top:12px;line-height:1.6">
        CognifyAI has analysed demand history across your SKU portfolio.
        Conformal prediction intervals are calibrated and ready. See recommended actions below.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(kpi_card(f"{high_risk_products}", "SKUs at critical risk", "Require attention", "down", "red"), unsafe_allow_html=True)
    with col2:
        st.markdown(kpi_card(f"{svc_proposed:.1f}%", "Forecast Service Level", f"+{svc_delta:.1f}% vs baseline", "up", "green"), unsafe_allow_html=True)
    with col3:
        st.markdown(kpi_card(f"${net_savings:,.0f}", "Projected savings", "vs current approach", "up", "default"), unsafe_allow_html=True)
    with col4:
        st.markdown(kpi_card(f"{roi:.2f}x", "Penalty Mitigation Ratio", "Target: > 1.5x", "up", "amber"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(section_header("Risk & Financial Overview"), unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        counts = clf["risk_level_counts"]
        labels = ["Low", "Medium", "High"]
        values = [counts.get(l, 0) for l in labels]
        colors = ["#2DD4A7", "#FFB547", "#FF4D6A"]

        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            marker_colors=colors,
            hole=0.55,
            textinfo="percent+label",
            textfont_size=14,
        ))
        fig = apply_cognify_theme(fig, "Risk Level Distribution")
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        metrics = ["Holding_Cost", "Stockout_Cost", "Total_Cost"]
        labels_m = ["Holding Cost", "Stockout Cost", "Total Cost"]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Baseline", x=labels_m,
            y=[fin["Baseline"][m] for m in metrics],
            marker_color="#9CA3C4", text=[f"${fin['Baseline'][m]:,.0f}" for m in metrics],
            textposition="outside"
        ))
        fig2.add_trace(go.Bar(
            name="Proposed", x=labels_m,
            y=[fin["Proposed"][m] for m in metrics],
            marker_color="#4F6BED", text=[f"${fin['Proposed'][m]:,.0f}" for m in metrics],
            textposition="outside"
        ))
        fig2 = apply_cognify_theme(fig2, "Cost Comparison")
        fig2.update_layout(barmode="group", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(insight_box(
        f"Migrating to our risk-aware policy is projected to reduce total inventory costs by <b>{fin['Impact']['Cost_Reduction_Pct']*100:.1f}%</b>, "
        f"eliminate <b>{stockout_reduction:,.0f} units of stockouts</b>, and increase service levels to <b>{svc_proposed:.1f}%</b>."
    ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(section_header("Risk Concentration by Store"), unsafe_allow_html=True)
    store_risk = df.groupby("store_id")["Risk_Score"].mean().reset_index()
    store_risk.columns = ["Store", "Avg Risk Score"]
    fig3 = px.bar(
        store_risk.sort_values("Avg Risk Score", ascending=True),
        x="Avg Risk Score", y="Store", orientation="h",
        color="Avg Risk Score", color_continuous_scale=["#1A1D2E", "#FF4D6A"],
    )
    fig3 = apply_cognify_theme(fig3, "Average Risk by Store")
    fig3.update_layout(height=320, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    with st.expander("🔬 Technical Appendix: Assumptions & Model Limitations"):
        st.markdown("""
        ### Current Model Assumptions & Limitations
        To maintain professional and technical honesty, the underlying mathematical and systems limits of this prototype are documented below:
        
        1. **Global Conformal Calibration:**
           - *Assumption:* The Split Conformal Prediction correction factor ($\\hat{q}$) is computed globally across the entire test partition residuals.
           - *Limitation:* It assumes residual variance is uniform (homoscedastic). In practice, high-volume products have different residual spreads than low-volume products. Locally adaptive conformal bands would improve individual SKU coverage metrics.
        
        2. **Quantile Model Assumptions:**
           - *Assumption:* Quantile Regression models for $q_{05}$ and $q_{95}$ are trained independently using Pinball Loss.
           - *Limitation:* There are no mathematical non-crossing constraints. In rare scenarios of severe volatility, the $q_{05}$ model could cross above the $q_{95}$ model.
           
        3. **No Active ERP Integration:**
           - *Assumption:* The action triggers (e.g. *Send to ERP*, *Approve PO*) are mock workflows.
           - *Limitation:* They generate simulated JSON payloads and toast notifications. In a production deployment, this would be wired directly to standard SAP or Oracle REST APIs via OAuth2.
           
        4. **Simulation Constraints:**
           - *Assumption:* The holding cost rate is a flat 5% of the product cost price, and the stockout penalty is a flat 40% (representing lost sales + brand damage penalties).
           - *Limitation:* Real-world holding and stockout penalties fluctuate by department and store location.
        """)
