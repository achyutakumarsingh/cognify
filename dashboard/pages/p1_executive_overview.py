"""
dashboard/pages/p1_executive_overview.py
Page 1 – Executive Overview
"""
from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dashboard.components.ui import kpi_card, section_header, insight_box, RISK_COLOURS
from dashboard.data_loader import load_business_data, load_financial_summary, load_classification_report


def render():
    section_header(
        "🏢 Executive Overview",
        "Live operational intelligence summary for supply chain leadership."
    )

    df   = load_business_data()
    fin  = load_financial_summary()
    clf  = load_classification_report()

    # ─── KPI Row ──────────────────────────────────────────────────────────────
    total_products     = df["item_id"].nunique()
    high_risk_products = clf["risk_level_counts"].get("High", 0)
    net_savings        = fin["Impact"]["Net_Savings"]
    stockout_reduction = fin["Impact"]["Stockout_Reduction_Units"]
    svc_baseline       = fin["Baseline"]["Service_Level"] * 100
    svc_proposed       = fin["Proposed"]["Service_Level"] * 100
    svc_delta          = svc_proposed - svc_baseline
    roi                = fin["Impact"]["ROI"]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        kpi_card("Products Monitored", f"{total_products:,}", icon="📦", color="#818cf8")
    with c2:
        kpi_card("High-Risk Products", f"{high_risk_products:,}",
                 delta="Require immediate attention", icon="🔴", color="#f87171")
    with c3:
        kpi_card("Expected Cost Savings", f"${net_savings:,.2f}",
                 delta="Over evaluation window", icon="💰", color="#4ade80")
    with c4:
        kpi_card("Stockout Reduction", f"{stockout_reduction:,.0f} units",
                 delta="fewer lost-sale events", icon="📉", color="#4ade80")
    with c5:
        kpi_card("Service Level", f"{svc_proposed:.1f}%",
                 delta=f"▲ {svc_delta:.1f}% vs baseline", icon="⭐", color="#fbbf24")
    with c6:
        kpi_card("Penalty Mitigation Ratio", f"{roi:.2f}x",
                 delta="Penalty avoided per $1 safety stock", icon="📈", color="#34d399")

    # Add honest data scope label right under the KPIs
    st.markdown(
        "<div style='text-align:right;font-size:0.75rem;color:#64748b;margin-top:-10px'>"
        "⚠️ Metrics are evaluated on a representative backtest subset of 5,600 product-store testing periods over 28 days."
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Risk distribution donut ──────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### Risk Level Distribution")
        counts = clf["risk_level_counts"]
        labels = ["Low", "Medium", "High"]
        values = [counts.get(l, 0) for l in labels]
        colors = [RISK_COLOURS[l] for l in labels]

        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            marker_colors=colors,
            hole=0.55,
            textinfo="percent+label",
            textfont_size=14,
        ))
        fig.update_layout(
            paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
            font_color="#e0e7ff", showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20), height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### Cost Comparison: Baseline vs Proposed")
        metrics = ["Holding_Cost", "Stockout_Cost", "Total_Cost"]
        labels_m = ["Holding Cost", "Stockout Cost", "Total Cost"]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Baseline (Point Forecast)", x=labels_m,
            y=[fin["Baseline"][m] for m in metrics],
            marker_color="#64748b", text=[f"${fin['Baseline'][m]:,.2f}" for m in metrics],
            textposition="outside"
        ))
        fig2.add_trace(go.Bar(
            name="Proposed (Risk-Aware)", x=labels_m,
            y=[fin["Proposed"][m] for m in metrics],
            marker_color="#4f46e5", text=[f"${fin['Proposed'][m]:,.2f}" for m in metrics],
            textposition="outside"
        ))
        fig2.update_layout(
            barmode="group", paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
            font_color="#e0e7ff", margin=dict(t=20, b=20, l=20, r=20), height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ─── Executive Summary ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📋 Executive Briefing")
    
    summary_text = (
        f"<b>1. Daily Summary:</b> Today, the AI system scanned <b>{total_products:,} items</b> across all store locations. "
        f"The Risk Engine flagged <b>{high_risk_products:,} products</b> in critical status due to high demand volatility or "
        f"widened uncertainty bounds, requiring planner action.<br><br>"
        f"<b>2. Recommended Actions:</b> We advise supply chain planners to open the <b>Risk & Operations</b> tab, review "
        f"the high-risk list, and approve the recommended purchase order buffers. This will automatically increase inventory "
        f"levels to the Conformal 90% confidence upper bound for volatile items, preventing critical stockouts.<br><br>"
        f"<b>3. Operational & Financial Impact:</b> Migrating from traditional point-forecast replenishment to our risk-aware "
        f"policy is projected to reduce total inventory costs by <b>{fin['Impact']['Cost_Reduction_Pct']*100:.1f}%</b>, "
        f"eliminate <b>{stockout_reduction:,.0f} units of stockouts</b>, and increase the service level from "
        f"<b>{svc_baseline:.1f}% to {svc_proposed:.1f}%</b>.<br><br>"
        f"<b>4. Investment Efficiency:</b> For every $1 of additional safety stock holding cost invested, the business mitigates "
        f"<b>${roi:.2f} in stockout penalties</b> (avoided lost revenue and customer service fees), yielding a highly efficient "
        f"operational return."
    )
    insight_box(summary_text, "info")

    # ─── Risk by store ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🗺️ Risk Concentration by Store")
    store_risk = df.groupby("store_id")["Risk_Score"].mean().reset_index()
    store_risk.columns = ["Store", "Avg Risk Score"]
    fig3 = px.bar(
        store_risk.sort_values("Avg Risk Score", ascending=True),
        x="Avg Risk Score", y="Store", orientation="h",
        color="Avg Risk Score", color_continuous_scale="Reds",
    )
    fig3.update_layout(
        paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
        font_color="#e0e7ff", height=320,
        margin=dict(t=20, b=20, l=20, r=20),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ─── Known Limitations & Assumptions (Technical Credibility) ──────────────
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
