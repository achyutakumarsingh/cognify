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

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        kpi_card("Products Monitored", f"{total_products:,}", icon="📦", color="#818cf8")
    with c2:
        kpi_card("High-Risk Products", f"{high_risk_products:,}",
                 delta="Require immediate attention", icon="🔴", color="#f87171")
    with c3:
        kpi_card("Expected Cost Savings", f"${net_savings:,.0f}",
                 delta="vs. baseline ordering", icon="💰", color="#4ade80")
    with c4:
        kpi_card("Stockout Reduction", f"{stockout_reduction:,.0f} units",
                 delta="fewer lost-sale events", icon="📉", color="#4ade80")
    with c5:
        kpi_card("Service Level", f"{svc_proposed:.1f}%",
                 delta=f"▲ {svc_delta:.1f}% vs baseline", icon="⭐", color="#fbbf24")
    with c6:
        roi = fin["Impact"]["ROI"]
        kpi_card("Implied ROI", f"{roi:.2f}x",
                 delta="per $1 additional safety stock", icon="📈", color="#34d399")

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
            name="Baseline", x=labels_m,
            y=[fin["Baseline"][m] for m in metrics],
            marker_color="#64748b", text=[f"${fin['Baseline'][m]:,.0f}" for m in metrics],
            textposition="outside"
        ))
        fig2.add_trace(go.Bar(
            name="Proposed", x=labels_m,
            y=[fin["Proposed"][m] for m in metrics],
            marker_color="#4f46e5", text=[f"${fin['Proposed'][m]:,.0f}" for m in metrics],
            textposition="outside"
        ))
        fig2.update_layout(
            barmode="group", paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
            font_color="#e0e7ff", margin=dict(t=20, b=20, l=20, r=20), height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ─── Executive Summary ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📋 Executive Summary")
    insight_box(
        f"<b>Today's Operational Status:</b> The AI forecasting system is monitoring "
        f"<b>{total_products:,} products</b> across all stores. The risk engine has identified "
        f"<b>{high_risk_products:,} high-risk products</b> requiring immediate planner attention."
        f"<br><br>"
        f"By switching from traditional point-forecast ordering to risk-aware intelligent replenishment, "
        f"the business is expected to achieve a <b>{fin['Impact']['Cost_Reduction_Pct']*100:.1f}% reduction "
        f"in total inventory costs</b>, eliminate <b>{stockout_reduction:,.0f} stockout units</b>, and "
        f"improve service levels from <b>{svc_baseline:.1f}%</b> to <b>{svc_proposed:.1f}%</b>."
        f"<br><br>"
        f"For every $1 of additional safety stock investment, the system returns "
        f"<b>${roi:.2f} in avoided stockout penalties</b> — a positive ROI at standard "
        f"supply chain economics.",
        "success"
    )

    # ─── Risk by store heatmap ────────────────────────────────────────────────
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
