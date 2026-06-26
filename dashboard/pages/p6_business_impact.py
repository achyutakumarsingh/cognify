"""
dashboard/pages/p6_business_impact.py
Page 6 – Business Impact
"""
from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from dashboard.components.ui import section_header, kpi_card, insight_box
from dashboard.data_loader import load_business_data, load_financial_summary, load_sensitivity_summary


def render():
    section_header(
        "💰 Business Impact",
        "Financial simulation: Baseline vs. Intelligent Risk-Aware System."
    )

    df  = load_business_data()
    fin = load_financial_summary()
    sens = load_sensitivity_summary()

    imp = fin["Impact"]
    bl  = fin["Baseline"]
    pr  = fin["Proposed"]

    # ─── KPI cards ────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Net Savings",       f"${imp['Net_Savings']:,.0f}", color="#4ade80", icon="💰")
    with c2: kpi_card("Cost Reduction",    f"{imp['Cost_Reduction_Pct']*100:.1f}%", color="#4ade80", icon="📉")
    with c3: kpi_card("Stockout Reduction",f"{imp['Stockout_Reduction_Units']:,.0f} units", color="#4ade80", icon="📦")
    with c4: kpi_card("Service Level ↑",  f"{(pr['Service_Level'] - bl['Service_Level'])*100:.1f}pp", color="#fbbf24", icon="⭐")
    with c5: kpi_card("ROI",               f"{imp['ROI']:.2f}x", color="#818cf8", icon="📈")

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Comparison table ──────────────────────────────────────────────────────
    st.markdown("#### 📋 KPI Comparison: Baseline vs. Proposed")
    comparison = pd.DataFrame([
        {"KPI": "Service Level",       "Baseline": f"{bl['Service_Level']*100:.1f}%",    "Proposed": f"{pr['Service_Level']*100:.1f}%",    "Change": f"▲ {(pr['Service_Level']-bl['Service_Level'])*100:.1f}pp"},
        {"KPI": "Fill Rate",           "Baseline": f"{bl['Fill_Rate']*100:.1f}%",         "Proposed": f"{pr['Fill_Rate']*100:.1f}%",         "Change": f"▲ {(pr['Fill_Rate']-bl['Fill_Rate'])*100:.1f}pp"},
        {"KPI": "Stockout Units",      "Baseline": f"{bl['Stockout_Units']:,.0f}",         "Proposed": f"{pr['Stockout_Units']:,.0f}",         "Change": f"▼ {bl['Stockout_Units']-pr['Stockout_Units']:,.0f}"},
        {"KPI": "Overstock Units",     "Baseline": f"{bl['Overstock_Units']:,.0f}",        "Proposed": f"{pr['Overstock_Units']:,.0f}",        "Change": ""},
        {"KPI": "Holding Cost ($)",    "Baseline": f"${bl['Holding_Cost']:,.2f}",          "Proposed": f"${pr['Holding_Cost']:,.2f}",          "Change": f"▲ ${pr['Holding_Cost']-bl['Holding_Cost']:,.2f}"},
        {"KPI": "Stockout Cost ($)",   "Baseline": f"${bl['Stockout_Cost']:,.2f}",         "Proposed": f"${pr['Stockout_Cost']:,.2f}",         "Change": f"▼ ${bl['Stockout_Cost']-pr['Stockout_Cost']:,.2f}"},
        {"KPI": "Total Op. Cost ($)",  "Baseline": f"${bl['Total_Cost']:,.2f}",            "Proposed": f"${pr['Total_Cost']:,.2f}",            "Change": f"▼ ${bl['Total_Cost']-pr['Total_Cost']:,.2f} ({imp['Cost_Reduction_Pct']*100:.1f}%)"},
    ])
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ─── Visual comparison ────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Cost Breakdown")
        metrics   = ["Holding Cost", "Stockout Cost", "Total Cost"]
        bl_vals   = [bl["Holding_Cost"], bl["Stockout_Cost"], bl["Total_Cost"]]
        prop_vals = [pr["Holding_Cost"], pr["Stockout_Cost"], pr["Total_Cost"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Baseline", x=metrics, y=bl_vals, marker_color="#64748b"))
        fig.add_trace(go.Bar(name="Proposed", x=metrics, y=prop_vals, marker_color="#4f46e5"))
        fig.update_layout(barmode="group", paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
                          font_color="#e0e7ff", height=320, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Service Level Comparison")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=["Baseline", "Proposed"],
            y=[bl["Service_Level"]*100, pr["Service_Level"]*100],
            marker_color=["#64748b", "#4f46e5"],
            text=[f"{bl['Service_Level']*100:.1f}%", f"{pr['Service_Level']*100:.1f}%"],
            textposition="outside",
        ))
        fig2.add_hline(y=90, line_dash="dash", line_color="#fbbf24", annotation_text="90% Target")
        fig2.update_layout(
            paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
            font_color="#e0e7ff", height=320,
            yaxis=dict(range=[0, 100]),
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ─── Scenario results ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🌍 Performance by Demand Scenario")
    scenarios = sens.get("Scenarios", {})
    if scenarios:
        sc_rows = []
        for name, data in scenarios.items():
            sc_rows.append({
                "Scenario": name.replace("_", " "),
                "Baseline Cost ($)":  f"${data['Baseline']['Total_Cost']:,.2f}",
                "Proposed Cost ($)":  f"${data['Proposed']['Total_Cost']:,.2f}",
                "Net Savings ($)":    f"${data['Impact']['Net_Savings']:,.2f}",
                "Baseline SL":        f"{data['Baseline']['Service_Level']*100:.1f}%",
                "Proposed SL":        f"{data['Proposed']['Service_Level']*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(sc_rows), use_container_width=True, hide_index=True)

    insight_box(
        f"The Intelligent System reduces total operating cost by <b>{imp['Cost_Reduction_Pct']*100:.1f}%</b> "
        f"by dynamically scaling safety stock to match each product's individual risk profile. "
        f"The incremental holding cost of <b>${imp['Extra_Holding_Cost']:,.0f}</b> is more than offset by "
        f"<b>${bl['Stockout_Cost'] - pr['Stockout_Cost']:,.0f}</b> in avoided stockout penalties.",
        "success"
    )
