"""
dashboard/pages/p7_scenario_simulator.py
Page 7 – Interactive Scenario Simulator
"""
from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from dashboard.components.ui import section_header, insight_box
from dashboard.data_loader import load_business_data


def _compute_kpis(df, h_rate, s_rate, level_map):
    """Re-run financial model in memory with custom parameters."""
    df = df.copy()
    # Re-price with current parameters
    df["Baseline_HC"]  = df["Baseline_Overstock_Units"] * df["sell_price"] * h_rate
    df["Baseline_SC"]  = df["Baseline_Stockout_Units"]  * df["sell_price"] * s_rate
    df["Baseline_TC"]  = df["Baseline_HC"] + df["Baseline_SC"]

    df["Proposed_HC"]  = df["Proposed_Overstock_Units"] * df["sell_price"] * h_rate
    df["Proposed_SC"]  = df["Proposed_Stockout_Units"]  * df["sell_price"] * s_rate
    df["Proposed_TC"]  = df["Proposed_HC"] + df["Proposed_SC"]

    bl_svc = (df["Baseline_Stockout_Units"] == 0).mean()
    pr_svc = (df["Proposed_Stockout_Units"]  == 0).mean()

    return {
        "Baseline SL":      bl_svc * 100,
        "Proposed SL":      pr_svc * 100,
        "Baseline Cost":    df["Baseline_TC"].sum(),
        "Proposed Cost":    df["Proposed_TC"].sum(),
        "Net Savings":      df["Baseline_TC"].sum() - df["Proposed_TC"].sum(),
        "Cost Reduction %": (df["Baseline_TC"].sum() - df["Proposed_TC"].sum()) / max(df["Baseline_TC"].sum(), 1e-9) * 100,
    }


def render():
    section_header(
        "🔬 Scenario Simulator",
        "Adjust cost parameters and confidence levels to see dynamic recommendations."
    )

    df = load_business_data()

    st.markdown("#### ⚙️ Configure Scenario Parameters")
    col1, col2, col3, col4 = st.columns(4)
    holding_rate  = col1.slider("Holding Cost Rate (%)", 1, 20, 5) / 100
    stockout_rate = col2.slider("Stockout Penalty Rate (%)", 10, 150, 40) / 100
    confidence    = col3.slider("Confidence Level (%)", 50, 95, 90, step=5)
    vol_filter    = col4.select_slider("Demand Volatility Scenario",
                                       options=["Low", "Medium", "High", "All"],
                                       value="All")

    # ─── Filter data by volatility scenario ───────────────────────────────────
    if vol_filter == "All":
        df_sim = df
    else:
        q75 = df["rolling_std_28"].quantile(0.75)
        q50 = df["rolling_std_28"].quantile(0.50)
        if vol_filter == "High":
            df_sim = df[df["rolling_std_28"] >= q75]
        elif vol_filter == "Medium":
            df_sim = df[(df["rolling_std_28"] >= q50) & (df["rolling_std_28"] < q75)]
        else:
            df_sim = df[df["rolling_std_28"] < q50]

    kpis = _compute_kpis(df_sim, holding_rate, stockout_rate, {})

    # ─── KPI outputs ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 Simulated Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline Service Level",  f"{kpis['Baseline SL']:.1f}%")
    c2.metric("Proposed Service Level",  f"{kpis['Proposed SL']:.1f}%",
              f"▲ {kpis['Proposed SL'] - kpis['Baseline SL']:.1f}pp")
    c3.metric("Net Cost Savings",        f"${kpis['Net Savings']:,.0f}")
    c4.metric("Cost Reduction",          f"{kpis['Cost Reduction %']:.1f}%")

    # ─── Sweep chart – holding rate ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔁 Sensitivity: Net Savings vs Holding Cost Rate")
    h_range = np.arange(0.01, 0.21, 0.01)
    savings_curve = [_compute_kpis(df_sim, h, stockout_rate, {})["Net Savings"] for h in h_range]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=(h_range * 100).tolist(), y=savings_curve,
        mode="lines+markers", line=dict(color="#4ade80", width=2),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#ef4444", annotation_text="Break-even")
    fig.update_layout(
        paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
        font_color="#e0e7ff", height=320,
        xaxis_title="Holding Cost Rate (%)", yaxis_title="Net Savings ($)",
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── Recommendation at current settings ───────────────────────────────────
    st.markdown("---")
    if kpis["Net Savings"] > 0:
        insight_box(
            f"At a holding cost of <b>{holding_rate*100:.0f}%</b> and stockout penalty of "
            f"<b>{stockout_rate*100:.0f}%</b>, the Intelligent System generates <b>${kpis['Net Savings']:,.0f} "
            f"in net savings</b> and improves service level by <b>{kpis['Proposed SL']-kpis['Baseline SL']:.1f} "
            f"percentage points</b>. <b>Recommendation: Deploy the Intelligent System.</b>",
            "success"
        )
    else:
        insight_box(
            f"At the current cost parameters, holding costs outweigh the benefit of additional safety stock. "
            f"Consider reducing the confidence level or stock buffer before deploying.",
            "warning"
        )
