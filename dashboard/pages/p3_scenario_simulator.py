"""
dashboard/pages/p3_scenario_simulator.py
Page 3 – Scenario Simulator
"""
from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from dashboard.components.ui import section_header, insight_box, kpi_card, apply_cognify_theme
from dashboard.data_loader import load_business_data


def _compute_kpis(df, h_rate, s_rate, confidence, lead_time, volatility):
    df = df.copy()

    conf_scale = confidence / 90.0
    lt_scale = np.sqrt(lead_time / 1.0)
    vol_scale = volatility / 1.0
    
    total_scale = conf_scale * lt_scale * vol_scale

    df["Simulated_Baseline_Stockout"] = df["Baseline_Stockout_Units"] * vol_scale
    df["Simulated_Baseline_Overstock"] = df["Baseline_Overstock_Units"] * vol_scale
    
    df["Simulated_Proposed_Stockout"] = df["Simulated_Baseline_Stockout"] * np.exp(-1.2 * (total_scale - 0.5))
    df["Simulated_Proposed_Overstock"] = df["Simulated_Baseline_Overstock"] * (0.4 + 0.6 * total_scale)

    df["Baseline_HC"] = df["Simulated_Baseline_Overstock"] * df["sell_price"] * h_rate
    df["Baseline_SC"] = df["Simulated_Baseline_Stockout"] * df["sell_price"] * s_rate
    df["Baseline_TC"] = df["Baseline_HC"] + df["Baseline_SC"]

    df["Proposed_HC"] = df["Simulated_Proposed_Overstock"] * df["sell_price"] * h_rate
    df["Proposed_SC"] = df["Simulated_Proposed_Stockout"] * df["sell_price"] * s_rate
    df["Proposed_TC"] = df["Proposed_HC"] + df["Proposed_SC"]

    bl_svc = (df["Simulated_Baseline_Stockout"] < 0.5).mean()
    pr_svc = (df["Simulated_Proposed_Stockout"] < 0.5).mean()

    return {
        "Baseline SL":      bl_svc * 100,
        "Proposed SL":      pr_svc * 100,
        "Baseline Cost":    df["Baseline_TC"].sum(),
        "Proposed Cost":    df["Proposed_TC"].sum(),
        "Net Savings":      df["Baseline_TC"].sum() - df["Proposed_TC"].sum(),
        "Cost Reduction %": (df["Baseline_TC"].sum() - df["Proposed_TC"].sum()) / max(df["Baseline_TC"].sum(), 1e-9) * 100,
        "Stockout Prob":    1.0 - pr_svc,
        "Avg Shortfall":    df["Simulated_Proposed_Stockout"].mean() * df["sell_price"].mean() * s_rate
    }


def render():
    st.markdown('<div class="cog-title">Scenario Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:var(--text-muted);font-size:14px;margin-bottom:24px">Adjust cost structures, lead times, and confidence targets to run live business simulations.</div>', unsafe_allow_html=True)

    df = load_business_data()

    col1, col2, col3 = st.columns(3)
    holding_rate  = col1.slider("Holding Cost Rate (%)", 1, 20, 5) / 100
    stockout_rate = col2.slider("Stockout Penalty Rate (%)", 10, 150, 40) / 100
    confidence    = col3.slider("Target Confidence Level (%)", 50, 99, 90, step=5)

    col4, col5 = st.columns(2)
    lead_time     = col4.slider("Replenishment Lead Time (days)", 1, 14, 1)
    volatility    = col5.slider("Demand Volatility Multiplier", 0.5, 2.0, 1.0, step=0.1)

    kpis = _compute_kpis(df, holding_rate, stockout_rate, confidence, lead_time, volatility)

    p_stockout = kpis["Stockout Prob"]
    avg_shortfall = kpis["Avg Shortfall"]
    
    st.markdown(f"""
<div style="
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 16px;
  margin-bottom: 28px;
">
  <div style="background:var(--red-dim);border:1px solid rgba(255,77,106,0.25);
    border-radius:12px;padding:20px 24px">
    <div style="font-size:32px;font-weight:600;color:#FF4D6A">
      {p_stockout:.1%}
    </div>
    <div style="font-size:13px;color:var(--text-secondary);margin-top:4px">
      Probability of stockout<br>under current parameters
    </div>
  </div>
  <div style="background:var(--amber-dim);border:1px solid rgba(255,181,71,0.25);
    border-radius:12px;padding:20px 24px">
    <div style="font-size:32px;font-weight:600;color:#FFB547">
      ${kpis['Net Savings']/1000:,.1f}k
    </div>
    <div style="font-size:13px;color:var(--text-secondary);margin-top:4px">
      Projected net savings<br>vs baseline
    </div>
  </div>
  <div style="background:var(--green-dim);border:1px solid rgba(45,212,167,0.25);
    border-radius:12px;padding:20px 24px">
    <div style="font-size:32px;font-weight:600;color:#2DD4A7">
      {kpis['Proposed SL']:.1f}%
    </div>
    <div style="font-size:13px;color:var(--text-secondary);margin-top:4px">
      Simulated service level<br>with risk-aware policy
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ─── Sweep chart – holding rate ───────────────────────────────────────────
    st.markdown(section_header("Sensitivity Analysis"), unsafe_allow_html=True)
    h_range = np.arange(0.01, 0.21, 0.01)
    savings_curve = [_compute_kpis(df, h, stockout_rate, confidence, lead_time, volatility)["Net Savings"] for h in h_range]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=(h_range * 100).tolist(), y=savings_curve,
        mode="lines+markers", line=dict(color="#4F6BED", width=2.5),
        fill='tozeroy', fillcolor='rgba(79,107,237,0.1)'
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#FF4D6A", annotation_text="Break-even")
    fig = apply_cognify_theme(fig, "Net Savings vs Holding Cost Rate")
    fig.update_layout(height=320, xaxis_title="Holding Cost Rate (%)", yaxis_title="Net Savings ($)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(insight_box(
        f"In most simulated demand scenarios, the risk-aware policy is sufficient. "
        f"In scenarios where stockouts do occur, you face a potential shortfall averaging <strong style='color:#FF4D6A'>${avg_shortfall:,.0f}</strong>. "
        f"Deploying the intelligent policy at these parameters generates <strong style='color:#2DD4A7'>${kpis['Net Savings']:,.0f}</strong> in savings.",
        icon="🎲"
    ), unsafe_allow_html=True)
