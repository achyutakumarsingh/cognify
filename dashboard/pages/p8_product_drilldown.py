"""
dashboard/pages/p8_product_drilldown.py
Page 8 – Product Drill-Down
"""
from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
from dashboard.components.ui import section_header, insight_box, risk_badge, RISK_COLOURS
from dashboard.data_loader import load_business_data


def render():
    section_header(
        "🔍 Product Drill-Down",
        "Deep-dive into any product's forecast, risk, and business impact."
    )

    df = load_business_data()

    # ─── Search ───────────────────────────────────────────────────────────────
    col_s, col_t = st.columns([2, 1])
    search_term = col_s.text_input("🔍 Search Product ID", placeholder="e.g. FOODS_1_001", key="pd_search")
    store_opts  = ["All"] + sorted(df["store_id"].unique())
    store_sel   = col_t.selectbox("📍 Store", store_opts, key="pd_store")

    if search_term:
        mask = df["item_id"].str.contains(search_term.strip().upper(), case=False)
        if store_sel != "All":
            mask &= df["store_id"] == store_sel
        dff = df[mask]
    else:
        # Default: show first item
        if store_sel != "All":
            dff = df[df["store_id"] == store_sel]
        else:
            dff = df
        first_item = dff["item_id"].iloc[0] if len(dff) > 0 else None
        if first_item:
            dff = dff[dff["item_id"] == first_item]

    if dff.empty:
        st.warning("No products found for this search.")
        return

    # If multiple items match, let user pick one
    unique_items = dff["item_id"].unique()
    if len(unique_items) > 1:
        sel_item = st.selectbox("Multiple matches found — select product:", sorted(unique_items))
        dff = dff[dff["item_id"] == sel_item]
    else:
        sel_item = unique_items[0]

    # If multiple stores, pick one
    unique_stores = dff["store_id"].unique()
    if len(unique_stores) > 1:
        sel_store = st.selectbox("Select store:", sorted(unique_stores))
        dff = dff[dff["store_id"] == sel_store]
    else:
        sel_store = unique_stores[0]

    dff = dff.reset_index(drop=True)

    # ─── Header summary ───────────────────────────────────────────────────────
    risk_level = dff["Risk_Level"].mode()[0]
    risk_score = dff["Risk_Score"].mean()
    root_cause = dff["Root_Cause"].mode()[0]
    action     = dff["Recommended_Action"].mode()[0]

    st.markdown(f"""
    <div style="background:#1e1b4b;border-radius:12px;padding:20px;margin-bottom:1rem">
      <h3 style="color:#e0e7ff;margin:0">{sel_item} &nbsp; @ &nbsp; {sel_store}</h3>
      <p style="color:#a5b4fc;margin:4px 0 12px">Supply Chain Risk Profile</p>
      <div style="display:flex;gap:16px;flex-wrap:wrap">
        <span style="color:#c7d2fe">Risk Level: {risk_badge(risk_level)}</span>
        <span style="color:#c7d2fe">Risk Score: <b style="color:#818cf8">{risk_score:.1f}</b>/100</span>
        <span style="color:#c7d2fe">Records: <b>{len(dff)}</b> periods</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── KPIs ─────────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Demand",    f"{dff['actual'].mean():.2f} units")
    c2.metric("Avg Forecast",  f"{dff['point'].mean():.2f} units")
    c3.metric("MAE",           f"{(dff['actual'] - dff['point']).abs().mean():.2f}")
    c4.metric("Rec. Stock",    f"{dff['Recommended_Stock_Level'].mean():.2f} units")
    c5.metric("Avg Savings",   f"${(dff['Baseline_Total_Cost'] - dff['Proposed_Total_Cost']).mean():.2f}")

    # ─── Forecast chart ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📈 Historical Demand, Forecast & Prediction Interval")
    x = list(range(len(dff)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=dff["actual"].tolist(), mode="lines+markers",
                             name="Actual", line=dict(color="#818cf8", width=2)))
    fig.add_trace(go.Scatter(x=x, y=dff["point"].tolist(), mode="lines",
                             name="Forecast", line=dict(color="#fbbf24", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=dff["upper_sym"].tolist(), mode="lines",
                             name="90% Upper", line=dict(color="#4ade80", width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=dff["lower_sym"].tolist(), mode="lines",
                             fill="tonexty", fillcolor="rgba(74,222,128,0.07)",
                             name="90% Lower", line=dict(color="#4ade80", width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=dff["Recommended_Stock_Level"].tolist(), mode="lines",
                             name="Rec. Inventory", line=dict(color="#f472b6", width=2, dash="longdash")))
    fig.update_layout(
        paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
        font_color="#e0e7ff", height=380, margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(bgcolor="#1e1b4b"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── Recommendation box ───────────────────────────────────────────────────
    level_kind = {"Low": "success", "Medium": "warning", "High": "danger"}
    insight_box(
        f"<b>Root Cause:</b> {root_cause}<br><b>Recommended Action:</b> {action}",
        level_kind.get(risk_level, "info")
    )
