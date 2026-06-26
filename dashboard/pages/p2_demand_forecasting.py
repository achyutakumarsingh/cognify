"""
dashboard/pages/p2_demand_forecasting.py
Page 2 – Demand Forecasting
"""
from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from dashboard.components.ui import section_header, insight_box
from dashboard.data_loader import load_business_data, load_stage3_evaluation, load_feature_importance


def render():
    section_header(
        "📈 Demand Forecasting",
        "XGBoost-powered point forecast with actual demand comparison."
    )

    df = load_business_data()
    ev = load_stage3_evaluation()

    # ─── Model KPIs ───────────────────────────────────────────────────────────
    test_metrics = ev.get("test_metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("RMSE",  f"{test_metrics.get('RMSE', 0):.4f}")
    with c2: st.metric("MAE",   f"{test_metrics.get('MAE', 0):.4f}")
    with c3: st.metric("MAPE",  f"{test_metrics.get('MAPE', 0):.2f}%")
    with c4: st.metric("WAPE",  f"{test_metrics.get('WAPE', 0):.2f}%")

    st.markdown("---")

    # ─── Filters ──────────────────────────────────────────────────────────────
    stores = sorted(df["store_id"].unique())
    col_f1, col_f2 = st.columns(2)
    store_sel = col_f1.selectbox("📍 Select Store", stores, key="fc_store")

    items_in_store = sorted(df[df["store_id"] == store_sel]["item_id"].unique())
    item_sel = col_f2.selectbox("📦 Select Product", items_in_store, key="fc_item")

    # ─── Filter ───────────────────────────────────────────────────────────────
    mask = (df["store_id"] == store_sel) & (df["item_id"] == item_sel)
    dff = df[mask].reset_index(drop=True)

    if dff.empty:
        st.warning("No data for this selection.")
        return

    # ─── Forecast chart ───────────────────────────────────────────────────────
    st.markdown("#### Actual Demand vs Forecast")
    x = list(range(len(dff)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=dff["actual"].tolist(),
        mode="lines+markers", name="Actual Demand",
        line=dict(color="#818cf8", width=2),
        marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=dff["point"].tolist(),
        mode="lines", name="Point Forecast",
        line=dict(color="#fbbf24", width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=dff["upper_sym"].tolist(),
        mode="lines", name="90% Upper Bound",
        line=dict(color="#4ade80", width=1, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=dff["lower_sym"].tolist(),
        mode="lines", name="90% Lower Bound",
        fill="tonexty", fillcolor="rgba(74,222,128,0.08)",
        line=dict(color="#4ade80", width=1, dash="dash"),
    ))
    fig.update_layout(
        paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
        font_color="#e0e7ff",
        legend=dict(bgcolor="#1e1b4b", bordercolor="#4f46e5", borderwidth=1),
        height=380, margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── Residuals ────────────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    residuals = dff["actual"] - dff["point"]
    with col_l:
        st.markdown("#### Residuals Over Time")
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(
            x=x, y=residuals.tolist(), mode="lines+markers",
            line=dict(color="#f472b6", width=2), marker=dict(size=4),
            name="Residual"
        ))
        fig_r.add_hline(y=0, line_dash="dash", line_color="#64748b")
        fig_r.update_layout(
            paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
            font_color="#e0e7ff", height=260,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with col_r:
        st.markdown("#### Residual Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=residuals.tolist(), nbinsx=20,
            marker_color="#818cf8", opacity=0.8,
        ))
        fig_hist.update_layout(
            paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
            font_color="#e0e7ff", height=260,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ─── Summary stats ────────────────────────────────────────────────────────
    st.markdown("---")
    avg_actual = dff["actual"].mean()
    avg_forecast = dff["point"].mean()
    mae_item = residuals.abs().mean()
    insight_box(
        f"<b>Product: {item_sel} @ {store_sel}</b><br>"
        f"Average daily demand: <b>{avg_actual:.2f} units</b> &nbsp;|&nbsp; "
        f"Average forecast: <b>{avg_forecast:.2f} units</b> &nbsp;|&nbsp; "
        f"MAE: <b>{mae_item:.2f} units</b>",
        "info"
    )

    # ─── Feature Importance ───────────────────────────────────────────────────
    st.markdown("#### 🎯 Top Feature Importances (Global Model)")
    try:
        fi = load_feature_importance()
        # CSV has columns: feature, gain, weight
        top_fi = fi.nlargest(15, "gain")
        fig_fi = go.Figure(go.Bar(
            x=top_fi["gain"].tolist(),
            y=top_fi["feature"].tolist(),
            orientation="h",
            marker_color="#818cf8"
        ))
        fig_fi.update_layout(
            paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
            font_color="#e0e7ff", height=400,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    except Exception:
        pass
