"""
dashboard/pages/p5_risk_intelligence.py
Page 5 – Risk Intelligence
"""
from __future__ import annotations
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dashboard.components.ui import section_header, kpi_card, insight_box, RISK_COLOURS, RISK_EMOJI
from dashboard.data_loader import load_business_data, load_classification_report


def render():
    section_header(
        "⚠️ Risk Intelligence",
        "Composite risk scores, root causes, and operational recommendations."
    )

    df   = load_business_data()
    clf  = load_classification_report()

    # ─── Top KPIs ─────────────────────────────────────────────────────────────
    counts = clf["risk_level_counts"]
    boundaries = clf.get("score_boundaries", {})
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("🟢 Low Risk",    f"{counts.get('Low', 0):,}", color="#22c55e")
    with c2: kpi_card("🟡 Medium Risk", f"{counts.get('Medium', 0):,}", color="#f59e0b")
    with c3: kpi_card("🔴 High Risk",   f"{counts.get('High', 0):,}", color="#ef4444")
    with c4: kpi_card("Avg Risk Score", f"{df['Risk_Score'].mean():.1f}", color="#818cf8")

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Filters and sorting ──────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    store_opts = ["All"] + sorted(df["store_id"].unique())
    store_sel  = col_f1.selectbox("📍 Filter by Store", store_opts, key="ri_store")
    cat_opts   = ["All"] + sorted(df["cat_id"].unique())
    cat_sel    = col_f2.selectbox("📂 Filter by Category", cat_opts, key="ri_cat")
    level_opts = ["All", "High", "Medium", "Low"]
    level_sel  = col_f3.selectbox("⚠️ Filter by Risk Level", level_opts, key="ri_level")

    sort_by = st.selectbox("Sort by", ["Highest Risk Score", "Highest Savings Potential", "Largest Forecast Error"])

    mask = pd.Series([True] * len(df))
    if store_sel != "All": mask &= df["store_id"] == store_sel
    if cat_sel   != "All": mask &= df["cat_id"]   == cat_sel
    if level_sel != "All": mask &= df["Risk_Level"] == level_sel
    dff = df[mask].copy()

    if sort_by == "Highest Risk Score":
        dff = dff.sort_values("Risk_Score", ascending=False)
    elif sort_by == "Highest Savings Potential":
        dff["_savings"] = dff["Baseline_Total_Cost"] - dff["Proposed_Total_Cost"]
        dff = dff.sort_values("_savings", ascending=False)
    else:
        dff["_err"] = (dff["actual"] - dff["point"]).abs()
        dff = dff.sort_values("_err", ascending=False)

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    # ─── Risk score distribution ───────────────────────────────────────────────
    with col_l:
        st.markdown("#### Risk Score Distribution")
        fig = px.histogram(
            dff, x="Risk_Score", color="Risk_Level",
            color_discrete_map=RISK_COLOURS, nbins=40,
        )
        fig.update_layout(
            paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
            font_color="#e0e7ff", height=320,
            margin=dict(t=20, b=20, l=20, r=20),
            legend_title_text="Risk Level",
            legend=dict(bgcolor="#1e1b4b"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ─── Top high-risk items ───────────────────────────────────────────────────
    with col_r:
        st.markdown("#### Top 10 Highest Risk Items")
        top10 = dff.groupby("item_id")["Risk_Score"].mean().nlargest(10).reset_index()
        fig2 = go.Figure(go.Bar(
            x=top10["Risk_Score"].tolist(), y=top10["item_id"].tolist(),
            orientation="h", marker_color="#ef4444",
            text=top10["Risk_Score"].round(1).tolist(), textposition="outside",
        ))
        fig2.update_layout(
            paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
            font_color="#e0e7ff", height=320,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ─── Detail table ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Risk Intelligence Table")
    n_show = st.slider("Rows to display", 10, 100, 25, 5)

    display_cols = ["item_id", "store_id", "Risk_Score", "Risk_Level",
                    "Root_Cause", "Recommended_Action", "Intervention",
                    "Recommended_Stock_Level", "Confidence_Level"]
    df_display = dff[display_cols].head(n_show).copy()
    df_display.columns = ["Product", "Store", "Risk Score", "Risk Level",
                           "Root Cause", "Recommended Action", "Intervention",
                           "Rec. Stock Level", "Confidence %"]

    def _colour_row(row):
        c = RISK_COLOURS.get(row["Risk Level"], "transparent")
        return [f"background-color:{c}18"] * len(row)

    st.dataframe(
        df_display.style.apply(_colour_row, axis=1),
        use_container_width=True,
        height=420,
    )
