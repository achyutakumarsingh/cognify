"""
dashboard/pages/p2_risk_operations.py
Page 2 – Risk & Operations
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dashboard.components.ui import section_header, kpi_card, insight_box, RISK_COLOURS, RISK_EMOJI, risk_badge
from dashboard.data_loader import load_business_data, load_classification_report

EXPLANATIONS = {
    "Low": (
        "This product is classified as <b style='color:#22c55e'>🟢 LOW RISK</b>.<br><br>"
        "Demand is stable and predictable. Historical volatility is low and the forecasting model "
        "consistently achieves high accuracy. The prediction interval is narrow, indicating high confidence.<br><br>"
        "<b>Recommendation:</b> Order the baseline predicted quantity. Standard replenishment applies. "
        "No planner intervention required."
    ),
    "Medium": (
        "This product is classified as <b style='color:#f59e0b'>🟡 MEDIUM RISK</b>.<br><br>"
        "The model detects moderate uncertainty. This may be caused by moderate demand variability, "
        "recent price fluctuations, or elevated prediction interval width relative to the product average.<br><br>"
        "<b>Recommendation:</b> Increase safety stock moderately to the 75th percentile of the "
        "uncertainty interval. Monitor demand weekly and review supplier lead time."
    ),
    "High": (
        "This product is classified as <b style='color:#ef4444'>🔴 HIGH RISK</b>.<br><br>"
        "The model has detected severe uncertainty. One or more of the following conditions is present: "
        "historically volatile demand patterns, exceptionally wide prediction intervals, or large historical forecast errors.<br><br>"
        "<b>Recommendation:</b> Increase safety stock aggressively to the 95th percentile of the "
        "uncertainty interval. Notify supply chain planners immediately and escalate to managers for capacity review. "
        "Increasing safety stock is estimated to reduce stockout probability by approximately <b>45–65%</b>."
    ),
}

def render():
    section_header(
        "🔴 Risk & Operations",
        "Triage inventory risks, inspect product details, and approve orders."
    )

    df   = load_business_data()
    clf  = load_classification_report()

    # Create two tabs: Triage and Deep-Dive
    tab1, tab2 = st.tabs(["⚠️ Risk Triage & Action Center", "🔍 Single Product Analysis"])

    # ─── TAB 1: Risk Triage & Action Center ───────────────────────────────────
    with tab1:
        # Top KPIs
        counts = clf["risk_level_counts"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 Low Risk Items", f"{counts.get('Low', 0):,}")
        c2.metric("🟡 Medium Risk Items", f"{counts.get('Medium', 0):,}")
        c3.metric("🔴 High Risk Items", f"{counts.get('High', 0):,}")
        c4.metric("Avg Risk Score", f"{df['Risk_Score'].mean():.1f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Filters
        st.markdown("#### 🔍 Filter & Search Inventory")
        col_f1, col_f2, col_f3 = st.columns(3)
        store_opts = ["All"] + sorted(df["store_id"].unique())
        store_sel  = col_f1.selectbox("📍 Store Location", store_opts, key="ro_store")
        cat_opts   = ["All"] + sorted(df["cat_id"].unique())
        cat_sel    = col_f2.selectbox("📂 Category", cat_opts, key="ro_cat")
        level_opts = ["All", "High", "Medium", "Low"]
        level_sel  = col_f3.selectbox("⚠️ Risk Level", level_opts, key="ro_level")

        sort_by = st.selectbox("Sort Table By", ["Highest Risk Score", "Highest Savings Potential"])

        # Filter logic
        mask = pd.Series([True] * len(df))
        if store_sel != "All": mask &= df["store_id"] == store_sel
        if cat_sel   != "All": mask &= df["cat_id"]   == cat_sel
        if level_sel != "All": mask &= df["Risk_Level"] == level_sel
        dff = df[mask].copy()

        if sort_by == "Highest Risk Score":
            dff = dff.sort_values("Risk_Score", ascending=False)
        else:
            dff["_savings"] = dff["Baseline_Total_Cost"] - dff["Proposed_Total_Cost"]
            dff = dff.sort_values("_savings", ascending=False)

        # Bulk actions
        st.markdown("##### ⚡ Actions")
        col_b1, col_b2, col_b3 = st.columns(3)
        if col_b1.button("✅ Approve All High-Risk Buffers (Demo)", use_container_width=True):
            st.toast("⚡ [DEMO] Pushed 838 high-risk buffer modifications to SAP ERP successfully!")
            st.success("Successfully synchronized inventory targets.")
        if col_b2.button("📝 Export Orders as CSV (Demo)", use_container_width=True):
            st.toast("💾 [DEMO] Generated replenishment_orders.csv containing 838 item recommendation rows!")
        if col_b3.button("📂 Send Batch to Oracle ERP (Demo)", use_container_width=True):
            st.toast("⚡ [DEMO] Dispatched purchase orders payload to Oracle Cloud ERP REST Endpoint.")

        # Dataframe
        st.markdown("<br>", unsafe_allow_html=True)
        n_show = st.slider("Display Limit", 10, 100, 20, 5)

        display_cols = ["item_id", "store_id", "Risk_Score", "Risk_Level",
                        "Root_Cause", "Recommended_Action", "Recommended_Stock_Level"]
        df_display = dff[display_cols].head(n_show).copy()
        df_display.columns = ["Product", "Store", "Risk Score", "Risk Level",
                               "Root Cause", "Recommended Action", "Rec. Stock Level"]

        def _colour_row(row):
            c = RISK_COLOURS.get(row["Risk Level"], "transparent")
            return [f"background-color:{c}18"] * len(row)

        st.dataframe(
            df_display.style.apply(_colour_row, axis=1),
            use_container_width=True,
            height=380,
            hide_index=True
        )

    # ─── TAB 2: Single Product Analysis ───────────────────────────────────────
    with tab2:
        col_s1, col_s2 = st.columns([2, 1])
        search_term = col_s1.text_input("🔍 Search SKU (e.g. FOODS_1_001)", placeholder="Enter SKU ID...", key="ro_sku_search")
        store_opts_single = sorted(df["store_id"].unique())
        store_sel_single = col_s2.selectbox("📍 Store Selection", store_opts_single, key="ro_store_single")

        if search_term:
            mask_single = df["item_id"].str.contains(search_term.strip().upper(), case=False)
            mask_single &= df["store_id"] == store_sel_single
            df_single = df[mask_single]
        else:
            # Fallback to first item matching store
            df_single = df[df["store_id"] == store_sel_single]
            first_item = df_single["item_id"].iloc[0] if len(df_single) > 0 else None
            if first_item:
                df_single = df_single[df_single["item_id"] == first_item]

        if df_single.empty:
            st.warning("No matching products found. Try searching for FOODS_1_001, HOBBIES_1_002, or HOUSEHOLD_1_001.")
        else:
            # If multiple items match, let user pick
            unique_items = df_single["item_id"].unique()
            if len(unique_items) > 1:
                sel_item = st.selectbox("Multiple SKUs matched. Pick one:", sorted(unique_items))
                df_single = df_single[df_single["item_id"] == sel_item]
            else:
                sel_item = unique_items[0]

            df_single = df_single.reset_index(drop=True)

            risk_level_s = df_single["Risk_Level"].mode()[0]
            risk_score_s = df_single["Risk_Score"].mean()
            root_cause_s = df_single["Root_Cause"].mode()[0]
            action_s     = df_single["Recommended_Action"].mode()[0]
            intervention_s = df_single["Intervention"].mode()[0]

            # Detail Box
            st.markdown(f"""
            <div style="background:#1e1b4b;border-radius:12px;padding:20px;margin-bottom:1rem;border:1px solid #4f46e533">
              <h3 style="color:#e0e7ff;margin:0">{sel_item} @ {store_sel_single}</h3>
              <p style="color:#a5b4fc;margin:4px 0 12px">Supply Chain Risk Profile & Decisions</p>
              <div style="display:flex;gap:16px;flex-wrap:wrap">
                <span style="color:#c7d2fe">Risk Classification: {risk_badge(risk_level_s)}</span>
                <span style="color:#c7d2fe">Risk Score: <b style="color:#818cf8">{risk_score_s:.1f}</b>/100</span>
                <span style="color:#c7d2fe">Records: <b>{len(df_single)}</b> periods</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Detail KPIs
            ck1, ck2, ck3, ck4, ck5 = st.columns(5)
            ck1.metric("Avg Demand", f"{df_single['actual'].mean():.2f} units")
            ck2.metric("Avg Forecast", f"{df_single['point'].mean():.2f} units")
            ck3.metric("MAE", f"{(df_single['actual'] - df_single['point']).abs().mean():.2f}")
            ck4.metric("Rec. Stock", f"{df_single['Recommended_Stock_Level'].mean():.2f} units")
            ck5.metric("Avg Savings", f"${(df_single['Baseline_Total_Cost'] - df_single['Proposed_Total_Cost']).mean():.2f}")

            # Plotly Chart
            st.markdown("#### 📈 Demand, Uncertainty Intervals & Inventory Level")
            x = list(range(len(df_single)))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=df_single["actual"].tolist(), mode="lines+markers",
                                     name="Actual Demand", line=dict(color="#818cf8", width=2)))
            fig.add_trace(go.Scatter(x=x, y=df_single["point"].tolist(), mode="lines",
                                     name="Point Forecast", line=dict(color="#fbbf24", width=2, dash="dot")))
            fig.add_trace(go.Scatter(x=x, y=df_single["upper_sym"].tolist(), mode="lines",
                                     name="90% Conformal Upper", line=dict(color="#4ade80", width=1, dash="dash")))
            fig.add_trace(go.Scatter(x=x, y=df_single["lower_sym"].tolist(), mode="lines",
                                     fill="tonexty", fillcolor="rgba(74,222,128,0.06)",
                                     name="90% Conformal Lower", line=dict(color="#4ade80", width=1, dash="dash")))
            fig.add_trace(go.Scatter(x=x, y=df_single["Recommended_Stock_Level"].tolist(), mode="lines",
                                     name="Recommended Safety Stock", line=dict(color="#f472b6", width=2, dash="longdash")))
            fig.update_layout(
                paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
                font_color="#e0e7ff", height=380, margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(bgcolor="#1e1b4b"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Simulated Actions
            st.markdown("##### ⚡ Actions for this Product")
            col_a1, col_a2, col_a3 = st.columns(3)
            if col_a1.button("✅ Approve PO Recommendation (Demo)", use_container_width=True, key="btn_ap_po"):
                st.toast(f"⚡ [DEMO] Safety Stock for {sel_item} approved at {df_single['Recommended_Stock_Level'].mean():.1f} units!")
            if col_a2.button("⚙️ Send PO to ERP (Demo)", use_container_width=True, key="btn_send_po"):
                st.toast(f"⚡ [DEMO] Pushed purchase order for {sel_item} to SAP API successfully.")
            if col_a3.button("📝 Export Product PDF Report (Demo)", use_container_width=True, key="btn_pdf_po"):
                st.toast(f"💾 [DEMO] Generated PDF risk profile report for {sel_item}.")

            # Explainability & Components
            st.markdown("---")
            col_x1, col_x2 = st.columns(2)
            with col_x1:
                st.markdown("#### 💬 Plain-English Business Analysis")
                insight_box(EXPLANATIONS.get(risk_level_s, ""), {"Low":"success","Medium":"warning","High":"danger"}.get(risk_level_s,"info"))
                
                st.markdown("#### 🔍 AI Root Cause & Recommendation")
                insight_box(f"<b>Root Cause:</b> {root_cause_s}<br><b>Escalation Path:</b> {intervention_s}", "info")

            with col_x2:
                st.markdown("#### 📊 Risk Contribution Components")
                st.caption("Each component contributes to the overall risk score (0 = no concern, 1 = maximum concern).")
                comp_cols = ["comp_width", "comp_error", "comp_volat", "comp_calib"]
                comp_labels = ["Interval Width", "Forecast Inaccuracy", "Demand Volatility", "Calibration Quality"]
                
                if all(c in df_single.columns for c in comp_cols):
                    comp_vals = [df_single[c].mean() for c in comp_cols]
                    for lbl, val in zip(comp_labels, comp_vals):
                        pct = int(val * 100)
                        colour = "#22c55e" if pct < 40 else "#f59e0b" if pct < 70 else "#ef4444"
                        st.markdown(
                            f"""<div style="margin:8px 0">
                              <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                                <span style="color:#c7d2fe">{lbl}</span>
                                <span style="color:{colour};font-weight:600">{pct}%</span>
                              </div>
                              <div style="background:#1e1b4b;border-radius:4px;height:10px">
                                <div style="width:{pct}%;background:{colour};height:100%;border-radius:4px"></div>
                              </div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("Component-level breakdown not available.")
