"""
dashboard/pages/p2_risk_operations.py
Page 2 – Risk & Operations
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dashboard.components.ui import section_header, kpi_card, insight_box, risk_badge, apply_cognify_theme
from dashboard.data_loader import load_business_data, load_classification_report

EXPLANATIONS = {
    "Low": (
        "This product is classified as <b style='color:var(--green-safe)'>LOW RISK</b>.<br><br>"
        "Demand is stable and predictable. Historical volatility is low and the forecasting model "
        "consistently achieves high accuracy. The prediction interval is narrow, indicating high confidence.<br><br>"
        "<b>Recommendation:</b> Order the baseline predicted quantity. Standard replenishment applies. "
        "No planner intervention required."
    ),
    "Medium": (
        "This product is classified as <b style='color:var(--amber-warn)'>MEDIUM RISK</b>.<br><br>"
        "The model detects moderate uncertainty. This may be caused by moderate demand variability, "
        "recent price fluctuations, or elevated prediction interval width relative to the product average.<br><br>"
        "<b>Recommendation:</b> Increase safety stock moderately to the 75th percentile of the "
        "uncertainty interval. Monitor demand weekly and review supplier lead time."
    ),
    "High": (
        "This product is classified as <b style='color:var(--red-alert)'>HIGH RISK</b>.<br><br>"
        "The model has detected severe uncertainty. One or more of the following conditions is present: "
        "historically volatile demand patterns, exceptionally wide prediction intervals, or large historical forecast errors.<br><br>"
        "<b>Recommendation:</b> Increase safety stock aggressively to the 95th percentile of the "
        "uncertainty interval. Notify supply chain planners immediately and escalate to managers for capacity review. "
        "Increasing safety stock is estimated to reduce stockout probability by approximately <b>45–65%</b>."
    ),
}

def render():
    st.markdown('<div class="cog-title">Risk Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:var(--text-muted);font-size:14px;margin-bottom:24px">Triage inventory risks, inspect product details, and approve orders.</div>', unsafe_allow_html=True)

    df   = load_business_data()
    clf  = load_classification_report()

    # Create two tabs
    tab1, tab2 = st.tabs(["⚠️ Risk Triage & Action Center", "🔍 Single Product Analysis"])

    # ─── TAB 1: Risk Triage & Action Center ───────────────────────────────────
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)

        # Filters
        col_f1, col_f2, col_f3 = st.columns(3)
        store_opts = ["All"] + sorted(df["store_id"].unique())
        store_sel  = col_f1.selectbox("📍 Store Location", store_opts, key="ro_store")
        cat_opts   = ["All"] + sorted(df["cat_id"].unique())
        cat_sel    = col_f2.selectbox("📂 Category", cat_opts, key="ro_cat")
        level_opts = ["All", "High", "Medium", "Low"]
        level_sel  = col_f3.selectbox("⚠️ Risk Level", level_opts, key="ro_level")

        sort_by = st.selectbox("Sort Table By", ["Highest Financial Exposure", "Highest Risk Score"])

        # Filter logic
        mask = pd.Series([True] * len(df))
        if store_sel != "All": mask &= df["store_id"] == store_sel
        if cat_sel   != "All": mask &= df["cat_id"]   == cat_sel
        if level_sel != "All": mask &= df["Risk_Level"] == level_sel
        dff = df[mask].copy()

        dff["financial_exposure"] = (dff["Baseline_Total_Cost"] - dff["Proposed_Total_Cost"]).clip(lower=0)

        if sort_by == "Highest Financial Exposure":
            dff = dff.sort_values("financial_exposure", ascending=False)
        else:
            dff = dff.sort_values("Risk_Score", ascending=False)

        # Bulk actions
        st.markdown("##### ⚡ Actions")
        col_b1, col_b2, col_b3 = st.columns(3)
        if col_b1.button("✅ Approve All High-Risk Buffers (Demo)", use_container_width=True):
            st.toast("⚡ [DEMO] Pushed buffer modifications to ERP successfully!")
            st.success("Successfully synchronized inventory targets.")
        
        csv_data = dff.to_csv(index=False)
        with col_b2:
            st.download_button("⬇ Export Orders as CSV", data=csv_data, file_name="cognify_po_recommendations.csv", mime="text/csv", use_container_width=True)
            
        if col_b3.button("📂 Send Batch to Oracle ERP (Demo)", use_container_width=True):
            st.toast("⚡ [DEMO] Dispatched purchase orders payload to Oracle Cloud ERP REST Endpoint.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(section_header("High-Risk Exposure List"), unsafe_allow_html=True)
        
        n_show = st.slider("Display Limit", 10, 100, 20, 5)
        top_risk_skus = dff.head(n_show)

        for _, row in top_risk_skus.iterrows():
            exposure = row["financial_exposure"]
            prob = row["Risk_Score"] / 100.0  # Approx probability mapping for demo
            days_to_stockout = max(1, int(15 - (prob * 10))) # Simulated days
            
            border_color = '#FF4D6A' if row.Risk_Level == 'High' else ('#FFB547' if row.Risk_Level == 'Medium' else '#2DD4A7')
            
            st.markdown(f"""
<div style="
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-left: 4px solid {border_color};
  border-radius: 12px;
  padding: 18px 24px;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
">
  <div>
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
      <span style="font-weight:600;color:var(--text-primary);font-size:15px">
        {row.item_id} @ {row.store_id}
      </span>
      {risk_badge(row.Risk_Level)}
    </div>
    <div style="font-size:13px;color:var(--text-secondary)">
      Estimated {days_to_stockout} days to stockout · 
      Risk Score: <strong style="color:{border_color}">
      {row.Risk_Score:.0f}/100</strong>
    </div>
  </div>
  <div style="text-align:right">
    <div style="font-size:22px;font-weight:600;color:{border_color}">
      ${exposure:,.0f}
    </div>
    <div style="font-size:12px;color:var(--text-muted)">potential exposure</div>
  </div>
</div>
""", unsafe_allow_html=True)


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
            df_single = df[df["store_id"] == store_sel_single]
            first_item = df_single["item_id"].iloc[0] if len(df_single) > 0 else None
            if first_item:
                df_single = df_single[df_single["item_id"] == first_item]

        if df_single.empty:
            st.warning("No matching products found. Try searching for FOODS_1_001.")
        else:
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
            avg_savings = (df_single['Baseline_Total_Cost'] - df_single['Proposed_Total_Cost']).mean()

            # Purchase Order Card UI (Decision Engine)
            urgency = "URGENT" if risk_level_s == "High" else risk_level_s
            
            st.markdown(section_header("Recommended Actions"), unsafe_allow_html=True)
            
            st.markdown(f"""
<div style="
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 24px 28px;
  margin-bottom: 24px;
">
  <!-- Header row -->
  <div style="display:flex;justify-content:space-between;
    align-items:flex-start;margin-bottom:16px">
    <div>
      <div style="font-size:18px;font-weight:600;color:var(--text-primary)">
        {sel_item}
      </div>
      <div style="font-size:13px;color:var(--text-muted);margin-top:2px">
        Store: {store_sel_single} | {root_cause_s}
      </div>
    </div>
    {risk_badge(urgency)}
  </div>
  
  <!-- Decision row -->
  <div style="
    background: var(--surface-2);
    border-radius: 10px;
    padding: 16px 20px;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  ">
    <div>
      <div style="font-size:11px;color:var(--text-muted);
        text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">Order qty</div>
      <div style="font-size:22px;font-weight:600;color:var(--cognify-blue-light)">
        {df_single['Recommended_Stock_Level'].mean():.0f}
      </div>
      <div style="font-size:12px;color:var(--text-muted)">units</div>
    </div>
    <div>
      <div style="font-size:11px;color:var(--text-muted);
        text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">Order by</div>
      <div style="font-size:22px;font-weight:600;color:var(--text-primary)">
        Today
      </div>
      <div style="font-size:12px;color:var(--text-muted)">deadline</div>
    </div>
    <div>
      <div style="font-size:11px;color:var(--text-muted);
        text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">Risk Score</div>
      <div style="font-size:22px;font-weight:600;color:var(--amber-warn)">
        {risk_score_s:.0f}/100
      </div>
      <div style="font-size:12px;color:var(--text-muted)">AI confidence</div>
    </div>
    <div>
      <div style="font-size:11px;color:var(--text-muted);
        text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">Cost of delay</div>
      <div style="font-size:22px;font-weight:600;color:var(--red-alert)">
        ${avg_savings:,.0f}
      </div>
      <div style="font-size:12px;color:var(--text-muted)">if not ordered today</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

            # Simulated Actions
            col_a1, col_a2, col_a3 = st.columns(3)
            if col_a1.button("✅ Approve PO Recommendation", use_container_width=True, key="btn_ap_po"):
                st.toast(f"⚡ [DEMO] Safety Stock for {sel_item} approved at {df_single['Recommended_Stock_Level'].mean():.1f} units!")
            if col_a2.button("⚙️ Send PO to ERP", use_container_width=True, key="btn_send_po"):
                st.toast(f"⚡ [DEMO] Pushed purchase order for {sel_item} to SAP API successfully.")
            if col_a3.button("📝 Export Product PDF Report", use_container_width=True, key="btn_pdf_po"):
                st.toast(f"💾 [DEMO] Generated PDF risk profile report for {sel_item}.")

            st.markdown("<br>", unsafe_allow_html=True)

            # Plotly Chart
            st.markdown("#### 📈 Demand, Uncertainty Intervals & Inventory Level")
            x = list(range(len(df_single)))
            fig = go.Figure()
            
            # Forecast Line
            fig.add_trace(go.Scatter(x=x, y=df_single["point"].tolist(), mode="lines",
                                     name="Forecast", line=dict(color="#4F6BED", width=2.5, dash="solid")))
            
            # Actuals Line
            fig.add_trace(go.Scatter(x=x, y=df_single["actual"].tolist(), mode="lines",
                                     name="Actuals", line=dict(color="#E8EAF0", width=1.5)))
            
            # Prediction Interval (Opacity, not solid)
            fig.add_trace(go.Scatter(x=x, y=df_single["upper_sym"].tolist(), mode="lines",
                                     name="90% interval (Upper)", line=dict(color="rgba(79,107,237,0.0)", width=0)))
            fig.add_trace(go.Scatter(x=x, y=df_single["lower_sym"].tolist(), mode="lines",
                                     fill="tonexty", fillcolor="rgba(79,107,237,0.12)",
                                     name="90% interval", line=dict(color="rgba(79,107,237,0.3)", width=0.5)))
            
            # Reorder point / Safety stock / Current stock lines
            recommended_qty = df_single["Recommended_Stock_Level"].mean()
            current_qty = df_single["actual"].mean() * 0.8 # Synthetic current stock proxy
            
            fig.add_hline(
                y=recommended_qty,
                line_dash="dot",
                line_color="#2DD4A7",
                annotation_text="Safety stock",
                annotation_position="right",
                annotation_font_color="#2DD4A7"
            )
            fig.add_hline(
                y=current_qty,
                line_color="#FF4D6A",
                line_width=1.5,
                annotation_text=f"Current stock proxy",
                annotation_position="left",
                annotation_font_color="#FF4D6A"
            )

            fig = apply_cognify_theme(fig, "Demand vs Inventory Projections")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

            # Explainability & Components
            st.markdown("---")
            col_x1, col_x2 = st.columns(2)
            with col_x1:
                st.markdown("#### 💬 Plain-English Business Analysis")
                st.markdown(insight_box(EXPLANATIONS.get(risk_level_s, ""), "💡"), unsafe_allow_html=True)
                
            with col_x2:
                st.markdown("#### 📊 Risk Contribution Components")
                st.caption("Each component contributes to the overall risk score (0 = no concern, 1 = maximum concern).")
                comp_cols = ["comp_width", "comp_error", "comp_volat", "comp_calib"]
                comp_labels = ["Interval Width", "Forecast Inaccuracy", "Demand Volatility", "Calibration Quality"]
                
                if all(c in df_single.columns for c in comp_cols):
                    comp_vals = [df_single[c].mean() for c in comp_cols]
                    for lbl, val in zip(comp_labels, comp_vals):
                        pct = int(val * 100)
                        colour = "var(--green-safe)" if pct < 40 else "var(--amber-warn)" if pct < 70 else "var(--red-alert)"
                        st.markdown(
f"""<div style="margin:8px 0">
  <div style="display:flex;justify-content:space-between;margin-bottom:4px">
    <span style="color:var(--text-secondary);font-size:13px">{lbl}</span>
    <span style="color:{colour};font-weight:600;font-size:13px">{pct}%</span>
  </div>
  <div style="background:var(--surface-2);border-radius:4px;height:8px">
    <div style="width:{pct}%;background:{colour};height:100%;border-radius:4px"></div>
  </div>
</div>""",
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("Component-level breakdown not available.")
