"""
dashboard/pages/p9_explainability.py
Page 9 – Explainability
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
from dashboard.components.ui import section_header, insight_box, RISK_COLOURS, RISK_EMOJI
from dashboard.data_loader import load_business_data


EXPLANATIONS = {
    "Low": (
        "This product is classified as <b style='color:#22c55e'>🟢 LOW RISK</b>.\n"
        "<br><br>\n"
        "Demand is stable and predictable. Historical volatility is low and the forecasting model "
        "consistently achieves high accuracy for this product. The prediction interval is narrow, "
        "indicating high model confidence.\n"
        "<br><br>\n"
        "<b>Recommendation:</b> Order the predicted quantity. Standard replenishment process applies. "
        "No planner intervention is required."
    ),
    "Medium": (
        "This product is classified as <b style='color:#f59e0b'>🟡 MEDIUM RISK</b>.\n"
        "<br><br>\n"
        "The model detects moderate uncertainty for this product. This may be caused by moderate "
        "demand variability, recent price changes, or elevated prediction interval width relative to "
        "the product average. Calibration performance for this product segment is acceptable but not "
        "as strong as low-risk products.\n"
        "<br><br>\n"
        "<b>Recommendation:</b> Increase safety stock moderately to the 75th percentile of the "
        "uncertainty interval. Monitor demand weekly and review supplier lead time. No immediate "
        "escalation required but planners should keep this item on watch."
    ),
    "High": (
        "This product is classified as <b style='color:#ef4444'>🔴 HIGH RISK</b>.\n"
        "<br><br>\n"
        "The model has detected severe uncertainty for this product. One or more of the following "
        "conditions is present: <b>historically volatile demand patterns</b>, <b>exceptionally wide "
        "prediction intervals</b> (significantly above the product average), <b>large historical "
        "forecast errors</b>, or <b>degraded calibration quality</b> for this product's segment "
        "during volatile periods.\n"
        "<br><br>\n"
        "<b>Recommendation:</b> Increase safety stock aggressively to the 95th percentile of the "
        "uncertainty interval. Notify supply chain planners immediately. Escalate to supply chain "
        "manager for capacity review. Increase replenishment frequency to reduce exposure per order cycle. "
        "Increasing safety stock is estimated to reduce stockout probability by approximately "
        "<b>45–65%</b> for this category of product."
    ),
}


def render():
    section_header(
        "💬 Explainability",
        "Plain-English explanations for every recommendation — no technical jargon."
    )

    df = load_business_data()

    # ─── Product selector ──────────────────────────────────────────────────────
    stores = sorted(df["store_id"].unique())
    col_f1, col_f2 = st.columns(2)
    store_sel = col_f1.selectbox("📍 Store", stores, key="ex_store")
    items = sorted(df[df["store_id"] == store_sel]["item_id"].unique())
    item_sel = col_f2.selectbox("📦 Product", items, key="ex_item")

    mask = (df["store_id"] == store_sel) & (df["item_id"] == item_sel)
    dff = df[mask].reset_index(drop=True)

    if dff.empty:
        st.warning("No data for this selection.")
        return

    risk_level = dff["Risk_Level"].mode()[0]
    risk_score = dff["Risk_Score"].mean()
    root_cause = dff["Root_Cause"].mode()[0]
    action     = dff["Recommended_Action"].mode()[0]
    intervention = dff["Intervention"].mode()[0]

    # ─── Risk badge ───────────────────────────────────────────────────────────
    c = RISK_COLOURS.get(risk_level, "#64748b")
    e = RISK_EMOJI.get(risk_level, "⚪")
    st.markdown(
        f"""<div style="text-align:center;padding:24px;background:{c}22;
        border:2px solid {c}55;border-radius:16px;margin-bottom:24px">
          <div style="font-size:3rem">{e}</div>
          <h2 style="color:{c};margin:8px 0">{risk_level.upper()} RISK</h2>
          <p style="color:#e2e8f0;margin:0">Risk Score: <b>{risk_score:.1f}</b> / 100 &nbsp;|&nbsp; Product: {item_sel} @ {store_sel}</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # ─── Plain English explanation ─────────────────────────────────────────────
    st.markdown("#### 🗣️ What This Means For Your Business")
    insight_box(EXPLANATIONS.get(risk_level, ""), {"Low":"success","Medium":"warning","High":"danger"}.get(risk_level,"info"))

    # ─── Specific root cause ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔍 Specific Root Cause Detected by the AI")
    insight_box(f"<b>AI Analysis:</b> {root_cause}", "info")

    # ─── Recommended action ───────────────────────────────────────────────────
    st.markdown("#### ✅ Your Recommended Action")
    insight_box(
        f"<b>Primary Action:</b> {action}<br><b>Operational Escalation:</b> {intervention}",
        {"Low":"success","Medium":"warning","High":"danger"}.get(risk_level,"info")
    )

    # ─── Component breakdown ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 Risk Score Component Breakdown")
    st.caption("Each component contributes to the overall risk score (0 = no concern, 1 = maximum concern).")

    comp_cols = ["comp_width", "comp_error", "comp_volat", "comp_calib"]
    comp_labels = ["Interval Width", "Forecast Error", "Demand Volatility", "Calibration Quality"]

    # These are stored in Stage 6 scored data via the parquet file – check if available
    if all(c in dff.columns for c in comp_cols):
        comp_vals = [dff[c].mean() for c in comp_cols]
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
        st.info("Component-level breakdown not available in this dataset version.")
