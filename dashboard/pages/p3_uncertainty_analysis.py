"""
dashboard/pages/p3_uncertainty_analysis.py
Page 3 – Uncertainty Analysis
"""
from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from dashboard.components.ui import section_header, insight_box
from dashboard.data_loader import (load_conformal_predictions, load_quantile_predictions,
                                   load_calibration_report)


# q_hat from stage 4 – approximate from coverage curve data
def _conformal_interval(df, alpha: float):
    """
    Given a calibration dataset and desired coverage, approximate the conformal
    interval width by linearly interpolating the calibration curve.
    """
    report = load_calibration_report()
    curve = report["conformal_calibration_curve"]
    target = 1 - alpha
    # Find the entry whose target_coverage is closest
    closest = min(curve, key=lambda r: abs(r["target_coverage"] - target))
    factor = target / closest["target_coverage"] if closest["empirical_coverage"] > 0 else 1.0
    return factor


def render():
    section_header(
        "🎯 Uncertainty Analysis",
        "Prediction intervals from Quantile Regression and Conformal Prediction."
    )

    df_conf = load_conformal_predictions()
    df_quant = load_quantile_predictions()

    # ─── Confidence slider ────────────────────────────────────────────────────
    confidence = st.slider(
        "🎚️ Desired Confidence Level (%)", min_value=50, max_value=95, value=90, step=5,
        help="For Split Conformal, intervals are dynamically resized. For QR, the 90% model is shown."
    )
    alpha = 1 - confidence / 100

    # ─── Method selector ──────────────────────────────────────────────────────
    method = st.radio("Method", ["Split Conformal Prediction", "Quantile Regression"], horizontal=True)

    # ─── Store / Item filter ──────────────────────────────────────────────────
    stores = sorted(df_conf["store_id"].unique())
    col_f1, col_f2 = st.columns(2)
    store_sel = col_f1.selectbox("📍 Store", stores, key="unc_store")
    items = sorted(df_conf[df_conf["store_id"] == store_sel]["item_id"].unique())
    item_sel = col_f2.selectbox("📦 Product", items, key="unc_item")

    mask_c = (df_conf["store_id"] == store_sel) & (df_conf["item_id"] == item_sel)
    mask_q = (df_quant["store_id"] == store_sel) & (df_quant["item_id"] == item_sel)

    dfc = df_conf[mask_c].reset_index(drop=True)
    dfq = df_quant[mask_q].reset_index(drop=True)

    if dfc.empty:
        st.warning("No data found for this selection.")
        return

    x = list(range(len(dfc)))

    # ─── Conformal interval (scale by slider) ─────────────────────────────────
    # The stored width_sym was computed for 90% coverage.
    # We scale proportionally by the conformal calibration curve.
    report = load_calibration_report()
    curve = report["conformal_calibration_curve"]
    target_cov = confidence / 100
    # Interpolate q_hat scaling factor from the stored curve
    pts = sorted(curve, key=lambda r: r["target_coverage"])
    widths = {r["target_coverage"]: r["avg_interval_width"] for r in pts}
    base_width_90 = widths.get(0.9, dfc["width_sym"].mean())
    target_width = None
    for r in pts:
        if abs(r["target_coverage"] - target_cov) < 0.05:
            target_width = r["avg_interval_width"]
            break
    if target_width is None:
        target_width = base_width_90
    scale = target_width / base_width_90 if base_width_90 > 0 else 1.0
    lower_conf = (dfc["point"] - dfc["width_sym"] / 2 * scale).tolist()
    upper_conf = (dfc["point"] + dfc["width_sym"] / 2 * scale).tolist()

    if method == "Split Conformal Prediction":
        lower_plot = lower_conf
        upper_plot = upper_conf
        width_vals = dfc["width_sym"] * scale
    else:
        lower_plot = dfq["lower"].tolist() if not dfq.empty else lower_conf
        upper_plot = dfq["upper"].tolist() if not dfq.empty else upper_conf
        width_vals = dfq["width"] if not dfq.empty else dfc["width_sym"]

    # ─── Chart ────────────────────────────────────────────────────────────────
    st.markdown(f"#### {method} – {confidence}% Prediction Intervals")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=dfc["actual"].tolist(), mode="lines+markers",
        name="Actual", line=dict(color="#818cf8", width=2), marker=dict(size=4)
    ))
    fig.add_trace(go.Scatter(
        x=x, y=dfc["point"].tolist(), mode="lines",
        name="Point Forecast", line=dict(color="#fbbf24", width=2, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=x, y=upper_plot, mode="lines",
        name=f"{confidence}% Upper", line=dict(color="#4ade80", width=1, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=x, y=lower_plot, mode="lines",
        fill="tonexty", fillcolor="rgba(74,222,128,0.08)",
        name=f"{confidence}% Lower", line=dict(color="#4ade80", width=1, dash="dash")
    ))
    fig.update_layout(
        paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
        font_color="#e0e7ff",
        legend=dict(bgcolor="#1e1b4b"),
        height=380, margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── Summary metrics ──────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    # Empirical coverage for this item
    inside = ((dfc["actual"] >= lower_conf) & (dfc["actual"] <= upper_conf)).mean()
    col1.metric("Empirical Coverage", f"{inside*100:.1f}%", f"Target: {confidence}%")
    col2.metric("Avg Interval Width", f"{float(width_vals.mean()):.2f} units")
    col3.metric("Method", method.split(" ")[0])
    if method == "Split Conformal Prediction":
        col4.metric("Dynamic Confidence", "✅ Yes", "Zero retraining needed")
    else:
        col4.metric("Dynamic Confidence", "❌ No", "New model per confidence level")

    # ─── Method comparison ────────────────────────────────────────────────────
    st.markdown("#### 📊 Method Comparison Summary")
    qr_row = report.get("quantile_calibration", {})
    conf_row = next((r for r in curve if r["target_coverage"] == 0.9), {})

    comp = {
        "Metric": ["Target Coverage", "Empirical Coverage", "Avg Interval Width", "Dynamic Resizing", "Theoretical Guarantee"],
        "Split Conformal": [
            "90%",
            f"{conf_row.get('empirical_coverage', 0)*100:.1f}%",
            f"{conf_row.get('avg_interval_width', 0):.2f}",
            "✅ Yes",
            "✅ Yes",
        ],
        "Quantile Regression": [
            "90%",
            f"{qr_row.get('empirical_coverage', 0)*100:.1f}%",
            f"{qr_row.get('avg_interval_width', 0):.2f}",
            "❌ No",
            "❌ No",
        ],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)

    insight_box(
        "Split Conformal Prediction provides <b>theoretical marginal coverage guarantees</b> and "
        "supports <b>dynamic confidence resizing</b> at zero compute cost. Quantile Regression "
        "requires a complete model retraining for every new confidence level.",
        "info"
    )
