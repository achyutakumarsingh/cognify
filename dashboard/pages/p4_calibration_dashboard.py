"""
dashboard/pages/p4_calibration_dashboard.py
Page 4 – Calibration Dashboard
"""
from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from dashboard.components.ui import section_header, insight_box
from dashboard.data_loader import load_calibration_report, load_segment_analysis


def render():
    section_header(
        "📐 Calibration Dashboard",
        "Statistical calibration quality of prediction intervals."
    )

    report = load_calibration_report()
    df_seg = load_segment_analysis()

    curve    = report["conformal_calibration_curve"]
    qr_cal   = report.get("quantile_calibration", {})

    # ─── Calibration curve ────────────────────────────────────────────────────
    st.markdown("#### Reliability Diagram – Conformal vs Perfect Calibration")
    target_covs = [r["target_coverage"] for r in curve]
    empirical_covs = [r["empirical_coverage"] for r in curve]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Perfect Calibration", line=dict(color="#64748b", dash="dash", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=target_covs, y=empirical_covs, mode="lines+markers",
        name="Split Conformal", line=dict(color="#4ade80", width=3),
        marker=dict(size=10, color="#4ade80"),
    ))
    if qr_cal:
        fig.add_trace(go.Scatter(
            x=[0.9], y=[qr_cal["empirical_coverage"]], mode="markers",
            name=f"Quantile Regression (90%)",
            marker=dict(size=14, color="#fbbf24", symbol="diamond"),
        ))
    fig.update_layout(
        paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
        font_color="#e0e7ff", height=360,
        xaxis_title="Target Coverage", yaxis_title="Empirical Coverage",
        legend=dict(bgcolor="#1e1b4b"),
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    insight_box(
        "A well-calibrated model lies close to the diagonal. Points <b>below</b> the diagonal "
        "indicate under-coverage (intervals are too narrow). Points <b>above</b> indicate "
        "over-coverage (intervals are too wide, reducing sharpness).",
        "info"
    )

    # ─── Metric table ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📋 Calibration Metrics by Confidence Level")

    metric_rows = []
    for r in curve:
        metric_rows.append({
            "Method": "Split Conformal",
            "Target Coverage": f"{r['target_coverage']*100:.0f}%",
            "Empirical Coverage": f"{r['empirical_coverage']*100:.1f}%",
            "Coverage Error": f"{r['coverage_error']*100:.2f}%",
            "Avg Width": f"{r['avg_interval_width']:.3f}",
            "Winkler Score": f"{r['winkler_score']:.3f}",
        })
    if qr_cal:
        metric_rows.append({
            "Method": "Quantile Regression",
            "Target Coverage": "90%",
            "Empirical Coverage": f"{qr_cal['empirical_coverage']*100:.1f}%",
            "Coverage Error": f"{qr_cal['coverage_error']*100:.2f}%",
            "Avg Width": f"{qr_cal['avg_interval_width']:.3f}",
            "Winkler Score": f"{qr_cal['winkler_score']:.3f}",
        })
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
| Metric | Business Meaning |
|---|---|
| **Target Coverage** | The confidence level we asked the model to achieve (e.g., 90%). |
| **Empirical Coverage** | The actual % of true values that fell within the interval on the test set. |
| **Coverage Error** | How far off the model was (smaller is better). |
| **Avg Width** | The average interval size in units. Smaller means more precise (sharper). |
| **Winkler Score** | A combined penalty for both poor coverage and wide intervals. Lower is better. |
        """)

    # ─── Segment analysis ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🗂️ Coverage by Segment")
    segment_type = st.selectbox("Group By", ["store_id", "cat_id", "dept_id", "state_id"])
    df_filtered = df_seg[df_seg["Segment_Type"] == segment_type].copy()
    df_filtered["Coverage %"] = (df_filtered["Coverage"] * 100).round(1)

    col_l, col_r = st.columns(2)
    for method_name, col in [("Conformal", col_l), ("Quantile", col_r)]:
        dff = df_filtered[df_filtered["Method"] == method_name]
        with col:
            st.markdown(f"**{method_name}**")
            fig_s = go.Figure(go.Bar(
                x=dff["Segment_Name"].tolist(),
                y=dff["Coverage %"].tolist(),
                marker_color="#4f46e5",
                text=dff["Coverage %"].apply(lambda v: f"{v:.1f}%").tolist(),
                textposition="outside",
            ))
            fig_s.add_hline(y=90, line_dash="dash", line_color="#ef4444",
                            annotation_text="90% Target")
            fig_s.update_layout(
                paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
                font_color="#e0e7ff", height=300,
                margin=dict(t=20, b=20, l=20, r=20),
                yaxis=dict(range=[0, 110])
            )
            st.plotly_chart(fig_s, use_container_width=True)
