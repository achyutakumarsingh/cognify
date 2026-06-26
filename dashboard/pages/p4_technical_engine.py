"""
dashboard/pages/p4_technical_engine.py
Page 4 – Technical Engine
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dashboard.components.ui import section_header, insight_box
from dashboard.data_loader import (load_business_data, load_stage3_evaluation, load_feature_importance,
                                   load_conformal_predictions, load_quantile_predictions,
                                   load_calibration_report, load_segment_analysis)


def render():
    section_header(
        "🧠 Technical Engine",
        "Deep-dive statistical validation of the underlying forecasting models and conformal calibration."
    )

    # Tabs for different technical areas
    t1, t2, t3, t4 = st.tabs([
        "📈 Point Forecast & Residuals",
        "🎯 Uncertainty Bounds",
        "📐 Calibration & Reliability",
        "🧬 Feature Importance"
    ])

    df = load_business_data()

    # ─── TAB 1: Point Forecasts & Residuals ───────────────────────────────────
    with t1:
        ev = load_stage3_evaluation()
        test_metrics = ev.get("test_metrics", {})
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE",  f"{test_metrics.get('RMSE', 0):.4f}")
        c2.metric("MAE",   f"{test_metrics.get('MAE', 0):.4f}")
        c3.metric("MAPE",  f"{test_metrics.get('MAPE', 0):.2f}%")
        c4.metric("WAPE",  f"{test_metrics.get('WAPE', 0):.2f}%")

        st.markdown("---")

        # Filters
        stores = sorted(df["store_id"].unique())
        col_f1, col_f2 = st.columns(2)
        store_sel = col_f1.selectbox("📍 Store Location", stores, key="te_store_t1")
        items_in_store = sorted(df[df["store_id"] == store_sel]["item_id"].unique())
        item_sel = col_f2.selectbox("📦 Product Select", items_in_store, key="te_item_t1")

        mask = (df["store_id"] == store_sel) & (df["item_id"] == item_sel)
        dff = df[mask].reset_index(drop=True)

        if dff.empty:
            st.warning("No data found.")
        else:
            st.markdown("#### Actual Demand vs Tweedie XGBoost Forecast")
            x = list(range(len(dff)))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=dff["actual"].tolist(), mode="lines+markers",
                                     name="Actual Demand", line=dict(color="#818cf8", width=2)))
            fig.add_trace(go.Scatter(x=x, y=dff["point"].tolist(), mode="lines",
                                     name="XGBoost Forecast", line=dict(color="#fbbf24", width=2, dash="dot")))
            fig.update_layout(
                paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
                font_color="#e0e7ff", height=320, margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(bgcolor="#1e1b4b"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Residuals
            st.markdown("##### Error & Residual Distribution")
            col_l, col_r = st.columns(2)
            residuals = dff["actual"] - dff["point"]
            with col_l:
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(x=x, y=residuals.tolist(), mode="lines+markers",
                                         line=dict(color="#f472b6", width=2), marker=dict(size=4),
                                         name="Residual"))
                fig_r.add_hline(y=0, line_dash="dash", line_color="#64748b")
                fig_r.update_layout(paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
                                    font_color="#e0e7ff", height=240, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_r, use_container_width=True)
            with col_r:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=residuals.tolist(), nbinsx=20, marker_color="#818cf8", opacity=0.8))
                fig_hist.update_layout(paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
                                       font_color="#e0e7ff", height=240, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_hist, use_container_width=True)

    # ─── TAB 2: Uncertainty Bounds ────────────────────────────────────────────
    with t2:
        df_conf = load_conformal_predictions()
        df_quant = load_quantile_predictions()

        confidence = st.slider("Desired Confidence Level (%)", 50, 95, 90, step=5, key="te_conf_slider")
        method = st.radio("Method Comparison", ["Split Conformal Prediction", "Quantile Regression"], horizontal=True)

        stores_t2 = sorted(df_conf["store_id"].unique())
        col_c1, col_c2 = st.columns(2)
        store_sel_t2 = col_c1.selectbox("📍 Store Location", stores_t2, key="te_store_t2")
        items_t2 = sorted(df_conf[df_conf["store_id"] == store_sel_t2]["item_id"].unique())
        item_sel_t2 = col_c2.selectbox("📦 Product Select", items_t2, key="te_item_t2")

        mask_c = (df_conf["store_id"] == store_sel_t2) & (df_conf["item_id"] == item_sel_t2)
        mask_q = (df_quant["store_id"] == store_sel_t2) & (df_quant["item_id"] == item_sel_t2)

        dfc = df_conf[mask_c].reset_index(drop=True)
        dfq = df_quant[mask_q].reset_index(drop=True)

        if not dfc.empty:
            report_t2 = load_calibration_report()
            curve_t2 = report_t2["conformal_calibration_curve"]
            target_cov = confidence / 100
            
            pts = sorted(curve_t2, key=lambda r: r["target_coverage"])
            widths = {r["target_coverage"]: r["avg_interval_width"] for r in pts}
            base_width_90 = widths.get(0.9, dfc["width_sym"].mean())
            target_width = next((r["avg_interval_width"] for r in pts if abs(r["target_coverage"] - target_cov) < 0.05), base_width_90)
            
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

            fig_ci = go.Figure()
            x_val = list(range(len(dfc)))
            fig_ci.add_trace(go.Scatter(x=x_val, y=dfc["actual"].tolist(), mode="lines+markers",
                                        name="Actual Demand", line=dict(color="#818cf8", width=2)))
            fig_ci.add_trace(go.Scatter(x=x_val, y=dfc["point"].tolist(), mode="lines",
                                        name="Point Forecast", line=dict(color="#fbbf24", width=2, dash="dot")))
            fig_ci.add_trace(go.Scatter(x=x_val, y=upper_plot, mode="lines",
                                        name=f"{confidence}% Upper", line=dict(color="#4ade80", width=1, dash="dash")))
            fig_ci.add_trace(go.Scatter(x=x_val, y=lower_plot, mode="lines",
                                        fill="tonexty", fillcolor="rgba(74,222,128,0.06)",
                                        name=f"{confidence}% Lower", line=dict(color="#4ade80", width=1, dash="dash")))
            fig_ci.update_layout(paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
                                 font_color="#e0e7ff", height=320, margin=dict(t=20, b=20, l=20, r=20),
                                 legend=dict(bgcolor="#1e1b4b"))
            st.plotly_chart(fig_ci, use_container_width=True)

            # Method Comparison Metrics Table
            st.markdown("##### Global Calibration Method Comparison")
            qr_row = report_t2.get("quantile_calibration", {})
            conf_row = next((r for r in curve_t2 if r["target_coverage"] == 0.9), {})
            
            comp_table = {
                "Metric": ["Target Coverage", "Empirical Coverage", "Avg Interval Width", "Dynamic Resizing", "Mathematical Guarantee"],
                "Split Conformal": ["90%", f"{conf_row.get('empirical_coverage', 0)*100:.1f}%", f"{conf_row.get('avg_interval_width', 0):.2f}", "✅ Yes", "✅ Yes"],
                "Quantile Regression": ["90%", f"{qr_row.get('empirical_coverage', 0)*100:.1f}%", f"{qr_row.get('avg_interval_width', 0):.2f}", "❌ No", "❌ No"],
            }
            st.dataframe(pd.DataFrame(comp_table), use_container_width=True, hide_index=True)

    # ─── TAB 3: Calibration & Reliability ─────────────────────────────────────
    with t3:
        report_t3 = load_calibration_report()
        df_seg_t3 = load_segment_analysis()
        curve_t3 = report_t3["conformal_calibration_curve"]
        qr_cal = report_t3.get("quantile_calibration", {})

        st.markdown("#### Reliability Diagram (Conformal vs perfect calibration)")
        target_covs = [r["target_coverage"] for r in curve_t3]
        empirical_covs = [r["empirical_coverage"] for r in curve_t3]

        fig_diag = go.Figure()
        fig_diag.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                      name="Perfect Calibration", line=dict(color="#64748b", dash="dash", width=2)))
        fig_diag.add_trace(go.Scatter(x=target_covs, y=empirical_covs, mode="lines+markers",
                                      name="Conformal Prediction", line=dict(color="#4ade80", width=3),
                                      marker=dict(size=10)))
        if qr_cal:
            fig_diag.add_trace(go.Scatter(x=[0.9], y=[qr_cal["empirical_coverage"]], mode="markers",
                                          name="Quantile Regression (90%)", marker=dict(size=14, color="#fbbf24", symbol="diamond")))
        fig_diag.update_layout(paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
                               font_color="#e0e7ff", height=320, xaxis_title="Target Coverage",
                               yaxis_title="Empirical Coverage", legend=dict(bgcolor="#1e1b4b"),
                               margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_diag, use_container_width=True)

        # Segment analysis
        st.markdown("---")
        st.markdown("#### Segment-wise Interval Coverage")
        segment_type = st.selectbox("Group By Segment", ["store_id", "cat_id", "dept_id", "state_id"])
        df_filtered = df_seg_t3[df_seg_t3["Segment_Type"] == segment_type].copy()
        df_filtered["Coverage %"] = (df_filtered["Coverage"] * 100).round(1)

        col_l, col_r = st.columns(2)
        for method_name, col in [("Conformal", col_l), ("Quantile", col_r)]:
            dff_seg = df_filtered[df_filtered["Method"] == method_name]
            with col:
                st.markdown(f"**{method_name} Coverage**")
                fig_s = go.Figure(go.Bar(
                    x=dff_seg["Segment_Name"].tolist(), y=dff_seg["Coverage %"].tolist(),
                    marker_color="#4f46e5", text=dff_seg["Coverage %"].apply(lambda v: f"{v:.1f}%").tolist(),
                    textposition="outside"
                ))
                fig_s.add_hline(y=90, line_dash="dash", line_color="#ef4444")
                fig_s.update_layout(paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
                                    font_color="#e0e7ff", height=260, margin=dict(t=15, b=15, l=15, r=15),
                                    yaxis=dict(range=[0, 110]))
                st.plotly_chart(fig_s, use_container_width=True)

    # ─── TAB 4: Feature Importance ────────────────────────────────────────────
    with t4:
        st.markdown("#### Global Feature Gain Importances")
        try:
            fi = load_feature_importance()
            top_fi = fi.nlargest(15, "gain")
            fig_fi = go.Figure(go.Bar(
                x=top_fi["gain"].tolist(), y=top_fi["feature"].tolist(),
                orientation="h", marker_color="#818cf8"
            ))
            fig_fi.update_layout(paper_bgcolor="#0f0c29", plot_bgcolor="#0f0c29",
                                 font_color="#e0e7ff", height=420, margin=dict(t=15, b=15, l=15, r=15))
            st.plotly_chart(fig_fi, use_container_width=True)
            
            insight_box(
                "Feature importances are derived globally from the Tweedie XGBoost point forecaster. "
                "Lagged features (rolling means) and holiday markers dominate prediction weights.",
                "info"
            )
        except Exception:
            st.info("Feature importance data not found.")
