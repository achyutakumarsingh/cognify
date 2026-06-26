"""
dashboard/components/ui.py
Reusable UI primitives: KPI cards, metric tables, colour helpers, etc.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd

# ─── colour palette ─────────────────────────────────────────────────────────
RISK_COLOURS = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}
RISK_EMOJI   = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

BRAND_PRIMARY   = "#4f46e5"   # indigo-600
BRAND_SURFACE   = "#1e1b4b"   # indigo-950
BRAND_ACCENT    = "#818cf8"   # indigo-400


def kpi_card(label: str, value: str, delta: str = "", icon: str = "📊",
             color: str = BRAND_ACCENT) -> None:
    """Render a single KPI tile."""
    delta_html = f"<p style='margin:0;font-size:0.75rem;color:#86efac'>{delta}</p>" if delta else ""
    st.markdown(
        f"""
        <div style="
            background:linear-gradient(135deg,#1e1b4b,#312e81);
            border:1px solid {color}40;
            border-radius:12px;
            padding:20px 16px;
            text-align:center;
            box-shadow:0 4px 24px #0004;
        ">
          <div style="font-size:1.8rem">{icon}</div>
          <p style="margin:6px 0 2px;font-size:0.78rem;color:#c7d2fe;text-transform:uppercase;letter-spacing:.05em">{label}</p>
          <p style="margin:0;font-size:1.7rem;font-weight:700;color:{color}">{value}</p>
          {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "") -> None:
    sub = f"<p style='color:#a5b4fc;margin:4px 0 0'>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div style="margin-bottom:1rem">
          <h2 style="color:#e0e7ff;margin:0">{title}</h2>
          {sub}
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(level: str) -> str:
    c = RISK_COLOURS.get(level, "#64748b")
    e = RISK_EMOJI.get(level, "⚪")
    return f'<span style="background:{c}22;color:{c};border:1px solid {c}55;border-radius:6px;padding:2px 10px;font-size:0.8rem;font-weight:600">{e} {level}</span>'


def styled_dataframe(df: pd.DataFrame, risk_col: str | None = None) -> None:
    """Render a dataframe with optional risk-level colour coding."""
    if risk_col and risk_col in df.columns:
        def _colour_row(row):
            c = RISK_COLOURS.get(row[risk_col], "transparent")
            return [f"background-color:{c}18"] * len(row)
        st.dataframe(df.style.apply(_colour_row, axis=1), use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)


def insight_box(text: str, kind: str = "info") -> None:
    """Render a highlighted insight box."""
    colours = {
        "info":    ("#312e81", "#818cf8"),
        "success": ("#14532d", "#4ade80"),
        "warning": ("#78350f", "#fbbf24"),
        "danger":  ("#7f1d1d", "#f87171"),
    }
    bg, border = colours.get(kind, colours["info"])
    st.markdown(
        f"<div style='background:{bg}44;border-left:4px solid {border};"
        f"border-radius:6px;padding:12px 16px;margin:8px 0;color:#e2e8f0'>"
        f"{text}</div>",
        unsafe_allow_html=True,
    )
