"""
dashboard/components/ui.py
Reusable UI primitives: KPI cards, metric tables, colour helpers, etc.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd

def apply_cognify_theme(fig, title=""):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#9CA3C4", size=12),
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=14, color="#E8EAF0"),
            x=0, xanchor='left', pad=dict(b=16)
        ),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.05)',
            linecolor='rgba(255,255,255,0.1)',
            tickfont=dict(size=11, color="#5C6180"),
            title_font=dict(size=12, color="#9CA3C4"),
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.05)',
            linecolor='rgba(255,255,255,0.1)',
            tickfont=dict(size=11, color="#5C6180"),
            title_font=dict(size=12, color="#9CA3C4"),
        ),
        legend=dict(
            bgcolor='rgba(26,29,46,0.9)',
            bordercolor='rgba(255,255,255,0.08)',
            borderwidth=1,
            font=dict(size=12, color="#9CA3C4"),
        ),
        margin=dict(l=0, r=0, t=48, b=0),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1A1D2E",
            bordercolor="#4F6BED",
            font=dict(family="Inter", size=12, color="#E8EAF0"),
        ),
    )
    return fig


def kpi_card(value, label, delta=None, delta_direction="up", color="default"):
    color_map = {
        "red": ("var(--red-alert)", "var(--red-dim)"),
        "amber": ("var(--amber-warn)", "var(--amber-dim)"),
        "green": ("var(--green-safe)", "var(--green-dim)"),
        "default": ("var(--cognify-blue)", "var(--cognify-blue-dim)"),
    }
    accent, bg = color_map.get(color, color_map["default"])
    delta_html = ""
    if delta:
        arrow = "↑" if delta_direction == "up" else "↓"
        cls = "cog-delta-up" if delta_direction == "up" else "cog-delta-down"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'
    
    return f"""
<div style="
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-top: 3px solid {accent};
  border-radius: 12px;
  padding: 20px 24px;
  height: 100%;
">
  <div class="cog-kpi-value" style="color:{accent}">{value}</div>
  <div class="cog-kpi-label">{label}</div>
  {delta_html}
</div>
"""

def risk_badge(level):
    config = {
        "CRITICAL": ("#FF4D6A", "rgba(255,77,106,0.15)", "⬤"),
        "HIGH":     ("#FFB547", "rgba(255,181,71,0.15)",  "⬤"),
        "MEDIUM":   ("#4F6BED", "rgba(79,107,237,0.15)", "⬤"),
        "LOW":      ("#2DD4A7", "rgba(45,212,167,0.15)", "⬤"),
        "URGENT":   ("#FF4D6A", "rgba(255,77,106,0.15)", "⬤"),
    }
    level_up = str(level).upper()
    color, bg, dot = config.get(level_up, config["LOW"])
    return f"""
<span style="
  background:{bg};
  color:{color};
  border:1px solid {color}40;
  border-radius:6px;
  padding:3px 10px;
  font-size:11px;
  font-weight:600;
  letter-spacing:0.5px;
">{dot} {level_up}</span>
"""

def insight_box(text, icon="💡"):
    return f"""
<div style="
  background: var(--cognify-blue-dim);
  border: 1px solid var(--border-accent);
  border-radius: 10px;
  padding: 14px 18px;
  margin: 16px 0;
  font-size: 14px;
  color: var(--text-secondary);
  line-height: 1.6;
">
  <span style="margin-right:8px">{icon}</span>{text}
</div>
"""

def section_header(title):
    return f"""
<div style="
  margin: 32px 0 16px 0;
  display: flex;
  align-items: center;
  gap: 12px;
">
  <span style="
    font-size:11px;
    font-weight:600;
    color:var(--text-muted);
    text-transform:uppercase;
    letter-spacing:1.4px;
    white-space:nowrap;
  ">{title}</span>
  <div style="flex:1;height:1px;background:var(--border)"></div>
</div>
"""
