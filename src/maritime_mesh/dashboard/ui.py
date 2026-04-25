"""Reusable UI helpers for consistent dashboard look and feel."""

from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _maritime_logo_svg() -> str:
    """Return inline SVG logo for the dashboard masthead."""
    return """
    <svg width="68" height="68" viewBox="0 0 68 68" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <defs>
        <linearGradient id="mmSea" x1="10" y1="8" x2="58" y2="60" gradientUnits="userSpaceOnUse">
          <stop stop-color="#72E2FF"/>
          <stop offset="1" stop-color="#1089B9"/>
        </linearGradient>
        <linearGradient id="mmHull" x1="16" y1="40" x2="52" y2="56" gradientUnits="userSpaceOnUse">
          <stop stop-color="#0E4060"/>
          <stop offset="1" stop-color="#061D2F"/>
        </linearGradient>
      </defs>
      <rect x="2.5" y="2.5" width="63" height="63" rx="15" fill="#051A2A" stroke="rgba(164, 224, 255, 0.44)"/>
      <path d="M17 45.5C17 45.5 23.5 41 34 41C44.5 41 51 45.5 51 45.5V49.5C51 49.5 44.5 54 34 54C23.5 54 17 49.5 17 49.5V45.5Z" fill="url(#mmHull)"/>
      <path d="M20 49.8C20 49.8 24.8 52.8 34 52.8C43.2 52.8 48 49.8 48 49.8" stroke="#8BDCFB" stroke-width="1.6" stroke-linecap="round"/>
      <path d="M34 16V41.2" stroke="#A5E9FF" stroke-width="2.1" stroke-linecap="round"/>
      <path d="M34.6 19L48.5 33C48.5 33 40.4 34 34.6 31V19Z" fill="url(#mmSea)"/>
      <path d="M33.4 23L21 36.8C21 36.8 27.7 37.4 33.4 35.1V23Z" fill="#3EC6EE"/>
      <path d="M17.5 57C17.5 57 22 54.2 27.2 57C32.3 59.8 36.7 57 36.7 57" stroke="#6CD4F4" stroke-opacity="0.9" stroke-width="1.6" stroke-linecap="round"/>
      <path d="M31.3 57C31.3 57 35.8 54.2 41 57C46.2 59.8 50.5 57 50.5 57" stroke="#6CD4F4" stroke-opacity="0.9" stroke-width="1.6" stroke-linecap="round"/>
    </svg>
    """.strip()  # noqa: E501


def page_intro(title: str, description: str) -> None:
    """Render consistent page title/description block with premium masthead."""
    safe_title = escape(title)
    safe_description = escape(description)
    st.markdown(
        (
            "<section class='mm-hero'>"
            "<div class='mm-hero-brand'>"
            f"<div class='mm-logo-wrap'>{_maritime_logo_svg()}</div>"
            "<div>"
            "<div class='mm-eyebrow'>Maritime Intelligence Suite</div>"
            f"<div class='mm-page-title'>{safe_title}</div>"
            f"<div class='mm-page-description'>{safe_description}</div>"
            "</div>"
            "</div>"
            "<div class='mm-hero-chip'>Enterprise Maritime Analytics</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def section_intro(title: str, description: str) -> None:
    """Render section heading and supporting description."""
    st.markdown(f"<div class='mm-section-title'>{title}</div>", unsafe_allow_html=True)
    st.caption(description)


def info_panel(title: str, body: str) -> None:
    """Render lightweight informational panel."""
    st.markdown(
        (
            "<div class='mm-panel mm-panel-info'>"
            f"<div class='mm-panel-title'>{title}</div>"
            f"<div>{body}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_dataframe(df: pd.DataFrame, *, width: str = "stretch", hide_index: bool = True) -> None:
    """Render dataframes as custom HTML tables matching page background."""
    classes = "mm-html-table"
    if width == "stretch":
        classes += " mm-html-table-fluid"
    table_html = df.to_html(index=not hide_index, escape=True, classes=classes, border=0)
    st.markdown(
        f"<div class='mm-html-table-wrap'>{table_html}</div>",
        unsafe_allow_html=True,
    )


def apply_plotly_theme(fig: go.Figure, *, height: int = 460) -> go.Figure:
    """Apply shared maritime-themed Plotly layout defaults."""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        font={"family": "Inter, Segoe UI, system-ui, sans-serif", "size": 13, "color": "#dbeeff"},
        title_font={"color": "#e8f6ff", "size": 20},
        paper_bgcolor="rgba(8, 36, 58, 0.0)",
        plot_bgcolor="rgba(9, 42, 66, 0.72)",
        colorway=[
            "#58c8ef",
            "#2fb0de",
            "#7cdfff",
            "#5d8cff",
            "#35d4a4",
            "#a178ff",
            "#f6b75f",
        ],
        margin={"l": 24, "r": 24, "t": 56, "b": 24},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
            "bgcolor": "rgba(8, 35, 56, 0.72)",
        },
        hoverlabel={"bgcolor": "#0a304c", "font": {"color": "#eaf5ff", "size": 12}},
        xaxis={
            "gridcolor": "rgba(140, 201, 234, 0.16)",
            "zerolinecolor": "rgba(140, 201, 234, 0.18)",
            "linecolor": "rgba(158, 216, 246, 0.28)",
            "tickfont": {"color": "#cae7fb"},
            "title": {"font": {"color": "#dff2ff"}},
        },
        yaxis={
            "gridcolor": "rgba(140, 201, 234, 0.16)",
            "zerolinecolor": "rgba(140, 201, 234, 0.18)",
            "linecolor": "rgba(158, 216, 246, 0.28)",
            "tickfont": {"color": "#cae7fb"},
            "title": {"font": {"color": "#dff2ff"}},
        },
    )
    fig.update_annotations(font={"color": "#dff2ff"})
    return fig
