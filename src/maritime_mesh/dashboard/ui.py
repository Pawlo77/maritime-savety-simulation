"""Reusable UI helpers for consistent dashboard look and feel."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


def page_intro(title: str, description: str) -> None:
    """Render consistent page title/description block."""
    st.markdown(f"<div class='mm-page-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='mm-page-description'>{description}</div>", unsafe_allow_html=True)


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


def apply_plotly_theme(fig: go.Figure, *, height: int = 460) -> go.Figure:
    """Apply shared maritime-themed Plotly layout defaults."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        font={"family": "Inter, Segoe UI, system-ui, sans-serif", "size": 13, "color": "#0b1d2a"},
        paper_bgcolor="#f7fbfe",
        plot_bgcolor="#ffffff",
        colorway=[
            "#0077b6",
            "#00b4d8",
            "#48cae4",
            "#0096c7",
            "#023047",
            "#6d597a",
            "#ffb703",
        ],
        margin={"l": 24, "r": 24, "t": 56, "b": 24},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
            "bgcolor": "rgba(247,251,254,0.85)",
        },
        hoverlabel={"bgcolor": "#eaf6fc", "font": {"color": "#0b1d2a", "size": 12}},
    )
    return fig
