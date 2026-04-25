"""Styling helpers for dashboard pages."""

import streamlit as st


def apply_dashboard_style() -> None:
    """Apply minimal style unification for readability."""
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2.0rem;}
        .mm-card {
            border: 1px solid rgba(120,120,120,0.25);
            border-radius: 10px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.6rem;
            background: rgba(250,250,250,0.45);
        }
        .mm-muted {color: #5f6368; font-size: 0.92rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
