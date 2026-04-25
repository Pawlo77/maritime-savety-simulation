"""Styling helpers for dashboard pages."""

import streamlit as st


def apply_dashboard_style() -> None:
    """Apply maritime-mesh inspired dashboard visual system."""
    st.markdown(
        """
        <style>
        :root {
            --mm-text: #0b1d2a;
            --mm-muted: #35506a;
            --mm-surface: #ffffff;
            --mm-surface-soft: #f5fbff;
            --mm-border: rgba(10, 44, 74, 0.18);
            --mm-accent: #0077b6;
            --mm-accent-dark: #005f8f;
            --mm-aqua: #00b4d8;
            --mm-navy: #023047;
            --mm-info-bg: #e9f7ff;
            --mm-success-bg: #ecfdf3;
            --mm-warning-bg: #fff7ed;
            --mm-error-bg: #fef2f2;
        }
        .stApp {
            background:
                radial-gradient(circle at 0% -30%, rgba(0, 180, 216, 0.10), rgba(0, 0, 0, 0) 45%),
                radial-gradient(circle at 100% 0%, rgba(2, 48, 71, 0.08), rgba(0, 0, 0, 0) 35%),
                #f7fbfe;
        }
        .block-container {
            /* Keep page content below top navigation bar. */
            padding-top: 4.25rem !important;
            padding-bottom: 1.8rem;
            max-width: 1380px;
        }
        h1, h2, h3, h4 {letter-spacing: -0.01em;}
        .mm-page-title {
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--mm-text);
            margin-bottom: 0.2rem;
            border-left: 4px solid var(--mm-accent);
            padding-left: 0.6rem;
        }
        .mm-page-description {
            color: var(--mm-muted);
            font-size: 0.97rem;
            margin-bottom: 1.05rem;
        }
        .mm-section-title {
            font-size: 1.08rem;
            font-weight: 650;
            color: var(--mm-text);
            margin-top: 0.4rem;
            margin-bottom: 0.25rem;
        }
        .mm-card {
            border: 1px solid var(--mm-border);
            border-radius: 12px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.7rem;
            background: linear-gradient(165deg, #ffffff 0%, #f2f9fe 100%);
            box-shadow: 0 8px 20px rgba(2, 48, 71, 0.08);
        }
        .mm-muted {color: var(--mm-muted); font-size: 0.92rem;}
        .mm-card-label {
            color: var(--mm-muted);
            font-size: 0.84rem;
            margin-bottom: 0.22rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .mm-card-value {
            color: var(--mm-text);
            font-size: 1.22rem;
            font-weight: 700;
            margin-bottom: 0.18rem;
        }
        .mm-panel {
            border-radius: 12px;
            border: 1px solid var(--mm-border);
            padding: 0.72rem 0.92rem;
            margin: 0.28rem 0 0.9rem 0;
            font-size: 0.92rem;
            color: var(--mm-text);
            background: rgba(255, 255, 255, 0.88);
        }
        .mm-panel-title {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--mm-muted);
            margin-bottom: 0.18rem;
            font-weight: 650;
        }
        .mm-panel-info {background: var(--mm-info-bg);}
        div[data-testid="stTabs"] button[role="tab"] {
            border-radius: 8px;
            color: var(--mm-muted);
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--mm-navy);
            background: rgba(0, 119, 182, 0.10);
        }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(90deg, var(--mm-accent), var(--mm-aqua));
            border: 1px solid var(--mm-accent-dark);
            color: #ffffff;
        }
        div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(90deg, var(--mm-accent-dark), var(--mm-accent));
        }
        .mm-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.16rem 0.58rem;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 0.35rem;
            margin-bottom: 0.25rem;
        }
        .mm-badge-success {background: var(--mm-success-bg); color: #166534;}
        .mm-badge-warning {background: var(--mm-warning-bg); color: #9a3412;}
        .mm-badge-error {background: var(--mm-error-bg); color: #991b1b;}
        </style>
        """,
        unsafe_allow_html=True,
    )
