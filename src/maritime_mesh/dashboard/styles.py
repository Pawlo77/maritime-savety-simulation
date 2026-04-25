"""Styling helpers for dashboard pages."""

import streamlit as st


def apply_dashboard_style() -> None:
    """Apply premium maritime-inspired dashboard visual system."""
    st.markdown(
        """
        <style>
        :root {
            --mm-text: #dce8f4;
            --mm-muted: #9bb8d1;
            --mm-surface: rgba(7, 26, 44, 0.76);
            --mm-surface-soft: rgba(8, 34, 57, 0.68);
            --mm-border: rgba(147, 207, 243, 0.28);
            --mm-accent: #37b8e7;
            --mm-accent-dark: #1f90bc;
            --mm-aqua: #7ce0ff;
            --mm-navy: #04182b;
            --mm-info-bg: rgba(20, 89, 130, 0.26);
            --mm-success-bg: rgba(13, 111, 90, 0.30);
            --mm-warning-bg: rgba(151, 95, 28, 0.30);
            --mm-error-bg: rgba(130, 37, 37, 0.34);
            --primary-color: #35b8e7;
        }
        .stApp {
            background:
                radial-gradient(circle at 8% -12%, rgba(97, 206, 244, 0.22), rgba(0, 0, 0, 0) 38%),
                radial-gradient(circle at 95% -4%, rgba(26, 133, 190, 0.20), rgba(0, 0, 0, 0) 42%),
                radial-gradient(circle at 48% 112%, rgba(3, 70, 102, 0.42), rgba(0, 0, 0, 0) 46%),
                linear-gradient(175deg, #03111e 0%, #082238 46%, #0b2f46 100%);
            color: var(--mm-text);
            font-family: "Inter", "Segoe UI", system-ui, sans-serif;
        }
        .block-container {
            /* Keep page content below top navigation bar. */
            padding-top: 4.6rem !important;
            padding-bottom: 2rem;
            max-width: 1380px;
        }
        h1, h2, h3, h4 {
            letter-spacing: -0.01em;
            color: #e7f4ff !important;
        }
        h5, h6 {
            color: #dceeff !important;
        }
        p, li, label, span, div {
            color: inherit;
        }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {
            color: #d8ebfb !important;
        }
        [data-testid="stCaptionContainer"] {
            color: #bdd9ee !important;
        }
        [data-testid="stCaptionContainer"] p {
            color: #bdd9ee !important;
        }
        div[data-baseweb="tooltip"] {
            background: rgba(7, 29, 47, 0.96) !important;
            color: #eaf6ff !important;
            border: 1px solid rgba(126, 200, 236, 0.34) !important;
            border-radius: 10px !important;
            box-shadow: 0 10px 24px rgba(0, 7, 16, 0.34) !important;
        }
        div[data-baseweb="tooltip"] * {
            color: #eaf6ff !important;
        }
        [data-testid="stTooltipHoverTarget"] + div [role="tooltip"] {
            background: rgba(7, 29, 47, 0.96) !important;
            color: #eaf6ff !important;
            border: 1px solid rgba(126, 200, 236, 0.34) !important;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(9, 41, 64, 0.78), rgba(5, 25, 41, 0.70));
            border: 1px solid rgba(121, 195, 231, 0.24);
            border-radius: 12px;
            padding: 0.6rem 0.75rem;
        }
        [data-testid="stMetricLabel"] {
            color: #9fc5df !important;
            opacity: 1 !important;
            font-weight: 560 !important;
        }
        [data-testid="stMetricValue"] {
            color: #ecf7ff !important;
            opacity: 1 !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em;
        }
        [data-testid="stMetricDelta"] {
            opacity: 1 !important;
            font-weight: 600 !important;
        }
        div[data-baseweb="select"] label,
        div[data-baseweb="input"] label,
        .stSlider label,
        .stNumberInput label {
            color: #c9e4f8 !important;
            opacity: 1 !important;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(4, 24, 40, 0.94), rgba(6, 33, 54, 0.92));
            border-right: 1px solid rgba(147, 207, 243, 0.2);
        }
        [data-testid="stHeader"] {
            background: rgba(3, 16, 28, 0.72);
            border-bottom: 1px solid rgba(122, 190, 228, 0.18);
            backdrop-filter: blur(6px);
        }
        [data-testid="stNavigation"] {
            background: rgba(3, 17, 29, 0.70);
            border: 1px solid rgba(131, 199, 237, 0.2);
            border-radius: 14px;
            padding: 0.2rem 0.4rem 0.2rem 18.2rem;
            backdrop-filter: blur(6px);
            box-shadow: 0 12px 28px rgba(1, 7, 15, 0.32);
            position: relative;
            min-height: 58px;
        }
        [data-testid="stNavigation"]::before {
            content: "⚓  Maritime Mesh Command";
            position: absolute;
            left: 0.8rem;
            top: 50%;
            transform: translateY(-50%);
            display: inline-flex;
            align-items: center;
            height: 40px;
            padding: 0 0.9rem;
            border-radius: 10px;
            border: 1px solid rgba(145, 214, 247, 0.32);
            background: linear-gradient(135deg, rgba(10, 55, 84, 0.82), rgba(6, 34, 55, 0.78));
            color: #eaf7ff;
            font-size: 1.08rem;
            font-weight: 700;
            letter-spacing: 0.01em;
            white-space: nowrap;
        }
        [data-testid="stNavigation"] a {
            border-radius: 10px;
        }
        [data-testid="stNavigation"] a[aria-current="page"] {
            background: linear-gradient(100deg, rgba(56, 188, 234, 0.22), rgba(8, 126, 174, 0.18));
            border: 1px solid rgba(121, 210, 246, 0.32);
        }
        .mm-hero {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            border-radius: 18px;
            border: 1px solid var(--mm-border);
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            background: linear-gradient(140deg, rgba(10, 43, 69, 0.86), rgba(3, 22, 38, 0.78));
            box-shadow: 0 16px 34px rgba(1, 7, 16, 0.35);
        }
        .mm-hero-brand {
            display: flex;
            align-items: center;
            gap: 0.9rem;
        }
        .mm-logo-wrap {
            width: 68px;
            height: 68px;
            display: grid;
            place-items: center;
            border-radius: 16px;
            background: radial-gradient(circle at 32% 20%, rgba(129, 229, 255, 0.32), rgba(0, 0, 0, 0) 46%);
            border: 1px solid rgba(154, 220, 248, 0.30);
            flex-shrink: 0;
        }
        .mm-eyebrow {
            color: #8bccea;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.69rem;
            font-weight: 650;
            margin-bottom: 0.2rem;
        }
        .mm-page-title {
            font-size: 1.5rem;
            font-weight: 730;
            color: #f0f8ff;
            margin-bottom: 0.25rem;
        }
        .mm-page-description {
            color: var(--mm-muted);
            font-size: 0.94rem;
            margin-bottom: 0;
            max-width: 740px;
        }
        .mm-hero-chip {
            border-radius: 999px;
            border: 1px solid rgba(146, 213, 246, 0.38);
            background: rgba(16, 98, 139, 0.30);
            color: #bfe8ff;
            font-size: 0.74rem;
            font-weight: 640;
            padding: 0.3rem 0.6rem;
            white-space: nowrap;
        }
        .mm-section-title {
            font-size: 1.08rem;
            font-weight: 650;
            color: #dcecff;
            margin-top: 0.4rem;
            margin-bottom: 0.25rem;
        }
        .mm-card {
            border: 1px solid var(--mm-border);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.7rem;
            background: linear-gradient(165deg, rgba(8, 42, 67, 0.94), rgba(5, 27, 44, 0.88));
            box-shadow: 0 12px 28px rgba(0, 5, 13, 0.3);
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
            color: #f0f8ff;
            font-size: 1.22rem;
            font-weight: 700;
            margin-bottom: 0.18rem;
        }
        .mm-panel {
            border-radius: 13px;
            border: 1px solid var(--mm-border);
            padding: 0.72rem 0.92rem;
            margin: 0.28rem 0 0.9rem 0;
            font-size: 0.92rem;
            color: var(--mm-text);
            background: rgba(10, 44, 70, 0.62);
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
            border: 1px solid transparent;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #dff2ff;
            background: rgba(43, 156, 206, 0.18);
            border: 1px solid rgba(115, 202, 238, 0.34);
        }
        .stDataFrame, div[data-testid="stTable"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(121, 195, 231, 0.24);
        }
        [data-testid="stDataFrame"] div[role="grid"],
        [data-testid="stDataFrame"] div[role="rowgroup"],
        [data-testid="stDataFrame"] div[role="row"],
        [data-testid="stDataFrame"] div[role="columnheader"],
        [data-testid="stDataFrame"] div[role="gridcell"] {
            background: rgba(15, 68, 101, 0.86) !important;
            color: #eaf7ff !important;
            border-color: rgba(115, 191, 226, 0.20) !important;
        }
        [data-testid="stDataFrame"] div[role="columnheader"] {
            background: rgba(20, 78, 114, 0.94) !important;
            color: #e6f6ff !important;
            font-weight: 620 !important;
        }
        [data-testid="stDataFrame"] * {
            color: #eaf7ff !important;
        }
        div[data-testid="stTable"] table,
        div[data-testid="stTable"] thead,
        div[data-testid="stTable"] tbody,
        div[data-testid="stTable"] tr,
        div[data-testid="stTable"] th,
        div[data-testid="stTable"] td {
            background: rgba(15, 68, 101, 0.86) !important;
            color: #eaf7ff !important;
            border-color: rgba(115, 191, 226, 0.22) !important;
        }
        div[data-testid="stTable"] th {
            background: rgba(20, 78, 114, 0.94) !important;
            color: #e6f6ff !important;
            font-weight: 620 !important;
        }
        div[data-testid="stPlotlyChart"] {
            border-radius: 14px;
            border: 1px solid rgba(121, 195, 231, 0.22);
            background: linear-gradient(160deg, rgba(7, 35, 56, 0.80), rgba(4, 23, 39, 0.74));
            box-shadow: 0 10px 24px rgba(0, 6, 15, 0.26);
            padding: 0.2rem 0.3rem;
        }
        div.stButton > button {
            border-radius: 10px;
            font-weight: 620;
            border: 1px solid rgba(118, 195, 231, 0.28);
            background: linear-gradient(130deg, rgba(9, 44, 69, 0.85), rgba(5, 28, 47, 0.84));
            color: #dff1ff;
        }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(90deg, var(--mm-accent), var(--mm-aqua));
            border: 1px solid var(--mm-accent-dark);
            color: #ffffff;
            box-shadow: 0 8px 18px rgba(9, 84, 120, 0.34);
        }
        div.stButton > button[kind="secondary"] {
            background: linear-gradient(130deg, rgba(9, 44, 69, 0.88), rgba(5, 28, 47, 0.9));
            border: 1px solid rgba(118, 195, 231, 0.30);
            color: #dff1ff;
        }
        div.stButton > button:not([kind="primary"]):hover {
            border-color: rgba(149, 216, 248, 0.44);
            background: linear-gradient(130deg, rgba(13, 57, 87, 0.9), rgba(6, 34, 55, 0.9));
            color: #f0f9ff;
        }
        div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(90deg, var(--mm-accent-dark), var(--mm-accent));
        }
        div.stButton > button:disabled {
            border: 1px solid rgba(118, 195, 231, 0.18) !important;
            background: linear-gradient(130deg, rgba(8, 33, 53, 0.6), rgba(5, 22, 37, 0.6)) !important;
            color: rgba(198, 226, 244, 0.48) !important;
            opacity: 1 !important;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        textarea {
            background: rgba(7, 34, 55, 0.62) !important;
            border-color: rgba(120, 194, 228, 0.34) !important;
            color: #e5f1fb !important;
        }
        div[data-baseweb="select"] [data-baseweb="tag"] {
            background: linear-gradient(110deg, rgba(39, 174, 220, 0.98), rgba(22, 135, 180, 0.95)) !important;
            color: #f4fbff !important;
            border: 1px solid rgba(156, 226, 252, 0.42) !important;
        }
        div[data-baseweb="select"] [data-baseweb="tag"] span,
        div[data-baseweb="select"] [data-baseweb="tag"] svg {
            color: #f4fbff !important;
            fill: #f4fbff !important;
        }
        [data-testid="stRadio"] input[type="radio"] {
            accent-color: #35b8e7 !important;
        }
        [data-testid="stRadio"] svg circle {
            fill: #35b8e7 !important;
            stroke: #35b8e7 !important;
        }
        [data-testid="stRadio"] svg path {
            fill: #eaf9ff !important;
            stroke: #eaf9ff !important;
        }
        div[data-baseweb="radio"] label > div:first-child {
            border-color: rgba(133, 206, 240, 0.72) !important;
            background: rgba(7, 34, 55, 0.7) !important;
        }
        div[data-baseweb="radio"] input:checked + div {
            border-color: #69d2f7 !important;
            background: radial-gradient(circle, #9ae8ff 0%, #35b8e7 55%, #1788b7 100%) !important;
            box-shadow: 0 0 0 3px rgba(74, 187, 231, 0.26);
        }
        [data-testid="stRadio"] label[data-baseweb="radio"][aria-checked="true"] > div:first-child {
            border-color: #69d2f7 !important;
            background: radial-gradient(circle, #9ae8ff 0%, #35b8e7 55%, #1788b7 100%) !important;
            box-shadow: 0 0 0 3px rgba(74, 187, 231, 0.26);
        }
        .stAlert {
            border-radius: 12px;
            border: 1px solid rgba(134, 201, 235, 0.30);
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
        .mm-badge-warning {background: var(--mm-warning-bg); color: #ffe6c4;}
        .mm-badge-error {background: var(--mm-error-bg); color: #ffd4d4;}
        .mm-html-table-wrap {
            width: 100%;
            overflow-x: auto;
            border-radius: 12px;
            border: 1px solid rgba(129, 201, 234, 0.22);
            background: linear-gradient(165deg, rgba(8, 42, 67, 0.52), rgba(5, 27, 44, 0.42));
            backdrop-filter: blur(2px);
        }
        table.mm-html-table {
            width: 100%;
            border-collapse: collapse;
            color: #e9f6ff;
            background: transparent;
            font-size: 0.95rem;
        }
        table.mm-html-table thead th {
            text-align: left;
            font-weight: 600;
            color: #c7e6f9;
            padding: 0.5rem 0.55rem;
            border-bottom: 1px solid rgba(131, 199, 235, 0.18);
            background: rgba(9, 40, 63, 0.28);
            text-transform: lowercase;
        }
        table.mm-html-table tbody td {
            padding: 0.45rem 0.55rem;
            border-bottom: 1px solid rgba(131, 199, 235, 0.10);
            color: #e9f6ff;
            background: rgba(8, 38, 60, 0.16);
        }
        table.mm-html-table tbody tr:nth-child(even) td {
            background: rgba(8, 38, 60, 0.22);
        }
        table.mm-html-table tbody tr:hover td {
            background: rgba(20, 84, 121, 0.25);
        }
        </style>
        """,  # noqa: E501
        unsafe_allow_html=True,
    )
