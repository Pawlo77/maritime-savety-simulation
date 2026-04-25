"""Home page for dashboard overview and methodology."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from maritime_mesh.dashboard.constants import display_method_name, display_scenario_name
from maritime_mesh.dashboard.data_access import load_summary
from maritime_mesh.dashboard.ui import (
    apply_plotly_theme,
    info_panel,
    page_intro,
    render_dataframe,
    section_intro,
)


def _render_home_kpis(summary: pd.DataFrame) -> None:
    """Render top-level operational KPIs for the command center view."""
    if summary.empty:
        st.info("No `summary.csv` found yet. Run experiments to unlock executive KPIs.")
        return
    scenarios = int(summary["scenario"].nunique())
    methods = int(summary["method"].nunique())
    runs = len(summary)
    mean_survival = (
        float(summary["survival_ratio"].mean()) if "survival_ratio" in summary else float("nan")
    )
    mean_fatal = (
        float(summary["fatal_per_1k_hrs"].mean()) if "fatal_per_1k_hrs" in summary else float("nan")
    )
    best_method = "-"
    if "survival_ratio" in summary:
        by_method = (
            summary.groupby("method", as_index=False)["survival_ratio"]
            .mean()
            .sort_values("survival_ratio", ascending=False)
        )
        if not by_method.empty:
            best_method = display_method_name(str(by_method.iloc[0]["method"]))
    cols = st.columns(5)
    cols[0].metric("Scenarios", scenarios)
    cols[1].metric("Methods", methods)
    cols[2].metric("Runs", runs)
    cols[3].metric("Avg Survival", f"{mean_survival:.3f}" if pd.notna(mean_survival) else "N/A")
    cols[4].metric("Avg Fatal /1k hrs", f"{mean_fatal:.3f}" if pd.notna(mean_fatal) else "N/A")
    st.markdown(
        f"<span class='mm-badge mm-badge-success'>Top method: {best_method}</span>",
        unsafe_allow_html=True,
    )


def _render_home_charts(summary: pd.DataFrame) -> None:
    """Render lightweight charts that orient users before deep-dive pages."""
    if summary.empty:
        return
    left, right = st.columns(2)
    with left:
        if "survival_ratio" in summary:
            method_survival = (
                summary.groupby("method", as_index=False)["survival_ratio"]
                .mean()
                .sort_values("survival_ratio", ascending=False)
            )
            method_survival["method_label"] = method_survival["method"].map(display_method_name)
            fig = px.bar(
                method_survival,
                x="method_label",
                y="survival_ratio",
                text_auto=".3f",
                labels={"method_label": "Method", "survival_ratio": "Mean survival ratio"},
            )
            fig.update_traces(marker_color="#35b8e7")
            fig.update_layout(showlegend=False, title="Method Performance Snapshot")
            apply_plotly_theme(fig, height=360)
            st.plotly_chart(fig, width="stretch")
    with right:
        if "fatal_per_1k_hrs" in summary:
            by_scenario = (
                summary.groupby("scenario", as_index=False)["fatal_per_1k_hrs"]
                .mean()
                .sort_values("fatal_per_1k_hrs", ascending=True)
            )
            by_scenario["scenario_label"] = by_scenario["scenario"].map(display_scenario_name)
            fig = px.bar(
                by_scenario,
                x="scenario_label",
                y="fatal_per_1k_hrs",
                text_auto=".3f",
                labels={
                    "scenario_label": "Scenario",
                    "fatal_per_1k_hrs": "Mean fatalities per 1k hrs",
                },
            )
            fig.update_traces(marker_color="#35b8e7")
            fig.update_layout(showlegend=False, title="Scenario Risk Profile")
            apply_plotly_theme(fig, height=360)
            st.plotly_chart(fig, width="stretch")


def render(output_dir: Path) -> None:
    """Render dashboard home page."""
    summary = load_summary(output_dir=output_dir)
    summary_path = output_dir / "summary.csv"

    page_intro(
        "Fleet Command Center",
        (
            "Mission overview for simulation operations, experiment readiness, "
            "and KPI posture before deeper analysis."
        ),
    )
    info_panel(
        "Mission Brief",
        (
            "Use this command center to check dataset readiness, assess performance at a glance, "
            "then move to Run, Results, Map Playback, and Hypothesis for tactical analysis."
        ),
    )
    section_intro(
        "Operational Snapshot",
        "Live indicators are computed from current output artifacts in the selected directory.",
    )
    _render_home_kpis(summary=summary)
    _render_home_charts(summary=summary)

    tab_overview, tab_methodology, tab_workflow, tab_outputs, tab_readiness = st.tabs(
        ["Executive Overview", "Methodology", "Workflow", "Artifacts and KPIs", "Readiness Checks"]
    )

    with tab_overview:
        st.markdown("#### Program Scope")
        st.markdown(
            "- Simulates vessel movement, weather exposure, communication relay, "
            "and rescue response."
        )
        st.markdown(
            "- Compares baseline and adaptive methods under identical scenario and seed settings."
        )
        st.markdown("- Produces reproducible run logs and aggregate KPI summaries.")
        st.markdown("#### Command Modules")
        st.markdown(
            "- **Run**: configure scenarios, lanes, shore stations, and communication settings."
        )
        st.markdown(
            "- **Results**: compare KPI distributions, ranking, and statistical significance."
        )
        st.markdown(
            "- **Map Playback**: replay one run over time with event and communication overlays."
        )
        st.markdown("- **Hypothesis**: run pairwise non-parametric tests with corrected p-values.")
        if not summary.empty:
            scenario_mix = (
                summary.groupby("scenario", as_index=False)
                .size()
                .rename(columns={"size": "runs"})
                .sort_values("runs", ascending=False)
            )
            scenario_mix["scenario"] = scenario_mix["scenario"].map(display_scenario_name)
            render_dataframe(scenario_mix, width="stretch", hide_index=True)

    with tab_methodology:
        st.markdown("#### Simulation Approach")
        st.markdown("- Agent-based model with vessels, coastal stations, and rescue assets.")
        st.markdown(
            "- Weather field evolves over ticks and influences risk and communication reliability."
        )
        st.markdown("- Mesh relay and shore radio channels are used to share hazard information.")
        st.markdown("#### Decision Logic")
        st.markdown("- Vessel preparedness and forecast error affect evacuation decisions.")
        st.markdown("- Rescue dispatch responds to SOS flow and local environmental constraints.")
        st.markdown("- Collision and grounding events are tracked for safety KPI computation.")
        st.markdown("#### Modeling Principles")
        st.markdown(
            "- Shared seeds ensure fair method comparisons under equal stochastic conditions."
        )
        st.markdown(
            "- Per-run manifests preserve exact parameterization for reproducibility audits."
        )
        st.markdown(
            "- KPI families combine safety outcomes, responsiveness, and behavioral adaptation."
        )

    with tab_workflow:
        st.markdown("#### Recommended Operations Sequence")
        st.markdown("1. Choose output directory and verify artifact status.")
        st.markdown("2. Configure run parameters in **Run** and launch experiment matrix.")
        st.markdown("3. Analyze aggregate outcomes in **Results**.")
        st.markdown("4. Inspect single-run behavior in **Map Playback**.")
        st.markdown("5. Validate claims in **Hypothesis**.")
        st.markdown("#### Practical Tips")
        st.markdown(
            "- Increase seed count when comparing methods for stronger statistical stability."
        )
        st.markdown(
            "- Use scenario filters first, then focus on one KPI for ranking "
            "and outlier drill-down."
        )
        st.markdown("- Keep lane geometry and shore placement realistic to avoid invalid runs.")
        st.markdown("#### Suggested Cadence")
        st.markdown("- Run small pilot matrices first, then scale seeds once design is stable.")
        st.markdown("- Revisit map playback for every surprising KPI outlier before reporting.")

    with tab_outputs:
        st.markdown("#### Generated Artifacts")
        st.markdown("- `summary.csv`: method/scenario/seed aggregate KPI table.")
        st.markdown("- per-run parquet logs: tick-level entities and events used for playback.")
        st.markdown("- per-run manifest JSON: effective configuration for reproducibility.")
        st.markdown("#### KPI Reading Guide")
        st.markdown("- `fatal_per_1k_hrs` and `collision_per_1k_hrs`: lower is better.")
        st.markdown("- `survival_ratio`: higher is better.")
        st.markdown("- `avg_tta_hours`: lower rescue time-to-arrival is better.")
        st.markdown("- `mean_p_prep` and `evac_activation_rate`: context-dependent indicators.")
        st.info(f"Current output directory: `{output_dir}`")

    with tab_readiness:
        st.markdown("#### Data Readiness")
        summary_exists = summary_path.exists()
        run_files = sorted(output_dir.glob("*.parquet"))
        manifests = sorted(output_dir.glob("*.manifest.json"))
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("summary.csv", "Available" if summary_exists else "Missing")
        col_b.metric("Run logs (.parquet)", len(run_files))
        col_c.metric("Manifests (.json)", len(manifests))
        if summary_exists:
            modified_at = summary_path.stat().st_mtime
            st.caption(f"summary.csv last modified (epoch): `{modified_at:.0f}`")
            st.markdown(
                "<span class='mm-badge mm-badge-success'>Ready for analysis</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                (
                    "<span class='mm-badge mm-badge-warning'>"
                    "Run experiments to populate outputs</span>"
                ),
                unsafe_allow_html=True,
            )
        st.markdown("#### Next Recommended Action")
        if not summary_exists:
            st.write("Go to **Run** page and launch at least one scenario-method-seed matrix.")
        elif len(run_files) == 0:
            st.write("Summary exists but no run logs found. Re-run with log persistence enabled.")
        else:
            st.write("Proceed to **Results** for KPI ranking and significance checks.")
