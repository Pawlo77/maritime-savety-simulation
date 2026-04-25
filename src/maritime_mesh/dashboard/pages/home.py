"""Home page for dashboard overview and methodology."""

from pathlib import Path

import streamlit as st

from maritime_mesh.dashboard.ui import info_panel, page_intro


def render(output_dir: Path) -> None:
    """Render dashboard home page."""
    page_intro(
        "Dashboard Home",
        (
            "Overview of the Maritime Mesh simulation tool, experiment workflow, "
            "and interpretation guidance."
        ),
    )
    info_panel(
        "What This Tool Does",
        (
            "This dashboard helps you configure multi-scenario maritime safety "
            "experiments, compare communication methods, and inspect run-level "
            "behavior through timeline playback."
        ),
    )

    tab_overview, tab_methodology, tab_workflow, tab_outputs = st.tabs(
        ["Overview", "Methodology", "Workflow", "Outputs And KPIs"]
    )

    with tab_overview:
        st.markdown("#### Scope")
        st.markdown(
            "- Simulates vessel movement, weather exposure, communication relay, "
            "and rescue response."
        )
        st.markdown(
            "- Compares baseline and adaptive methods under identical scenario and seed settings."
        )
        st.markdown("- Produces reproducible run logs and aggregate KPI summaries.")
        st.markdown("#### Core Modules")
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

    with tab_workflow:
        st.markdown("#### Recommended Sequence")
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
