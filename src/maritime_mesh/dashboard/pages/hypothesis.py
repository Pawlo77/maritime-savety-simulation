"""Hypothesis testing page."""

from pathlib import Path

import streamlit as st

from maritime_mesh.dashboard.data_access import load_summary
from maritime_mesh.experiment.analysis import StatisticalAnalyser


def render(output_dir: Path) -> None:
    """Render hypothesis-testing table."""
    results = load_summary(output_dir)
    if results.empty:
        st.warning("No summary.csv found. Run experiments first.")
        return
    st.markdown("### Hypothesis Tests")
    st.markdown(
        "Non-parametric test outcomes (p-value, effect size, confidence intervals) "
        "for configured hypothesis comparisons."
    )
    analyser = StatisticalAnalyser(results_df=results)
    st.dataframe(analyser.full_report(), use_container_width=True)
