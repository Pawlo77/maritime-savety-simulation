"""Streamlit dashboard for maritime mesh experiment outputs."""

from pathlib import Path

import pandas as pd
import streamlit as st

from maritime_mesh.experiment.analysis import StatisticalAnalyser


def _load_summary(output_dir: Path) -> pd.DataFrame:
    """Load summary CSV if available."""
    summary_path = output_dir / "summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return pd.DataFrame()


def main() -> None:
    """Render dashboard views for precomputed experiment data."""
    st.set_page_config(page_title="Maritime Mesh Dashboard", layout="wide")
    st.title("Maritime Weather Mesh Simulation")
    output_dir = Path(st.sidebar.text_input("Output directory", "outputs/maritime_mesh"))
    results = _load_summary(output_dir)
    if results.empty:
        st.warning("No summary.csv found. Run experiment first.")
        return

    scenario = st.sidebar.selectbox("Scenario", sorted(results["scenario"].unique()))
    method_filter = st.sidebar.multiselect(
        "Methods",
        sorted(results["method"].unique()),
        default=sorted(results["method"].unique()),
    )
    filtered = results[(results["scenario"] == scenario) & (results["method"].isin(method_filter))]
    st.subheader("KPI Distribution")
    st.dataframe(filtered, use_container_width=True)

    st.subheader("Method Means")
    means = filtered.groupby("method").mean(numeric_only=True).reset_index()
    st.bar_chart(means.set_index("method")[["fatal_per_1k_hrs", "collision_per_1k_hrs", "survival_ratio"]])

    st.subheader("Hypothesis Table")
    analyser = StatisticalAnalyser(results_df=results)
    st.dataframe(analyser.full_report(), use_container_width=True)


if __name__ == "__main__":
    main()
