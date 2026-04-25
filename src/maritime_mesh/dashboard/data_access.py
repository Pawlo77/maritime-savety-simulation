"""Data-loading helpers for dashboard pages."""

import json
from pathlib import Path

import pandas as pd

from maritime_mesh.constants import WORLD_SIZE_NM
from maritime_mesh.dashboard.constants import KPI_COLUMNS

try:
    import streamlit as st
except Exception:  # pragma: no cover - streamlit unavailable in some test contexts.
    st = None

SUMMARY_REQUIRED_COLUMNS = {"scenario", "method", "seed", *KPI_COLUMNS}
RUN_REQUIRED_COLUMNS = {"tick", "entity_type", "world_size_nm", "simulation_seed"}
MANIFEST_SCHEMA_VERSION = 1


def _read_manifest(manifest_path: Path) -> dict:
    """Load and validate run manifest when present."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = int(payload.get("schema_version", -1))
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Result manifest schema mismatch. "
            f"Expected {MANIFEST_SCHEMA_VERSION}, found {schema_version}. "
            "Re-run experiments with the current version, or select/clean a fresh output directory."
        )
    return payload


def _validate_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure summary schema is compatible with dashboard pages."""
    missing = sorted(SUMMARY_REQUIRED_COLUMNS - set(summary_df.columns))
    if missing:
        raise ValueError(
            "summary.csv is missing required columns: "
            f"{', '.join(missing)}. Re-run experiments to regenerate outputs "
            "or switch to a clean output directory."
        )
    return summary_df


def _validate_run_log(run_df: pd.DataFrame, run_name: str) -> pd.DataFrame:
    """Ensure run parquet schema is compatible with playback page."""
    missing = sorted(RUN_REQUIRED_COLUMNS - set(run_df.columns))
    if missing:
        raise ValueError(
            f"{run_name}.parquet is missing required columns: {', '.join(missing)}. "
            "Re-run experiments with the latest logger or remove stale parquet files "
            "from this output directory."
        )
    return run_df


if st is not None:
    _cached_read_csv = st.cache_data(show_spinner=False)(pd.read_csv)
    _cached_read_parquet = st.cache_data(show_spinner=False)(pd.read_parquet)
else:
    _cached_read_csv = pd.read_csv
    _cached_read_parquet = pd.read_parquet


def load_summary(output_dir: Path) -> pd.DataFrame:
    """Load summary CSV if available."""
    summary_path = output_dir / "summary.csv"
    if summary_path.exists():
        return _validate_summary(_cached_read_csv(summary_path))
    return pd.DataFrame()


def load_run_log(output_dir: Path, scenario: str, method: str, seed: int) -> pd.DataFrame:
    """Load one per-run parquet tick log."""
    run_name = f"{scenario}_{method}_{seed}"
    run_path = output_dir / f"{run_name}.parquet"
    manifest_path = output_dir / f"{run_name}.manifest.json"
    if manifest_path.exists():
        _read_manifest(manifest_path)
    if not run_path.exists():
        return pd.DataFrame()
    return _validate_run_log(_cached_read_parquet(run_path), run_name=run_name)


def map_world_size(run_df: pd.DataFrame) -> float:
    """Infer world size from run logs."""
    if "world_size_nm" in run_df.columns:
        series = run_df["world_size_nm"].dropna()
        if not series.empty:
            return float(series.max())
    if "x_nm" not in run_df.columns or "y_nm" not in run_df.columns:
        return WORLD_SIZE_NM
    xy_max = max(float(run_df["x_nm"].max()), float(run_df["y_nm"].max()))
    return max(WORLD_SIZE_NM, xy_max)


def can_render_map(run_df: pd.DataFrame) -> bool:
    """Return whether run log contains required columns for map rendering."""
    required = {
        "tick",
        "entity_type",
        "x_nm",
        "y_nm",
    }
    return required.issubset(set(run_df.columns))
