"""Data-loading helpers for dashboard pages."""

from pathlib import Path

import pandas as pd

from maritime_mesh.constants import WORLD_SIZE_NM


def load_summary(output_dir: Path) -> pd.DataFrame:
    """Load summary CSV if available."""
    summary_path = output_dir / "summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return pd.DataFrame()


def load_run_log(output_dir: Path, scenario: str, method: str, seed: int) -> pd.DataFrame:
    """Load one per-run parquet tick log."""
    run_path = output_dir / f"{scenario}_{method}_{seed}.parquet"
    if not run_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(run_path)


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
