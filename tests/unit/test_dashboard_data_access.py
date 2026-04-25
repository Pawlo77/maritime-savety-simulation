"""Unit tests for dashboard data loading validation."""

import json
from pathlib import Path

import pandas as pd
import pytest

from maritime_mesh.dashboard.data_access import load_run_log, load_summary


def test_load_summary_raises_on_missing_required_columns(tmp_path: Path) -> None:
    """Summary loader should reject schema-incompatible CSV files."""
    summary = pd.DataFrame({"scenario": ["s1"], "method": ["proposed"], "seed": [0]})
    summary.to_csv(tmp_path / "summary.csv", index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_summary(tmp_path)


def test_load_run_log_validates_manifest_schema_version(tmp_path: Path) -> None:
    """Run-log loader should fail on incompatible manifest schema versions."""
    run_name = "scenario_1_proposed_0"
    manifest = {
        "engine": "maritime_mesh",
        "schema_version": 999,
        "scenario": "scenario_1",
        "method": "proposed",
        "seed": 0,
    }
    (tmp_path / f"{run_name}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="schema mismatch"):
        load_run_log(tmp_path, scenario="scenario_1", method="proposed", seed=0)


def test_load_run_log_raises_on_missing_required_columns(tmp_path: Path) -> None:
    """Run-log loader should reject parquet files without playback schema."""
    run_name = "scenario_1_proposed_0"
    manifest = {
        "engine": "maritime_mesh",
        "schema_version": 1,
        "scenario": "scenario_1",
        "method": "proposed",
        "seed": 0,
    }
    (tmp_path / f"{run_name}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame({"tick": [0], "entity_type": ["vessel"]}).to_parquet(
        tmp_path / f"{run_name}.parquet",
        index=False,
    )
    with pytest.raises(ValueError, match="missing required columns"):
        load_run_log(tmp_path, scenario="scenario_1", method="proposed", seed=0)
