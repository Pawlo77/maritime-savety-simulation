"""Per-tick logging and KPI aggregation."""

from pathlib import Path

import pandas as pd

from maritime_mesh.constants import MACRO_TICK_HOURS
from maritime_mesh.enums import VesselState


class KpiLogger:
    """Collect tick records and compute run-level KPIs."""

    def __init__(self) -> None:
        """Initialize record store."""
        self.records: list[dict] = []
        self._collisions = 0

    def add_collisions(self, count: int) -> None:
        """Track collisions detected in current tick."""
        self._collisions += count

    def log_tick(self, tick: int, vessels: list) -> None:
        """Append vessel state records for one tick."""
        for vessel in vessels:
            self.records.append(
                {
                    "tick": tick,
                    "vessel_id": vessel.unique_id,
                    "state": vessel.state.value,
                    "n_survivors": vessel.n_survivors,
                    "p_prep": vessel.p_prep,
                    "has_evacuated": vessel.has_evacuated,
                }
            )

    def compute_kpis(self) -> dict[str, float]:
        """Aggregate records into experiment KPI set."""
        if not self.records:
            return {
                "fatal_per_1k_hrs": 0.0,
                "collision_per_1k_hrs": 0.0,
                "survival_ratio": 1.0,
                "avg_tta_hours": 0.0,
                "evac_activation_rate": 0.0,
                "mean_p_prep": 0.0,
            }
        df = pd.DataFrame(self.records)
        unique_vessels = df["vessel_id"].nunique()
        total_hours = max(1e-9, unique_vessels * (df["tick"].max() + 1) * MACRO_TICK_HOURS)
        final_states = (
            df.sort_values("tick")
            .groupby("vessel_id", as_index=False)
            .tail(1)
            .set_index("vessel_id")
        )
        fatalities = float((final_states["state"] == VesselState.SUNK.value).sum())
        survivors = float(final_states["n_survivors"].sum())
        return {
            "fatal_per_1k_hrs": (fatalities / total_hours) * 1000.0,
            "collision_per_1k_hrs": (self._collisions / total_hours) * 1000.0,
            "survival_ratio": survivors / max(1.0, survivors + fatalities),
            "avg_tta_hours": 0.0,
            "evac_activation_rate": float(df.groupby("vessel_id")["has_evacuated"].max().mean()),
            "mean_p_prep": float(df["p_prep"].mean()),
        }

    def flush_to_parquet(self, path: str) -> None:
        """Write tick-level records to parquet path."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.records).to_parquet(output_path, index=False)
