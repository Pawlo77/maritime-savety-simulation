"""Per-tick logging and KPI aggregation."""

from pathlib import Path

import pandas as pd

from maritime_mesh.constants import MACRO_TICK_HOURS


class KpiLogger:
    """Collect tick records and compute run-level KPIs."""

    def __init__(self) -> None:
        """Initialize record store."""
        self.records: list[dict] = []
        self.event_records: list[dict] = []
        self._collisions = 0
        self._weather_probe_steps = 12
        self._rescue_tta_hours: list[float] = []

    def add_collisions(self, count: int) -> None:
        """Track collisions detected in current tick."""
        self._collisions += count

    def add_rescue_tta(self, tta_hours: float) -> None:
        """Track rescue time-to-arrival for KPI aggregation."""
        self._rescue_tta_hours.append(float(max(0.0, tta_hours)))

    def _append_weather_probes(self, tick: int, weather_field, simulation_seed: int) -> None:
        """Append coarse grid weather probes for map heat overlay."""
        for row in range(self._weather_probe_steps):
            for col in range(self._weather_probe_steps):
                x_nm = weather_field.world_size_nm * (col / max(1, self._weather_probe_steps - 1))
                y_nm = weather_field.world_size_nm * (row / max(1, self._weather_probe_steps - 1))
                self.records.append(
                    {
                        "tick": tick,
                        "entity_type": "weather_probe",
                        "entity_id": f"wx_{row}_{col}",
                        "x_nm": x_nm,
                        "y_nm": y_nm,
                        "hazard": weather_field.hazard_at(x_nm, y_nm),
                        "world_size_nm": weather_field.world_size_nm,
                        "simulation_seed": simulation_seed,
                    }
                )

    def log_tick(
        self,
        tick: int,
        vessels: list,
        coastal_stations: list,
        rescue_agents: list,
        relay_links: list[tuple[int, int]],
        collisions: list[tuple[int, int]],
        land_collisions: list[tuple[int, tuple[float, float]]],
        land_shapes: tuple,
        weather_field,
        world_size_nm: float,
        simulation_seed: int,
    ) -> None:
        """Append entities and events for one tick."""
        self._append_weather_probes(
            tick=tick, weather_field=weather_field, simulation_seed=simulation_seed
        )
        for vessel in vessels:
            self.records.append(
                {
                    "tick": tick,
                    "entity_type": "vessel",
                    "entity_id": vessel.unique_id,
                    "vessel_id": vessel.unique_id,
                    "state": vessel.state.value,
                    "n_survivors": vessel.n_survivors,
                    "p_prep": vessel.p_prep,
                    "has_evacuated": vessel.has_evacuated,
                    "x_nm": vessel.position[0],
                    "y_nm": vessel.position[1],
                    "heading_deg": vessel.heading_deg,
                    "speed_kn": vessel.speed_kn,
                    "hazard": vessel.last_true_hazard,
                    "forecast_error": vessel.forecast_error,
                    "w_hat_blend": vessel.w_hat_blend,
                    "shore_age_ticks": vessel.shore_age_ticks,
                    "shore_rx": vessel.last_shore_received,
                    "distance_to_shore_nm": vessel.last_distance_to_shore,
                    "mesh_observations": vessel.last_mesh_observation_count,
                    "error_probability": vessel.last_error_probability,
                    "world_size_nm": world_size_nm,
                    "simulation_seed": simulation_seed,
                }
            )
        for coastal_station in coastal_stations:
            self.records.append(
                {
                    "tick": tick,
                    "entity_type": "coastal_station",
                    "entity_id": coastal_station.unique_id,
                    "x_nm": coastal_station.position[0],
                    "y_nm": coastal_station.position[1],
                    "hazard": weather_field.hazard_at(*coastal_station.position),
                    "shore_broadcast": coastal_station.last_broadcast,
                    "queued_sos": len(coastal_station.sos_queue),
                    "world_size_nm": world_size_nm,
                    "simulation_seed": simulation_seed,
                }
            )
        for rescue in rescue_agents:
            self.records.append(
                {
                    "tick": tick,
                    "entity_type": "rescue_asset",
                    "entity_id": rescue.unique_id,
                    "x_nm": rescue.position[0],
                    "y_nm": rescue.position[1],
                    "asset_type": rescue.asset_type.value,
                    "mobilisation_ticks_remaining": rescue.mobilisation_ticks_remaining,
                    "target_x_nm": rescue.target_position[0],
                    "target_y_nm": rescue.target_position[1],
                    "world_size_nm": world_size_nm,
                    "simulation_seed": simulation_seed,
                }
            )
        for source_id, target_id in relay_links:
            self.event_records.append(
                {
                    "tick": tick,
                    "entity_type": "communication_link",
                    "entity_id": f"{source_id}_{target_id}",
                    "source_id": source_id,
                    "target_id": target_id,
                    "world_size_nm": world_size_nm,
                    "simulation_seed": simulation_seed,
                }
            )
        for vessel_a, vessel_b in collisions:
            self.event_records.append(
                {
                    "tick": tick,
                    "entity_type": "intervention_event",
                    "entity_id": f"collision_{vessel_a}_{vessel_b}",
                    "event_kind": "collision",
                    "source_id": vessel_a,
                    "target_id": vessel_b,
                    "world_size_nm": world_size_nm,
                    "simulation_seed": simulation_seed,
                }
            )
        for vessel_id, position in land_collisions:
            self.event_records.append(
                {
                    "tick": tick,
                    "entity_type": "intervention_event",
                    "entity_id": f"land_collision_{vessel_id}_{tick}",
                    "event_kind": "land_collision",
                    "source_id": vessel_id,
                    "x_nm": position[0],
                    "y_nm": position[1],
                    "world_size_nm": world_size_nm,
                    "simulation_seed": simulation_seed,
                }
            )
        for idx, land in enumerate(land_shapes):
            record = {
                "tick": tick,
                "entity_type": "landmass",
                "entity_id": f"land_{idx}",
                "world_size_nm": world_size_nm,
                "simulation_seed": simulation_seed,
            }
            if hasattr(land, "x0"):
                record.update(
                    {
                        "geometry_type": "rectangle",
                        "x0_nm": land.x0,
                        "y0_nm": land.y0,
                        "x1_nm": land.x1,
                        "y1_nm": land.y1,
                    }
                )
            elif hasattr(land, "points"):
                serialized = ";".join(f"{point[0]},{point[1]}" for point in land.points)
                record.update({"geometry_type": "polygon", "points_nm": serialized})
            self.records.append(record)

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
        vessel_df = df[df["entity_type"] == "vessel"].copy()
        unique_vessels = vessel_df["vessel_id"].nunique()
        total_hours = max(1e-9, unique_vessels * (df["tick"].max() + 1) * MACRO_TICK_HOURS)
        final_states = (
            vessel_df.sort_values("tick")
            .groupby("vessel_id", as_index=False)
            .tail(1)
            .set_index("vessel_id")
        )
        initial_survivors = (
            vessel_df.sort_values("tick")
            .groupby("vessel_id", as_index=False)
            .head(1)
            .set_index("vessel_id")["n_survivors"]
        )
        final_survivors = final_states["n_survivors"].astype(float)
        total_exposed_crew = float(initial_survivors.sum())
        survivors = float(final_survivors.sum())
        fatalities = max(0.0, total_exposed_crew - survivors)
        return {
            "fatal_per_1k_hrs": (fatalities / total_hours) * 1000.0,
            "collision_per_1k_hrs": (self._collisions / total_hours) * 1000.0,
            "survival_ratio": survivors / max(1.0, total_exposed_crew),
            "avg_tta_hours": float(pd.Series(self._rescue_tta_hours).mean())
            if self._rescue_tta_hours
            else 0.0,
            "evac_activation_rate": float(
                vessel_df.groupby("vessel_id")["has_evacuated"].max().mean()
            ),
            "mean_p_prep": float(vessel_df["p_prep"].mean()),
        }

    def flush_to_parquet(self, path: str) -> None:
        """Write tick-level records to parquet path."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame([*self.records, *self.event_records])
        for column in ("entity_id", "source_id", "target_id"):
            if column in frame.columns:
                frame[column] = frame[column].astype("string")
        frame.to_parquet(output_path, index=False)
