"""Per-tick logging and KPI aggregation."""

from pathlib import Path

import pandas as pd

from maritime_mesh.constants import MACRO_TICK_HOURS


class KpiLogger:
    """Collect tick records and compute run-level KPIs."""

    def __init__(
        self,
        *,
        output_path: str | Path | None = None,
        flush_every_ticks: int | None = None,
    ) -> None:
        """Initialize record store."""
        self.records: list[dict] = []
        self.event_records: list[dict] = []
        self._collisions = 0
        self._weather_probe_steps = 12
        self._rescue_tta_hours: list[float] = []
        self._output_path = Path(output_path) if output_path is not None else None
        default_flush_ticks = max(1, round(24.0 / max(1e-9, MACRO_TICK_HOURS)))
        self._flush_every_ticks = (
            int(flush_every_ticks) if flush_every_ticks is not None else default_flush_ticks
        )
        self._chunk_index = 0
        self._first_survivors_by_vessel: dict[int, float] = {}
        self._last_survivors_by_vessel: dict[int, float] = {}
        self._evac_by_vessel: dict[int, bool] = {}
        self._p_prep_sum = 0.0
        self._p_prep_count = 0
        self._last_tick_seen = -1

    def _run_chunk_path(self, chunk_index: int) -> Path:
        """Return deterministic file path for one flushed chunk."""
        if self._output_path is None:
            raise ValueError("output_path is not configured.")
        return self._output_path.with_name(
            f"{self._output_path.name}.part{chunk_index:05d}.parquet"
        )

    def configure_output(
        self, output_path: str | Path, *, clear_existing_chunks: bool = True
    ) -> None:
        """Attach logger to output location and optionally reset prior chunk files."""
        new_output_path = Path(output_path)
        output_changed = self._output_path != new_output_path
        self._output_path = new_output_path
        if output_changed:
            self._chunk_index = 0
        if clear_existing_chunks:
            self._chunk_index = 0
            for old_chunk in self._output_path.parent.glob(
                f"{self._output_path.name}.part*.parquet"
            ):
                old_chunk.unlink(missing_ok=True)

    def _update_kpi_accumulators(self, vessel) -> None:
        """Track KPI aggregates incrementally so full-record retention is unnecessary."""
        vessel_id = int(vessel.unique_id)
        survivors = float(vessel.n_survivors)
        self._last_survivors_by_vessel[vessel_id] = survivors
        if vessel_id not in self._first_survivors_by_vessel:
            self._first_survivors_by_vessel[vessel_id] = survivors
        self._evac_by_vessel[vessel_id] = self._evac_by_vessel.get(vessel_id, False) or bool(
            vessel.has_evacuated
        )
        self._p_prep_sum += float(vessel.p_prep)
        self._p_prep_count += 1

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
                sea_state, visibility, wind_norm = weather_field.channels_at(x_nm, y_nm)
                self.records.append(
                    {
                        "tick": tick,
                        "entity_type": "weather_probe",
                        "entity_id": f"wx_{row}_{col}",
                        "x_nm": x_nm,
                        "y_nm": y_nm,
                        "hazard": weather_field.hazard_at(x_nm, y_nm),
                        "sea_state": sea_state,
                        "visibility": visibility,
                        "wind_norm": wind_norm,
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
        despawned_arrivals: list[tuple[int, tuple[float, float]]],
        rescue_dispatches: list[tuple[int, int, tuple[float, float]]],
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
            self._update_kpi_accumulators(vessel)
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
                    "sos_sent": vessel.sos_sent,
                    "sos_reason": vessel.sos_reason,
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
                    "is_idle": bool(rescue.is_idle),
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
        for vessel_id, position in despawned_arrivals:
            self.event_records.append(
                {
                    "tick": tick,
                    "entity_type": "intervention_event",
                    "entity_id": f"despawned_{vessel_id}_{tick}",
                    "event_kind": "despawned",
                    "source_id": vessel_id,
                    "x_nm": position[0],
                    "y_nm": position[1],
                    "world_size_nm": world_size_nm,
                    "simulation_seed": simulation_seed,
                }
            )
        for rescue_id, vessel_id, dispatch_position in rescue_dispatches:
            self.event_records.append(
                {
                    "tick": tick,
                    "entity_type": "intervention_event",
                    "entity_id": f"rescue_dispatch_{rescue_id}_{vessel_id}_{tick}",
                    "event_kind": "rescue_dispatch",
                    "source_id": rescue_id,
                    "target_id": vessel_id,
                    "x_nm": dispatch_position[0],
                    "y_nm": dispatch_position[1],
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
            if tick == 0:
                self.records.append(record)
        self._last_tick_seen = max(self._last_tick_seen, int(tick))
        if self._output_path is not None and (tick + 1) % self._flush_every_ticks == 0:
            self._flush_chunk()

    def _flush_chunk(self) -> None:
        """Persist currently buffered records as one parquet chunk."""
        if self._output_path is None:
            return
        if not self.records and not self.event_records:
            return
        output_path = self._run_chunk_path(self._chunk_index)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame([*self.records, *self.event_records])
        for column in ("entity_id", "source_id", "target_id"):
            if column in frame.columns:
                frame[column] = frame[column].astype("string")
        frame.to_parquet(output_path, index=False)
        self.records.clear()
        self.event_records.clear()
        self._chunk_index += 1

    def compute_kpis(self) -> dict[str, float]:
        """Aggregate records into experiment KPI set."""
        if self.records and not self._last_survivors_by_vessel:
            vessel_rows = [row for row in self.records if row.get("entity_type") == "vessel"]
            vessel_rows.sort(
                key=lambda row: (int(row.get("vessel_id", -1)), int(row.get("tick", -1)))
            )
            self._last_tick_seen = max(
                self._last_tick_seen,
                max(
                    (int(row.get("tick", -1)) for row in vessel_rows), default=self._last_tick_seen
                ),
            )
            for row in vessel_rows:
                vessel_id = int(row["vessel_id"])
                survivors = float(row.get("n_survivors", 0.0))
                self._last_survivors_by_vessel[vessel_id] = survivors
                if vessel_id not in self._first_survivors_by_vessel:
                    self._first_survivors_by_vessel[vessel_id] = survivors
                self._evac_by_vessel[vessel_id] = self._evac_by_vessel.get(
                    vessel_id, False
                ) or bool(row.get("has_evacuated", False))
                self._p_prep_sum += float(row.get("p_prep", 0.0))
                self._p_prep_count += 1
        if not self._last_survivors_by_vessel:
            return {
                "fatal_per_1k_hrs": 0.0,
                "collision_per_1k_hrs": 0.0,
                "survival_ratio": 1.0,
                "avg_tta_hours": 0.0,
                "evac_activation_rate": 0.0,
                "mean_p_prep": 0.0,
            }
        unique_vessels = len(self._last_survivors_by_vessel)
        total_hours = max(1e-9, unique_vessels * (self._last_tick_seen + 1) * MACRO_TICK_HOURS)
        total_exposed_crew = float(sum(self._first_survivors_by_vessel.values()))
        survivors = float(sum(self._last_survivors_by_vessel.values()))
        fatalities = max(0.0, total_exposed_crew - survivors)
        return {
            "fatal_per_1k_hrs": (fatalities / total_hours) * 1000.0,
            "collision_per_1k_hrs": (self._collisions / total_hours) * 1000.0,
            "survival_ratio": survivors / max(1.0, total_exposed_crew),
            "avg_tta_hours": float(pd.Series(self._rescue_tta_hours).mean())
            if self._rescue_tta_hours
            else 0.0,
            "evac_activation_rate": float(pd.Series(list(self._evac_by_vessel.values())).mean())
            if self._evac_by_vessel
            else 0.0,
            "mean_p_prep": self._p_prep_sum / max(1, self._p_prep_count),
        }

    def flush_to_parquet(self, path: str) -> None:
        """Write tick-level records to parquet path."""
        if self._output_path is None or self._output_path != Path(path):
            self.configure_output(path, clear_existing_chunks=False)
        self._flush_chunk()
