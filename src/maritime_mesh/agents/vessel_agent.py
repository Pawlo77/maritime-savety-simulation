"""Vessel agent implementation."""

from math import atan2, cos, dist, radians, sin

import numpy as np

from maritime_mesh.agents.base_agent import AbstractMesaAgent
from maritime_mesh.behaviour.evacuation import EvacuationPolicy
from maritime_mesh.behaviour.human_factors import ErrorProbabilityModel
from maritime_mesh.behaviour.preparedness import PreparednessScorer
from maritime_mesh.behaviour.raft import RaftDeploymentModel, SurvivalModel
from maritime_mesh.communication.mesh_relay import MeshRelayProtocol
from maritime_mesh.communication.packet import MeshPacket
from maritime_mesh.communication.shore_radio import ShoreRadioModel
from maritime_mesh.constants import (
    ARCHETYPE_MOD_GREEN,
    ARCHETYPE_MOD_STANDARD,
    ARCHETYPE_MOD_VETERAN,
    MACRO_TICK_HOURS,
)
from maritime_mesh.enums import CrewArchetype, VesselState
from maritime_mesh.fusion.confidence import ConfidenceWeighter
from maritime_mesh.fusion.shore_trust import ForecastFuser
from maritime_mesh.mesa_compat import Model
from maritime_mesh.weather.weather_field import WeatherField
from maritime_mesh.world.lane import ShippingLane


class VesselAgent(AbstractMesaAgent):
    """Vessel navigating lane while processing weather intelligence."""

    def __init__(
        self,
        model: Model,
        unique_id: int,
        rng: np.random.Generator,
        weather_field: WeatherField,
        lane: ShippingLane,
        shore_radio: ShoreRadioModel,
        mesh_relay: MeshRelayProtocol,
        confidence_weighter: ConfidenceWeighter,
        forecast_fuser: ForecastFuser,
        error_prob_model: ErrorProbabilityModel,
        preparedness_scorer: PreparednessScorer,
        evacuation_policy: EvacuationPolicy,
        raft_model: RaftDeploymentModel,
        survival_model: SurvivalModel,
        position: tuple[float, float],
        speed_kn: float,
        archetype: CrewArchetype,
        shore_station_positions: tuple[tuple[float, float], ...],
        initial_target_waypoint_index: int | None = None,
        destination_waypoint_index: int | None = None,
    ) -> None:
        """Initialize vessel with composed behaviour components."""
        super().__init__(model=model, rng=rng)
        self.unique_id = unique_id
        self.weather_field = weather_field
        self.lane = lane
        self.shore_radio = shore_radio
        self.mesh_relay = mesh_relay
        self.confidence_weighter = confidence_weighter
        self.forecast_fuser = forecast_fuser
        self.error_prob_model = error_prob_model
        self.preparedness_scorer = preparedness_scorer
        self.evacuation_policy = evacuation_policy
        self.raft_model = raft_model
        self.survival_model = survival_model
        self.position = position
        self.heading_deg = 0.0
        self.speed_kn = speed_kn
        self.archetype = archetype
        self.archetype_modifier = {
            CrewArchetype.VETERAN: ARCHETYPE_MOD_VETERAN,
            CrewArchetype.STANDARD: ARCHETYPE_MOD_STANDARD,
            CrewArchetype.GREEN: ARCHETYPE_MOD_GREEN,
        }[archetype]
        self.hours_awake = 0.0
        self.state = VesselState.ACTIVE
        self.inbox: list[MeshPacket] = []
        self.shore_age_ticks = 1
        self.w_hat_shore: float | None = None
        self.p_prep = 0.0
        self.w_hat_blend = 0.0
        self.forecast_error = 0.0
        self.damage = 0
        self.stability_threshold = 3
        self.n_survivors = 10
        self.has_evacuated = False
        self.sos_sent = False
        self.sos_reason = ""
        self.last_true_hazard = 0.0
        self.last_mesh_observation_count = 0
        self.last_shore_received = False
        self.last_distance_to_shore = 0.0
        self.last_error_probability = 0.0
        self.reached_destination_this_tick = False
        self.shore_station_positions = shore_station_positions
        if initial_target_waypoint_index is not None:
            self._target_waypoint_index = initial_target_waypoint_index % len(self.lane.waypoints)
        else:
            closest_index = self.lane.closest_waypoint_index(self.position)
            self._target_waypoint_index = self.lane.next_index(closest_index)
        self.destination_waypoint_index = (
            destination_waypoint_index % len(self.lane.waypoints)
            if destination_waypoint_index is not None
            else None
        )

    def _move_is_safe(self, start: tuple[float, float], end: tuple[float, float]) -> bool:
        """Return whether movement segment avoids buffered shoreline."""
        if (
            not hasattr(self, "model")
            or not hasattr(self.model, "land")
            or not hasattr(self.model, "config")
        ):
            return True
        return not self.model.land.segment_intersects_land(
            start,
            end,
            clearance_nm=self.model.config.land_clearance_nm,
        )

    def _navigate_lane(self) -> None:
        """Move vessel along ordered lane waypoints by one macro-tick step."""
        hazard_slowdown = max(0.3, 1.0 - (0.45 * getattr(self, "last_true_hazard", 0.0)))
        destination_waypoint_index = getattr(self, "destination_waypoint_index", None)
        remaining_nm = self.speed_kn * hazard_slowdown * MACRO_TICK_HOURS
        while remaining_nm > 0.0:
            current_waypoint_index = self._target_waypoint_index
            waypoint = self.lane.waypoint_at(self._target_waypoint_index)
            dx = waypoint.x_nm - self.position[0]
            dy = waypoint.y_nm - self.position[1]
            distance = dist(self.position, (waypoint.x_nm, waypoint.y_nm))
            if distance == 0.0:
                if (
                    destination_waypoint_index is not None
                    and current_waypoint_index == destination_waypoint_index
                ):
                    self.reached_destination_this_tick = True
                    break
                self._target_waypoint_index = self.lane.next_index(self._target_waypoint_index)
                continue
            move_nm = min(distance, remaining_nm)
            angle = atan2(dy, dx)
            self.heading_deg = (90.0 - np.degrees(angle)) % 360.0
            proposed_position = (
                self.position[0] + (move_nm * cos(radians(90.0 - self.heading_deg))),
                self.position[1] + (move_nm * sin(radians(90.0 - self.heading_deg))),
            )
            if not self._move_is_safe(self.position, proposed_position):
                safe_position = None
                retry_move_nm = move_nm
                for _ in range(4):
                    retry_move_nm *= 0.5
                    candidate = (
                        self.position[0] + (retry_move_nm * cos(radians(90.0 - self.heading_deg))),
                        self.position[1] + (retry_move_nm * sin(radians(90.0 - self.heading_deg))),
                    )
                    if self._move_is_safe(self.position, candidate):
                        safe_position = candidate
                        move_nm = retry_move_nm
                        break
                if safe_position is None:
                    self._target_waypoint_index = self.lane.next_index(self._target_waypoint_index)
                    break
                proposed_position = safe_position
            self.position = proposed_position
            remaining_nm -= move_nm
            if move_nm >= distance:
                if (
                    destination_waypoint_index is not None
                    and current_waypoint_index == destination_waypoint_index
                ):
                    self.reached_destination_this_tick = True
                    break
                self._target_waypoint_index = self.lane.next_index(self._target_waypoint_index)
            else:
                break

    def _mesh_observations(self, current_tick: int) -> list[tuple[float, int, int]]:
        """Convert inbox packets into confidence-weighter tuples."""
        observations = []
        for packet in self.inbox:
            age = max(0, current_tick - packet.tick_sent)
            observations.append((packet.observed_hazard, age, packet.hop_count))
        return observations

    def step(self) -> None:
        """Advance vessel behaviour by one macro tick."""
        if self.state in {VesselState.SUNK, VesselState.RESCUED}:
            return
        self.reached_destination_this_tick = False

        w_true = self.weather_field.hazard_at(*self.position)
        self.last_true_hazard = w_true
        self.last_error_probability = self.error_prob_model.compute(
            hours_awake=self.hours_awake,
            t_utc_hours=self.model.utc_hours,
            archetype_modifier=self.archetype_modifier,
        )
        distance_to_shore = min(
            dist(self.position, station_position)
            for station_position in self.shore_station_positions
        )
        self.last_distance_to_shore = distance_to_shore
        if self.shore_radio.attempt_receive(distance_nm=distance_to_shore, local_hazard=w_true):
            # Human error can cause dropped interpretation even when radio receives.
            if self.rng.random() >= self.last_error_probability:
                self.w_hat_shore = self.shore_radio.broadcast(true_hazard=w_true)
                self.shore_age_ticks = 1
                self.last_shore_received = True
            else:
                self.shore_age_ticks += 1
                self.last_shore_received = False
        else:
            self.shore_age_ticks += 1
            self.last_shore_received = False

        mesh_input = self._mesh_observations(current_tick=self.model.tick)
        self.last_mesh_observation_count = len(mesh_input)
        w_hat_ship = self.confidence_weighter.fuse(mesh_input)
        self.w_hat_blend = self.forecast_fuser.fuse(
            w_hat_ship, self.w_hat_shore, self.shore_age_ticks
        )
        self.forecast_error = abs(self.w_hat_blend - w_true)
        self.p_prep = self.preparedness_scorer.score(self.forecast_error, self.archetype_modifier)

        if self.state == VesselState.ACTIVE:
            prep_for_evac = self.p_prep * 0.5
            evac_probability_raw = self.evacuation_policy.evacuation_probability(
                w_hat_blend=self.w_hat_blend,
                forecast_error=self.forecast_error,
                p_prep=prep_for_evac,
            )
            # Avoid mass instantaneous evacuations: ramp risk response over early ticks
            # and require higher hazard confidence for full-strength triggering.
            temporal_ramp = min(1.0, max(0.05, float(self.model.tick + 1) / 36.0))
            hazard_ramp = min(1.0, max(0.10, self.w_hat_blend / 0.75))
            severity_shape = min(1.0, max(0.03, (self.w_hat_blend / 0.80) ** 3))
            ramped_hazard = float(self.w_hat_blend * temporal_ramp * hazard_ramp * severity_shape)
            evac_probability = self.evacuation_policy.evacuation_probability(
                w_hat_blend=ramped_hazard,
                forecast_error=self.forecast_error,
                p_prep=prep_for_evac,
            )
            evacuate = self.evacuation_policy.should_evacuate(
                w_hat_blend=ramped_hazard,
                forecast_error=self.forecast_error,
                p_prep=prep_for_evac,
            )
            if evacuate and self.rng.random() < self.last_error_probability:
                evacuate = False
            if evacuate and self.raft_model.deploy(current_hazard=w_true, p_prep=self.p_prep):
                self.state = VesselState.EVAC
                self.has_evacuated = True
                self.sos_reason = (
                    "policy_trigger:"
                    f" p_raw={evac_probability_raw:.2f},"
                    f" p_eff={evac_probability:.2f},"
                    f" hazard={self.w_hat_blend:.2f},"
                    f" err={self.forecast_error:.2f},"
                    f" prep={self.p_prep:.2f}"
                )
            else:
                self._navigate_lane()
        elif self.state == VesselState.EVAC:
            survivors = 0
            for _ in range(self.n_survivors):
                if self.survival_model.survives_tick(current_hazard=w_true):
                    survivors += 1
            self.n_survivors = survivors
            if self.n_survivors == 0:
                self.state = VesselState.SUNK

        self.hours_awake += MACRO_TICK_HOURS
        self.inbox.clear()
