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
    SHORE_STATION_POSITION,
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
        self.last_true_hazard = 0.0
        self.last_mesh_observation_count = 0
        self.last_shore_received = False
        self.last_distance_to_shore = 0.0
        self.last_error_probability = 0.0

    def _navigate_lane(self) -> None:
        """Move vessel toward lane waypoint by one macro-tick step."""
        waypoint = self.lane.next_waypoint(self.position)
        dx = waypoint.x_nm - self.position[0]
        dy = waypoint.y_nm - self.position[1]
        distance = dist(self.position, (waypoint.x_nm, waypoint.y_nm))
        if distance == 0.0:
            return
        step_nm = self.speed_kn * MACRO_TICK_HOURS
        move_nm = min(distance, step_nm)
        angle = atan2(dy, dx)
        self.heading_deg = (90.0 - np.degrees(angle)) % 360.0
        self.position = (
            self.position[0] + (move_nm * cos(radians(90.0 - self.heading_deg))),
            self.position[1] + (move_nm * sin(radians(90.0 - self.heading_deg))),
        )

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

        w_true = self.weather_field.hazard_at(*self.position)
        self.last_true_hazard = w_true
        distance_to_shore = dist(self.position, SHORE_STATION_POSITION)
        self.last_distance_to_shore = distance_to_shore
        if self.shore_radio.attempt_receive(distance_nm=distance_to_shore, local_hazard=w_true):
            self.w_hat_shore = self.shore_radio.broadcast(true_hazard=w_true)
            self.shore_age_ticks = 1
            self.last_shore_received = True
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
        self.last_error_probability = self.error_prob_model.compute(
            hours_awake=self.hours_awake,
            t_utc_hours=self.model.utc_hours,
            archetype_modifier=self.archetype_modifier,
        )
        self.p_prep = self.preparedness_scorer.score(self.forecast_error, self.archetype_modifier)

        if self.state == VesselState.ACTIVE:
            evacuate = self.evacuation_policy.should_evacuate(
                w_hat_blend=self.w_hat_blend, forecast_error=self.forecast_error, p_prep=self.p_prep
            )
            if evacuate and self.raft_model.deploy(current_hazard=w_true, p_prep=self.p_prep):
                self.state = VesselState.EVAC
                self.has_evacuated = True
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
