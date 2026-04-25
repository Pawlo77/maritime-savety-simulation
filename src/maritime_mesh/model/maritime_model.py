"""Root Mesa model for maritime weather mesh simulation."""

from math import dist

import numpy as np

from maritime_mesh.agents.coastal_station_agent import CoastalStationAgent
from maritime_mesh.agents.rescue_agent import RescueAgent
from maritime_mesh.agents.vessel_agent import VesselAgent
from maritime_mesh.behaviour.evacuation import EvacuationPolicy
from maritime_mesh.behaviour.human_factors import ErrorProbabilityModel
from maritime_mesh.behaviour.preparedness import PreparednessScorer
from maritime_mesh.behaviour.raft import RaftDeploymentModel, SurvivalModel
from maritime_mesh.communication.mesh_relay import MeshRelayProtocol
from maritime_mesh.communication.packet import MeshPacket, SosPacket
from maritime_mesh.communication.shore_radio import ShoreRadioModel
from maritime_mesh.config import SimulationConfig
from maritime_mesh.constants import MACRO_TICK_HOURS, WORLD_SIZE_NM
from maritime_mesh.enums import CrewArchetype, MethodCondition, RescueAssetType, VesselState
from maritime_mesh.fusion.confidence import ConfidenceWeighter
from maritime_mesh.fusion.shore_trust import ForecastFuser, ShoreTrustDecay
from maritime_mesh.mesa_compat import Model, RandomActivation
from maritime_mesh.model.kpi_logger import KpiLogger
from maritime_mesh.weather.weather_field import WeatherField
from maritime_mesh.world.collision_detector import CollisionDetector
from maritime_mesh.world.lane import ShippingLane, Waypoint


class MaritimeModel(Model):
    """Orchestrates all agents, weather, relay, collisions, and KPI logging."""

    def __init__(self, config: SimulationConfig) -> None:
        """Initialize model components from immutable run config."""
        super().__init__()
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.weather_field = WeatherField(rng=self.rng)
        self.scheduler = RandomActivation(self)
        self.vessels: list[VesselAgent] = []
        self.rescue_agents: list[RescueAgent] = []
        self.collision_detector = CollisionDetector()
        self.kpi_logger = KpiLogger()
        self.tick = 0
        self.utc_hours = 0.0
        self._next_id = 1
        self._lanes = self._build_lanes()
        self._spawn_agents()

    def _new_id(self) -> int:
        """Return next deterministic agent identifier."""
        current = self._next_id
        self._next_id += 1
        return current

    def _build_lanes(self) -> list[ShippingLane]:
        """Construct canonical lanes used by vessel spawns."""
        return [
            ShippingLane("north_south", [Waypoint(20.0, 0.0), Waypoint(20.0, WORLD_SIZE_NM)]),
            ShippingLane("east_west", [Waypoint(0.0, 60.0), Waypoint(WORLD_SIZE_NM, 60.0)]),
            ShippingLane("diagonal", [Waypoint(10.0, 10.0), Waypoint(90.0, 90.0)]),
        ]

    def _spawn_agents(self) -> None:
        """Create coastal station and vessel population."""
        shore_radio = ShoreRadioModel(
            rng=self.rng,
            noise_std=self.config.scenario.shore_noise_std,
        )
        self.coastal_station = CoastalStationAgent(
            model=self,
            unique_id=self._new_id(),
            rng=self.rng,
            shore_radio=shore_radio,
            weather_field=self.weather_field,
        )
        self.scheduler.add(self.coastal_station)

        for _ in range(self.config.scenario.n_vessels):
            lane = self.rng.choice(self._lanes)
            archetype = (
                CrewArchetype.GREEN
                if self.rng.random() < self.config.scenario.green_crew_fraction
                else CrewArchetype.STANDARD
            )
            if self.rng.random() < 0.1:
                archetype = CrewArchetype.VETERAN
            preparedness = (
                PreparednessScorer(fixed_value=0.0)
                if self.config.method == MethodCondition.BASELINE_A
                else PreparednessScorer()
            )
            fuser = (
                ForecastFuser(trust_decay=ShoreTrustDecay(decay_k=1e-9))
                if self.config.method == MethodCondition.BASELINE_B
                else ForecastFuser()
            )
            vessel = VesselAgent(
                model=self,
                unique_id=self._new_id(),
                rng=self.rng,
                weather_field=self.weather_field,
                lane=lane,
                shore_radio=shore_radio,
                mesh_relay=MeshRelayProtocol(self.rng),
                confidence_weighter=ConfidenceWeighter(),
                forecast_fuser=fuser,
                error_prob_model=ErrorProbabilityModel(enabled=self.config.human_factors_enabled),
                preparedness_scorer=preparedness,
                evacuation_policy=EvacuationPolicy(
                    rng=self.rng, enabled=self.config.evacuation_enabled
                ),
                raft_model=RaftDeploymentModel(rng=self.rng),
                survival_model=SurvivalModel(rng=self.rng),
                position=(
                    float(self.rng.uniform(0.0, WORLD_SIZE_NM)),
                    float(self.rng.uniform(0.0, WORLD_SIZE_NM)),
                ),
                speed_kn=float(self.rng.uniform(10.0, 20.0)),
                archetype=archetype,
            )
            self.vessels.append(vessel)
            self.scheduler.add(vessel)

    def _relay_packets(self) -> list[tuple[int, int]]:
        """Relay local weather packets and SOS messages among nearby vessels."""
        relay_links: list[tuple[int, int]] = []
        active = [v for v in self.vessels if v.state in {VesselState.ACTIVE, VesselState.EVAC}]
        for sender in active:
            sender.mesh_relay.reset_tick()
            weather_packet = MeshPacket(
                sender_id=sender.unique_id,
                position=sender.position,
                observed_hazard=self.weather_field.hazard_at(*sender.position),
                tick_sent=self.tick,
                hop_count=0,
            )
            for receiver in active:
                if receiver.unique_id == sender.unique_id:
                    continue
                if dist(sender.position, receiver.position) <= 15.0:
                    receiver.inbox.append(weather_packet)
                    relay_links.append((sender.unique_id, receiver.unique_id))
            if sender.state == VesselState.EVAC:
                sos = SosPacket(
                    sender_id=sender.unique_id, position=sender.position, tick_sent=self.tick
                )
                self.coastal_station.receive_sos(sos)
        return relay_links

    def dispatch_rescue(self, packet: SosPacket) -> None:
        """Spawn rescue asset for SOS packet and add to scheduler."""
        asset = (
            RescueAssetType.HELICOPTER if self.rng.random() < 0.5 else RescueAssetType.PATROL_VESSEL
        )
        rescue_agent = RescueAgent(
            model=self,
            unique_id=self._new_id(),
            rng=self.rng,
            asset_type=asset,
            target_position=packet.position,
        )
        self.rescue_agents.append(rescue_agent)
        self.scheduler.add(rescue_agent)

    def step(self) -> None:
        """Advance model by one macro tick."""
        self.weather_field.step()
        if (
            self.config.scenario.hazard_spike_tick is not None
            and self.tick == self.config.scenario.hazard_spike_tick
            and self.config.scenario.hazard_spike_value is not None
        ):
            self.weather_field.inject_hazard_spike(25, 25, self.config.scenario.hazard_spike_value)
        if (
            self.config.scenario.noise_double_tick is not None
            and self.tick == self.config.scenario.noise_double_tick
        ):
            self.coastal_station.shore_radio.noise_std *= 2.0

        relay_links = self._relay_packets()
        self.scheduler.step()
        collisions = self.collision_detector.check_and_apply(self.vessels)
        self.kpi_logger.add_collisions(len(collisions))
        self.kpi_logger.log_tick(
            tick=self.tick,
            vessels=self.vessels,
            coastal_station=self.coastal_station,
            rescue_agents=self.rescue_agents,
            relay_links=relay_links,
            collisions=collisions,
            weather_field=self.weather_field,
        )
        self.tick += 1
        self.utc_hours += MACRO_TICK_HOURS

    def run(self) -> dict[str, float]:
        """Run configured number of ticks and return run KPIs."""
        for _ in range(self.config.n_ticks):
            self.step()
        return self.kpi_logger.compute_kpis()
