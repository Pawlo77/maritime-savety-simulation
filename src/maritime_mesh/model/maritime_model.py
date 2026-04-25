"""Root Mesa model for maritime weather mesh simulation."""

from collections import deque
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
from maritime_mesh.constants import MACRO_TICK_HOURS
from maritime_mesh.enums import CrewArchetype, MethodCondition, RescueAssetType, VesselState
from maritime_mesh.fusion.confidence import ConfidenceWeighter
from maritime_mesh.fusion.shore_trust import ForecastFuser, ShoreTrustDecay
from maritime_mesh.mesa_compat import Model, PhaseScheduler, RandomActivation
from maritime_mesh.model.kpi_logger import KpiLogger
from maritime_mesh.weather.weather_field import WeatherField
from maritime_mesh.world.collision_detector import CollisionDetector
from maritime_mesh.world.land import WorldLand
from maritime_mesh.world.lane import ShippingLane, Waypoint


class MaritimeModel(Model):
    """Orchestrates all agents, weather, relay, collisions, and KPI logging."""

    def __init__(self, config: SimulationConfig) -> None:
        """Initialize model components from immutable run config."""
        super().__init__()
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.weather_field = WeatherField(rng=self.rng, world_size_nm=config.world_size_nm)
        self.world_size_nm = config.world_size_nm
        self.land = WorldLand.default_for_world_size(self.world_size_nm)
        self.shore_station_positions = (
            config.shore_station_positions
            if config.shore_station_positions
            else (config.shore_station_position,)
        )
        self._validate_shore_station_positions()
        self.shore_station_position = self.shore_station_positions[0]
        self.scheduler = (
            PhaseScheduler(self, phases=("shore", "vessel", "rescue"))
            if self.config.scheduler_mode == "phased"
            else RandomActivation(self)
        )
        self.vessels: list[VesselAgent] = []
        self.rescue_agents: list[RescueAgent] = []
        self.collision_detector = CollisionDetector()
        self.kpi_logger = KpiLogger()
        self.tick = 0
        self.utc_hours = 0.0
        self._next_id = 1
        self._sos_dispatch_tick_by_vessel: dict[int, int] = {}
        self._sos_dispatch_station_by_vessel: dict[int, tuple[float, float]] = {}
        self._active_rescue_by_vessel: dict[int, int] = {}
        self._recorded_rescue_vessels: set[int] = set()
        self._lanes = self._build_lanes()
        self._validate_lane_geometry()
        self._spawn_agents()

    def _new_id(self) -> int:
        """Return next deterministic agent identifier."""
        current = self._next_id
        self._next_id += 1
        return current

    def _build_lanes(self) -> list[ShippingLane]:
        """Construct canonical lanes used by vessel spawns."""
        if self.config.lane_definitions:
            return [
                ShippingLane(
                    lane_name,
                    [Waypoint(x_nm=waypoint[0], y_nm=waypoint[1]) for waypoint in waypoints],
                )
                for lane_name, waypoints in self.config.lane_definitions
            ]
        return [
            ShippingLane(
                "coastal_corridor",
                [
                    Waypoint(self.world_size_nm * 0.16, self.world_size_nm * 0.12),
                    Waypoint(self.world_size_nm * 0.22, self.world_size_nm * 0.32),
                    Waypoint(self.world_size_nm * 0.20, self.world_size_nm * 0.54),
                    Waypoint(self.world_size_nm * 0.24, self.world_size_nm * 0.86),
                ],
            ),
            ShippingLane(
                "southern_arc",
                [
                    Waypoint(self.world_size_nm * 0.16, self.world_size_nm * 0.14),
                    Waypoint(self.world_size_nm * 0.36, self.world_size_nm * 0.18),
                    Waypoint(self.world_size_nm * 0.58, self.world_size_nm * 0.28),
                    Waypoint(self.world_size_nm * 0.86, self.world_size_nm * 0.36),
                ],
            ),
            ShippingLane(
                "northern_bypass",
                [
                    Waypoint(self.world_size_nm * 0.16, self.world_size_nm * 0.70),
                    Waypoint(self.world_size_nm * 0.38, self.world_size_nm * 0.72),
                    Waypoint(self.world_size_nm * 0.58, self.world_size_nm * 0.70),
                    Waypoint(self.world_size_nm * 0.78, self.world_size_nm * 0.84),
                    Waypoint(self.world_size_nm * 0.94, self.world_size_nm * 0.84),
                ],
            ),
        ]

    def _validate_shore_station_positions(self) -> None:
        """Ensure all configured shore stations are positioned on land."""
        for position in self.shore_station_positions:
            if not self.land.is_land(position):
                raise ValueError(f"Shore station at {position} must be placed on land.")

    def _validate_lane_geometry(self) -> None:
        """Ensure lane segments do not intersect static land geometry."""
        for lane in self._lanes:
            for idx in range(len(lane.waypoints) - 1):
                start = (lane.waypoints[idx].x_nm, lane.waypoints[idx].y_nm)
                end = (lane.waypoints[idx + 1].x_nm, lane.waypoints[idx + 1].y_nm)
                if self.land.segment_intersects_land(start, end):
                    raise ValueError(
                        f"Lane '{lane.name}' intersects land between {start} and {end}."
                    )

    def _spawn_agents(self) -> None:
        """Create coastal station and vessel population."""
        self.coastal_stations: list[CoastalStationAgent] = []
        for position in self.shore_station_positions:
            shore_radio = ShoreRadioModel(
                rng=self.rng,
                noise_std=self.config.scenario.shore_noise_std,
                broadcast_radius_nm=self.config.shore_broadcast_radius_nm,
                range_falloff=self.config.radio_range_falloff,
                weather_interference=self.config.radio_weather_interference,
                packet_loss_rate=self.config.radio_packet_loss_rate,
            )
            station = CoastalStationAgent(
                model=self,
                unique_id=self._new_id(),
                rng=self.rng,
                shore_radio=shore_radio,
                weather_field=self.weather_field,
                position=position,
            )
            self.coastal_stations.append(station)
            if isinstance(self.scheduler, PhaseScheduler):
                self.scheduler.add("shore", station)
            else:
                self.scheduler.add(station)
        # Backward-compatible alias used in tests and downstream code.
        self.coastal_station = self.coastal_stations[0]

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
                shore_radio=self.coastal_station.shore_radio,
                mesh_relay=MeshRelayProtocol(self.rng, max_hop_count=self.config.max_hop_count),
                confidence_weighter=ConfidenceWeighter(),
                forecast_fuser=fuser,
                error_prob_model=ErrorProbabilityModel(enabled=self.config.human_factors_enabled),
                preparedness_scorer=preparedness,
                evacuation_policy=EvacuationPolicy(
                    rng=self.rng, enabled=self.config.evacuation_enabled
                ),
                raft_model=RaftDeploymentModel(rng=self.rng),
                survival_model=SurvivalModel(rng=self.rng),
                position=self._sample_spawn_position(),
                speed_kn=float(self.rng.uniform(10.0, 20.0)),
                archetype=archetype,
                shore_station_positions=self.shore_station_positions,
            )
            self.vessels.append(vessel)
            if isinstance(self.scheduler, PhaseScheduler):
                self.scheduler.add("vessel", vessel)
            else:
                self.scheduler.add(vessel)

    def _sample_spawn_position(self) -> tuple[float, float]:
        """Draw vessel spawn position satisfying scenario shore-distance constraints."""
        min_distance = max(0.0, self.config.scenario.min_spawn_distance_nm)
        max_distance = max(min_distance, self.config.scenario.max_spawn_distance_nm)
        if min_distance > self.world_size_nm * 2:
            raise ValueError("Spawn constraints are infeasible for current world size.")
        for _ in range(200):
            candidate = (
                float(self.rng.uniform(0.0, self.world_size_nm)),
                float(self.rng.uniform(0.0, self.world_size_nm)),
            )
            if self.land.is_land(candidate):
                continue
            distance_to_shore = min(
                dist(candidate, station_position)
                for station_position in self.shore_station_positions
            )
            if min_distance <= distance_to_shore <= max_distance:
                return candidate
        raise ValueError(
            "Failed to sample vessel spawn position satisfying constraints after 200 attempts."
        )

    def _route_sos_to_station(self, packet: SosPacket) -> tuple[float, float] | None:
        """Attempt SOS delivery to shore stations; return receiver position."""
        local_hazard = self.weather_field.hazard_at(*packet.position)
        stations_by_distance = sorted(
            self.coastal_stations,
            key=lambda station: dist(packet.position, station.position),
        )
        for station in stations_by_distance:
            distance_nm = dist(packet.position, station.position)
            if station.shore_radio.attempt_receive(
                distance_nm=distance_nm, local_hazard=local_hazard
            ):
                station.receive_sos(packet)
                return station.position
        return None

    def _relay_packets(self) -> list[tuple[int, int]]:
        """Relay local weather packets and SOS messages among nearby vessels."""
        relay_links: list[tuple[int, int]] = []
        active = [v for v in self.vessels if v.state in {VesselState.ACTIVE, VesselState.EVAC}]
        vessel_by_id = {vessel.unique_id: vessel for vessel in active}
        for vessel in active:
            vessel.mesh_relay.reset_tick()
        if not self.config.mesh_enabled:
            for vessel in active:
                if vessel.state == VesselState.EVAC and not vessel.sos_sent:
                    receiver_position = self._route_sos_to_station(
                        SosPacket(
                            sender_id=vessel.unique_id,
                            position=vessel.position,
                            tick_sent=self.tick,
                        )
                    )
                    if receiver_position is not None:
                        vessel.sos_sent = True
            return relay_links
        weather_queue: deque[tuple[int, MeshPacket]] = deque()
        sos_queue: deque[tuple[int, SosPacket]] = deque()
        for sender in active:
            weather_queue.append(
                (
                    sender.unique_id,
                    MeshPacket(
                        sender_id=sender.unique_id,
                        position=sender.position,
                        observed_hazard=self.weather_field.hazard_at(*sender.position),
                        tick_sent=self.tick,
                        hop_count=0,
                    ),
                )
            )
            if sender.state == VesselState.EVAC and not sender.sos_sent:
                sos_queue.append(
                    (
                        sender.unique_id,
                        SosPacket(
                            sender_id=sender.unique_id,
                            position=sender.position,
                            tick_sent=self.tick,
                        ),
                    )
                )

        while weather_queue:
            transmitter_id, packet = weather_queue.popleft()
            transmitter = vessel_by_id.get(transmitter_id)
            if transmitter is None:
                continue
            for receiver in active:
                if receiver.unique_id == transmitter_id:
                    continue
                if (
                    dist(transmitter.position, receiver.position)
                    > self.config.vessel_radio_range_nm
                ):
                    continue
                if not receiver.mesh_relay.should_relay(packet):
                    continue
                relayed_packet = receiver.mesh_relay.relay(packet)
                receiver.inbox.append(relayed_packet)
                relay_links.append((transmitter_id, receiver.unique_id))
                if relayed_packet.hop_count < self.config.max_hop_count:
                    weather_queue.append((receiver.unique_id, relayed_packet))

        delivered_sos: set[int] = set()
        while sos_queue:
            transmitter_id, packet = sos_queue.popleft()
            if packet.sender_id in delivered_sos:
                continue
            receiver_position = self._route_sos_to_station(packet)
            if receiver_position is not None:
                source_vessel = vessel_by_id.get(packet.sender_id)
                if source_vessel is not None:
                    source_vessel.sos_sent = True
                delivered_sos.add(packet.sender_id)
                continue
            transmitter = vessel_by_id.get(transmitter_id)
            if transmitter is None:
                continue
            for receiver in active:
                if receiver.unique_id == transmitter_id:
                    continue
                if (
                    dist(transmitter.position, receiver.position)
                    > self.config.vessel_radio_range_nm
                ):
                    continue
                if not receiver.mesh_relay.should_relay(packet):
                    continue
                relayed_packet = receiver.mesh_relay.relay(packet)
                relay_links.append((transmitter_id, receiver.unique_id))
                if relayed_packet.hop_count < self.config.max_hop_count:
                    sos_queue.append((receiver.unique_id, relayed_packet))
        return relay_links

    def dispatch_rescue(self, packet: SosPacket, dispatch_position: tuple[float, float]) -> None:
        """Spawn rescue asset for SOS packet and add to scheduler."""
        if packet.sender_id in self._active_rescue_by_vessel:
            return
        vessel = next(
            (candidate for candidate in self.vessels if candidate.unique_id == packet.sender_id),
            None,
        )
        if vessel is not None and vessel.state in {VesselState.RESCUED, VesselState.SUNK}:
            return
        if packet.sender_id not in self._sos_dispatch_tick_by_vessel:
            self._sos_dispatch_tick_by_vessel[packet.sender_id] = self.tick
            self._sos_dispatch_station_by_vessel[packet.sender_id] = dispatch_position
        asset = (
            RescueAssetType.HELICOPTER if self.rng.random() < 0.5 else RescueAssetType.PATROL_VESSEL
        )
        rescue_agent = RescueAgent(
            model=self,
            unique_id=self._new_id(),
            rng=self.rng,
            asset_type=asset,
            target_position=packet.position,
            start_position=dispatch_position,
        )
        self.rescue_agents.append(rescue_agent)
        self._active_rescue_by_vessel[packet.sender_id] = rescue_agent.unique_id
        rescue_agent.target_vessel_id = packet.sender_id
        if isinstance(self.scheduler, PhaseScheduler):
            self.scheduler.add("rescue", rescue_agent)
        else:
            self.scheduler.add(rescue_agent)

    def _record_rescue_arrivals(self) -> None:
        """Track time-to-arrival once a distressed vessel is rescued."""
        for vessel in self.vessels:
            if vessel.state != VesselState.RESCUED:
                continue
            if vessel.unique_id in self._recorded_rescue_vessels:
                continue
            if vessel.unique_id not in self._sos_dispatch_tick_by_vessel:
                continue
            dispatch_tick = self._sos_dispatch_tick_by_vessel[vessel.unique_id]
            elapsed_ticks = max(0, self.tick - dispatch_tick)
            self.kpi_logger.add_rescue_tta(elapsed_ticks * MACRO_TICK_HOURS)
            self._recorded_rescue_vessels.add(vessel.unique_id)
            self._active_rescue_by_vessel.pop(vessel.unique_id, None)

    def _land_collisions(
        self,
        previous_positions: dict[int, tuple[float, float]],
    ) -> list[tuple[int, tuple[float, float]]]:
        """Apply grounding effects when active vessels intersect land."""
        events: list[tuple[int, tuple[float, float]]] = []
        for vessel in self.vessels:
            if vessel.state != VesselState.ACTIVE:
                continue
            previous = previous_positions.get(vessel.unique_id, vessel.position)
            intersects = self.land.segment_intersects_land(previous, vessel.position)
            if not intersects:
                continue
            vessel.position = previous
            vessel.damage += 1
            vessel.state = VesselState.EVAC
            vessel.has_evacuated = True
            events.append((vessel.unique_id, vessel.position))
        return events

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
            for station in self.coastal_stations:
                station.shore_radio.noise_std *= 2.0

        previous_positions = {vessel.unique_id: vessel.position for vessel in self.vessels}
        if isinstance(self.scheduler, PhaseScheduler):
            for station in self.coastal_stations:
                station.step()
            relay_links = self._relay_packets()
            for vessel in self.vessels:
                vessel.step()
            for rescue in self.rescue_agents:
                rescue.step()
        else:
            relay_links = self._relay_packets()
            self.scheduler.step()
        collisions = self.collision_detector.check_and_apply(self.vessels)
        land_collisions = self._land_collisions(previous_positions=previous_positions)
        self._record_rescue_arrivals()
        self.kpi_logger.add_collisions(len(collisions) + len(land_collisions))
        self.kpi_logger.log_tick(
            tick=self.tick,
            vessels=self.vessels,
            coastal_stations=self.coastal_stations,
            rescue_agents=self.rescue_agents,
            relay_links=relay_links,
            collisions=collisions,
            land_collisions=land_collisions,
            land_rectangles=self.land.rectangles,
            weather_field=self.weather_field,
            world_size_nm=self.world_size_nm,
            simulation_seed=self.config.seed,
        )
        self.tick += 1
        self.utc_hours += MACRO_TICK_HOURS

    def run(self) -> dict[str, float]:
        """Run configured number of ticks and return run KPIs."""
        for _ in range(self.config.n_ticks):
            self.step()
        return self.kpi_logger.compute_kpis()
