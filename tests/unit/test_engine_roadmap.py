"""Unit tests for roadmap-driven engine improvements."""

from dataclasses import replace

import pytest

from maritime_mesh.communication.packet import SosPacket
from maritime_mesh.config import ScenarioConfig, SimulationConfig
from maritime_mesh.enums import MethodCondition, VesselState
from maritime_mesh.model.maritime_model import MaritimeModel


def test_config_validation_rejects_invalid_green_fraction() -> None:
    """ScenarioConfig should reject crew fractions outside [0, 1]."""
    with pytest.raises(ValueError, match="green_crew_fraction"):
        ScenarioConfig(
            name="invalid",
            n_vessels=5,
            shore_noise_std=0.2,
            green_crew_fraction=1.5,
        )


def test_spawn_constraints_raise_when_infeasible(default_simulation_config) -> None:
    """Model should fail loudly for infeasible spawn constraints."""
    bad_scenario = replace(
        default_simulation_config.scenario,
        min_spawn_distance_nm=1000.0,
        max_spawn_distance_nm=1500.0,
    )
    bad_config = replace(default_simulation_config, scenario=bad_scenario)
    with pytest.raises(ValueError, match="infeasible"):
        MaritimeModel(bad_config)


def test_mesh_relay_applies_hop_limit(default_simulation_config) -> None:
    """Mesh packets should honor configured max hop count."""
    config = replace(default_simulation_config, max_hop_count=1, vessel_radio_range_nm=200.0)
    model = MaritimeModel(config)
    for idx, vessel in enumerate(model.vessels):
        vessel.position = (10.0 + idx, 50.0)
        vessel.state = VesselState.ACTIVE

    model._relay_packets()
    hop_counts = [packet.hop_count for vessel in model.vessels for packet in vessel.inbox]
    assert hop_counts
    assert max(hop_counts) <= 1


def test_dispatch_rescue_is_idempotent(default_simulation_config) -> None:
    """Repeated SOS for the same vessel should not spawn duplicate rescue assets."""
    model = MaritimeModel(default_simulation_config)
    vessel = model.vessels[0]
    vessel.state = VesselState.EVAC
    packet = SosPacket(sender_id=vessel.unique_id, position=vessel.position, tick_sent=model.tick)

    model.dispatch_rescue(packet, dispatch_position=model.coastal_station.position)
    first_count = len(model.rescue_agents)
    model.dispatch_rescue(packet, dispatch_position=model.coastal_station.position)

    assert len(model.rescue_agents) == first_count


def test_terminal_vessels_are_recycled_to_maintain_population(default_simulation_config) -> None:
    """Terminal vessel states should be despawned and replaced with fresh vessels."""
    model = MaritimeModel(default_simulation_config)
    initial_population = default_simulation_config.scenario.n_vessels
    original_ids = {vessel.unique_id for vessel in model.vessels}

    model.vessels[0].state = VesselState.SUNK
    model.vessels[1].state = VesselState.RESCUED
    removed_ids = {model.vessels[0].unique_id, model.vessels[1].unique_id}

    model._recycle_terminal_vessels()

    assert len(model.vessels) == initial_population
    assert all(
        vessel.state not in {VesselState.SUNK, VesselState.RESCUED} for vessel in model.vessels
    )
    assert removed_ids.isdisjoint({vessel.unique_id for vessel in model.vessels})
    assert len({vessel.unique_id for vessel in model.vessels} - original_ids) == len(removed_ids)


def test_recycling_keeps_population_stable_for_high_density_runs(default_simulation_config) -> None:
    """High-density fleets should keep constant population after repeated terminal churn."""
    dense_scenario = replace(default_simulation_config.scenario, n_vessels=100)
    dense_config = replace(default_simulation_config, scenario=dense_scenario, n_ticks=64)
    model = MaritimeModel(dense_config)

    for _ in range(8):
        for vessel in model.vessels[:10]:
            vessel.state = VesselState.SUNK
        for vessel in model.vessels[10:15]:
            vessel.state = VesselState.RESCUED
        model.step()
        assert len(model.vessels) == 100
        assert all(
            vessel.state not in {VesselState.SUNK, VesselState.RESCUED} for vessel in model.vessels
        )


def test_phased_scheduler_orders_shore_before_vessel(default_simulation_config) -> None:
    """Phased scheduler should execute shore phase before vessel phase."""
    config = replace(default_simulation_config, scheduler_mode="phased")
    model = MaritimeModel(config)
    order: list[str] = []

    for station in model.coastal_stations:
        original = station.step

        def station_step(orig=original):
            order.append("shore")
            return orig()

        station.step = station_step
    for vessel in model.vessels:
        original = vessel.step

        def vessel_step(orig=original):
            order.append("vessel")
            return orig()

        vessel.step = vessel_step

    model.scheduler.step()
    assert "shore" in order and "vessel" in order
    assert order.index("shore") < order.index("vessel")


def test_simulation_config_validation_rejects_packet_loss(tmp_path) -> None:
    """SimulationConfig should reject invalid packet-loss probabilities."""
    scenario = ScenarioConfig(
        name="valid",
        n_vessels=5,
        shore_noise_std=0.2,
        green_crew_fraction=0.4,
    )
    with pytest.raises(ValueError, match="radio_packet_loss_rate"):
        SimulationConfig(
            scenario=scenario,
            method=MethodCondition.PROPOSED,
            seed=1,
            n_ticks=10,
            mesh_enabled=True,
            evacuation_enabled=True,
            human_factors_enabled=True,
            output_dir=tmp_path,
            radio_packet_loss_rate=2.0,
        )
