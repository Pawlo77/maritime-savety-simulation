"""Unit tests for coastal station agent behaviour."""

from dataclasses import replace

from maritime_mesh.communication.packet import SosPacket
from maritime_mesh.model.maritime_model import MaritimeModel


def test_receive_sos_enqueues(default_simulation_config) -> None:
    """Ensure incoming SOS packets are queued for later dispatch."""
    model = MaritimeModel(default_simulation_config)
    packet = SosPacket(sender_id=1, position=(1.0, 1.0), tick_sent=1)
    model.coastal_station.receive_sos(packet)
    assert len(model.coastal_station.sos_queue) == 1


def test_step_processes_sos_queue(default_simulation_config) -> None:
    """Ensure step consumes queued SOS packets."""
    model = MaritimeModel(default_simulation_config)
    packet = SosPacket(sender_id=1, position=(1.0, 1.0), tick_sent=1)
    model.coastal_station.receive_sos(packet)
    model.coastal_station.step()
    assert len(model.coastal_station.sos_queue) == 0


def test_rescue_spawns_from_station_that_received_sos(default_simulation_config) -> None:
    """Rescue asset should launch from the shore station receiving the SOS."""
    config = replace(
        default_simulation_config,
        shore_station_positions=((0.0, 50.0), (5.0, 20.0)),
    )
    model = MaritimeModel(config)
    packet = SosPacket(sender_id=999, position=(82.0, 22.0), tick_sent=1)
    receiving_station = model.coastal_stations[1]

    receiving_station.receive_sos(packet)
    receiving_station.step()

    assert model.rescue_agents
    assert model.rescue_agents[-1].position == receiving_station.position


def test_sos_routed_to_station_that_receives(default_simulation_config) -> None:
    """SOS should be queued on the station where reception succeeds."""
    config = replace(
        default_simulation_config,
        shore_station_positions=((0.0, 50.0), (5.0, 20.0)),
    )
    model = MaritimeModel(config)

    model.coastal_stations[0].shore_radio.attempt_receive = lambda **_kwargs: False
    model.coastal_stations[1].shore_radio.attempt_receive = lambda **_kwargs: True
    packet = SosPacket(sender_id=77, position=(7.0, 20.0), tick_sent=3)

    receiver_position = model._route_sos_to_station(packet)

    assert receiver_position == model.coastal_stations[1].position
    assert len(model.coastal_stations[1].sos_queue) == 1
    assert len(model.coastal_stations[0].sos_queue) == 0


def test_receive_sos_deduplicates_same_packet(default_simulation_config) -> None:
    """Station queue should ignore repeated identical SOS packets."""
    model = MaritimeModel(default_simulation_config)
    packet = SosPacket(sender_id=5, position=(4.0, 9.0), tick_sent=3)
    model.coastal_station.receive_sos(packet)
    model.coastal_station.receive_sos(packet)
    assert len(model.coastal_station.sos_queue) == 1
