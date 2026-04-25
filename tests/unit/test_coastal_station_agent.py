"""Unit tests for coastal station agent behaviour."""

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
