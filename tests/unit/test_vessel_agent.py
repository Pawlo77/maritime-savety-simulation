from maritime_mesh.enums import VesselState
from maritime_mesh.model.maritime_model import MaritimeModel


def test_inbox_cleared_each_tick(default_simulation_config) -> None:
    model = MaritimeModel(default_simulation_config)
    vessel = model.vessels[0]
    vessel.inbox.append(vessel.inbox[0] if vessel.inbox else None)
    vessel.inbox = [packet for packet in vessel.inbox if packet is not None]
    vessel.step()
    assert vessel.inbox == []


def test_shore_age_ticks_increments_when_no_reception(default_simulation_config) -> None:
    model = MaritimeModel(default_simulation_config)
    vessel = model.vessels[0]

    def _never_receive(distance_nm: float, local_hazard: float) -> bool:
        _ = distance_nm, local_hazard
        return False

    vessel.shore_radio.attempt_receive = _never_receive
    before = vessel.shore_age_ticks
    vessel.step()
    assert vessel.shore_age_ticks == before + 1


def test_state_transition_to_evac_possible(default_simulation_config) -> None:
    model = MaritimeModel(default_simulation_config)
    vessel = model.vessels[0]

    def _always_evacuate(**_kwargs: object) -> bool:
        return True

    def _always_deploy(**_kwargs: object) -> bool:
        return True

    vessel.evacuation_policy.should_evacuate = _always_evacuate
    vessel.raft_model.deploy = _always_deploy
    vessel.step()
    assert vessel.state in {VesselState.EVAC, VesselState.SUNK, VesselState.RESCUED}
