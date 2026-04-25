import numpy as np

from maritime_mesh.communication.shore_radio import ShoreRadioModel


def test_broadcast_clipped_bounds(seeded_rng: np.random.Generator) -> None:
    radio = ShoreRadioModel(rng=seeded_rng, noise_std=10.0)
    value = radio.broadcast(0.9)
    assert 0.0 <= value <= 1.0


def test_receive_probability_distance_drop(seeded_rng: np.random.Generator) -> None:
    radio = ShoreRadioModel(rng=seeded_rng)
    near = radio.receive_probability(distance_nm=0.0, local_hazard=0.0)
    far = radio.receive_probability(distance_nm=300.0, local_hazard=0.0)
    assert near > far


def test_weather_interference_reduces_probability(seeded_rng: np.random.Generator) -> None:
    radio = ShoreRadioModel(rng=seeded_rng)
    calm = radio.receive_probability(distance_nm=25.0, local_hazard=0.0)
    rough = radio.receive_probability(distance_nm=25.0, local_hazard=1.0)
    assert rough < calm
