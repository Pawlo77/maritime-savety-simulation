import pytest

from maritime_mesh.fusion.shore_trust import ForecastFuser, ShoreTrustDecay


def test_compute_small_age_near_zero() -> None:
    model = ShoreTrustDecay()
    assert model.compute(1) < 0.02


def test_compute_large_age_tends_to_one() -> None:
    model = ShoreTrustDecay()
    assert model.compute(100) == pytest.approx(1.0, rel=0.05)


def test_fuser_returns_shore_when_mesh_none() -> None:
    fuser = ForecastFuser()
    assert fuser.fuse(None, 0.3, 5) == 0.3
