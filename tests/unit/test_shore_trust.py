"""Unit tests for shore trust decay and blend fallback logic."""

import pytest

from maritime_mesh.fusion.shore_trust import ForecastFuser, ShoreTrustDecay


def test_compute_small_age_near_zero() -> None:
    """Verify trust weight stays near zero for fresh shore observations."""
    model = ShoreTrustDecay()
    assert model.compute(1) < 0.02


def test_compute_large_age_tends_to_one() -> None:
    """Verify trust weight approaches one for stale shore observations."""
    model = ShoreTrustDecay()
    assert model.compute(100) == pytest.approx(1.0, rel=0.05)


def test_fuser_returns_shore_when_mesh_none() -> None:
    """Verify fuser returns shore estimate when mesh input is unavailable."""
    fuser = ForecastFuser()
    assert fuser.fuse(None, 0.3, 5) == 0.3
