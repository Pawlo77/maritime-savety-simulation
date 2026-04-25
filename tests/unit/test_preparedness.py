"""Unit tests for preparedness score calculation behaviour."""

import pytest

from maritime_mesh.behaviour.preparedness import PreparednessScorer


def test_score_identity_at_zero_inputs() -> None:
    """Verify perfect conditions map to maximum preparedness."""
    scorer = PreparednessScorer()
    assert scorer.score(0.0, 0.0) == pytest.approx(1.0)


def test_fixed_value_overrides_inputs() -> None:
    """Verify fixed-value mode ignores dynamic score inputs."""
    scorer = PreparednessScorer(fixed_value=0.0)
    assert scorer.score(0.9, 2.0) == 0.0


def test_score_is_clipped() -> None:
    """Verify computed preparedness remains in closed unit interval."""
    scorer = PreparednessScorer()
    assert 0.0 <= scorer.score(10.0, 10.0) <= 1.0
