import pytest

from maritime_mesh.behaviour.preparedness import PreparednessScorer


def test_score_identity_at_zero_inputs() -> None:
    scorer = PreparednessScorer()
    assert scorer.score(0.0, 0.0) == pytest.approx(1.0)


def test_fixed_value_overrides_inputs() -> None:
    scorer = PreparednessScorer(fixed_value=0.0)
    assert scorer.score(0.9, 2.0) == 0.0


def test_score_is_clipped() -> None:
    scorer = PreparednessScorer()
    assert 0.0 <= scorer.score(10.0, 10.0) <= 1.0
