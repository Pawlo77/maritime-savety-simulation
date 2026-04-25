"""Unit tests for evacuation policy decision behaviour."""

import numpy as np

from maritime_mesh.behaviour.evacuation import EvacuationPolicy


def test_disabled_policy_always_false(seeded_rng: np.random.Generator) -> None:
    """Verify ablation mode forces evacuation decisions to False."""
    policy = EvacuationPolicy(seeded_rng, enabled=False)
    assert policy.should_evacuate(1.0, 0.0, 1.0) is False


def test_high_error_reduces_evacuation_likelihood(seeded_rng: np.random.Generator) -> None:
    """Verify higher forecast error lowers evacuation activation frequency."""
    policy = EvacuationPolicy(seeded_rng)
    rng2 = np.random.default_rng(1234)
    policy2 = EvacuationPolicy(rng2)
    high_score = sum(policy.should_evacuate(0.9, 0.0, 1.0) for _ in range(200))
    low_score = sum(policy2.should_evacuate(0.9, 0.8, 1.0) for _ in range(200))
    assert high_score > low_score
