import numpy as np

from maritime_mesh.behaviour.raft import RaftDeploymentModel, SurvivalModel


def test_deploy_high_probability_case(seeded_rng: np.random.Generator) -> None:
    model = RaftDeploymentModel(seeded_rng)
    successes = sum(model.deploy(current_hazard=0.0, p_prep=1.0) for _ in range(100))
    assert successes > 70


def test_survival_drops_with_hazard(seeded_rng: np.random.Generator) -> None:
    model = SurvivalModel(seeded_rng)
    rng2 = np.random.default_rng(1234)
    model2 = SurvivalModel(rng2)
    calm = sum(model.survives_tick(0.1) for _ in range(300))
    rough = sum(model2.survives_tick(0.9) for _ in range(300))
    assert calm > rough
