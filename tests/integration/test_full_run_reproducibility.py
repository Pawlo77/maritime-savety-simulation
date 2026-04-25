from maritime_mesh.enums import MethodCondition
from maritime_mesh.experiment.scenarios import scenario_2_storm_corridor
from maritime_mesh.model.maritime_model import MaritimeModel


def test_same_seed_reproduces_identical_kpis() -> None:
    config_a = scenario_2_storm_corridor(method=MethodCondition.PROPOSED, seed=11)
    config_b = scenario_2_storm_corridor(method=MethodCondition.PROPOSED, seed=11)
    result_a = MaritimeModel(config_a).run()
    result_b = MaritimeModel(config_b).run()
    rounded_a = {k: round(v, 6) for k, v in result_a.items()}
    rounded_b = {k: round(v, 6) for k, v in result_b.items()}
    assert rounded_a == rounded_b
