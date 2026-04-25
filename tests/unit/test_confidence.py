from maritime_mesh.fusion.confidence import ConfidenceWeighter


def test_weight_origin_is_one() -> None:
    weighter = ConfidenceWeighter()
    assert weighter.weight(0, 0) == 1.0


def test_weight_decays_with_age_and_hops() -> None:
    weighter = ConfidenceWeighter()
    assert weighter.weight(2, 1) < weighter.weight(1, 1)
    assert weighter.weight(1, 2) < weighter.weight(1, 1)


def test_fuse_empty_returns_none() -> None:
    weighter = ConfidenceWeighter()
    assert weighter.fuse([]) is None
