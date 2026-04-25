"""Confidence weighting of relayed weather observations."""


class ConfidenceWeighter:
    """Age- and hop-decayed confidence model."""

    def weight(self, age_ticks: int, hops: int) -> float:
        """Compute confidence weight for one observation."""
        return (1.0 / (1.0 + float(age_ticks))) * (1.0 / (1.0 + float(hops)))

    def fuse(self, observations: list[tuple[float, int, int]]) -> float | None:
        """Compute weighted average of observations or None when empty."""
        if not observations:
            return None
        weighted_values = []
        weights = []
        for hazard, age_ticks, hops in observations:
            wgt = self.weight(age_ticks=age_ticks, hops=hops)
            weighted_values.append(hazard * wgt)
            weights.append(wgt)
        return sum(weighted_values) / sum(weights)
