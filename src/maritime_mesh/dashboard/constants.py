"""Dashboard constants and display metadata."""

KPI_COLUMNS = [
    "fatal_per_1k_hrs",
    "collision_per_1k_hrs",
    "survival_ratio",
    "avg_tta_hours",
    "evac_activation_rate",
    "mean_p_prep",
]

KPI_HIGHER_IS_BETTER = {
    "fatal_per_1k_hrs": False,
    "collision_per_1k_hrs": False,
    "survival_ratio": True,
    "avg_tta_hours": False,
    "evac_activation_rate": True,
    "mean_p_prep": True,
}

SCENARIO_CHOICES = [
    "scenario_1_calm_passage",
    "scenario_2_storm_corridor",
    "scenario_3_blind_shore",
    "scenario_4_deep_water_rescue",
]

SCENARIO_LABELS = {
    "scenario_1_calm_passage": "Scenario 1: Calm Passage",
    "scenario_2_storm_corridor": "Scenario 2: Storm Corridor",
    "scenario_3_blind_shore": "Scenario 3: Blind Shore",
    "scenario_4_deep_water_rescue": "Scenario 4: Deep Water Rescue",
}

METHOD_LABELS = {
    "baseline_a": "Baseline A (No mesh relay)",
    "baseline_b": "Baseline B (Weather-only routing)",
    "proposed": "Proposed (Adaptive maritime mesh)",
}

KPI_DESCRIPTIONS = {
    "fatal_per_1k_hrs": "Fatal events per 1,000 ship-hours (lower is better).",
    "collision_per_1k_hrs": "Collision events per 1,000 ship-hours (lower is better).",
    "survival_ratio": "Survivors divided by total exposed crew (higher is better).",
    "avg_tta_hours": "Average rescue time-to-arrival in hours (lower is better).",
    "evac_activation_rate": "Fraction of vessels that entered evacuation mode.",
    "mean_p_prep": "Average preparedness score across all vessels and ticks.",
}


def display_scenario_name(scenario_id: str) -> str:
    """Return human-readable scenario label."""
    return SCENARIO_LABELS.get(scenario_id, scenario_id.replace("_", " ").title())


def display_method_name(method_id: str) -> str:
    """Return human-readable method label."""
    return METHOD_LABELS.get(method_id, method_id.replace("_", " ").title())
