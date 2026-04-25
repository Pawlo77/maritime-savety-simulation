"""Dashboard constants and display metadata."""

KPI_COLUMNS = [
    "fatal_per_1k_hrs",
    "collision_per_1k_hrs",
    "survival_ratio",
    "avg_tta_hours",
    "evac_activation_rate",
    "mean_p_prep",
]

SCENARIO_CHOICES = [
    "scenario_1_calm_passage",
    "scenario_2_storm_corridor",
    "scenario_3_blind_shore",
    "scenario_4_deep_water_rescue",
]

KPI_DESCRIPTIONS = {
    "fatal_per_1k_hrs": "Fatal events per 1,000 ship-hours (lower is better).",
    "collision_per_1k_hrs": "Collision events per 1,000 ship-hours (lower is better).",
    "survival_ratio": "Survivors divided by total exposed crew (higher is better).",
    "avg_tta_hours": "Average rescue time-to-arrival in hours (lower is better).",
    "evac_activation_rate": "Fraction of vessels that entered evacuation mode.",
    "mean_p_prep": "Average preparedness score across all vessels and ticks.",
}
