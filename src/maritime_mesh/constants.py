"""Global constants for the maritime weather mesh simulation."""

WORLD_SIZE_NM: float = 100.0
"""Side length of the simulation world in nautical miles."""
WEATHER_GRID_CELLS: int = 50
"""Number of cells along each axis of the weather grid."""
MACRO_TICK_HOURS: float = 0.25
"""Duration of one macro scheduler tick in hours."""
MICRO_TICK_HOURS: float = 0.00833
"""Duration of one micro scheduler tick in hours."""
CLOSE_ENCOUNTER_THRESHOLD_NM: float = 5.0
"""Distance threshold for micro-tick activation in nautical miles."""
SHORE_STATION_POSITION: tuple[float, float] = (0.0, 50.0)
"""Fixed position of the coastal station in nautical miles."""
SHORE_BROADCAST_RADIUS_NM: float = 50.0
"""Maximum shore broadcast range in nautical miles."""
SHORE_NOISE_MEAN: float = 0.05
"""Mean Gaussian bias of shore weather broadcast."""
SHORE_NOISE_STD_NORMAL: float = 0.18
"""Shore forecast noise standard deviation under normal conditions."""
SHORE_NOISE_STD_STRESS: float = 0.36
"""Shore forecast noise standard deviation under stressed conditions."""
RADIO_RANGE_FALLOFF: float = 10.0
"""Sigmoid distance scale factor for radio reception probability."""
RADIO_WEATHER_INTERFERENCE: float = 0.8
"""Weight of local hazard on reception interference."""
VESSEL_RADIO_RANGE_NM: float = 15.0
"""Ship-to-ship radio range in nautical miles."""
LAND_PROFILE: str = "natural_coast"
"""Land generation profile used for world geometry."""
LAND_CLEARANCE_NM: float = 0.0
"""Minimum safety corridor around land where routes are disallowed."""
ROUTE_START_NEAR_SHORE_NM: float = 45.0
"""Maximum distance from route start to nearest shore station."""
ROUTE_END_NEAR_SHORE_NM: float = 60.0
"""Maximum distance from route end to nearest shore station."""
ROUTE_END_OFFMAP_MARGIN_NM: float = 12.0
"""Distance from map boundary qualifying as route exiting map."""
MAX_HOP_COUNT: int = 2
"""Maximum relay hops allowed for a packet."""
WEATHER_WEIGHT_SEA_STATE: float = 0.4
"""Weight of sea-state channel in composite hazard."""
WEATHER_WEIGHT_VISIBILITY: float = 0.3
"""Weight of inverted visibility channel in composite hazard."""
WEATHER_WEIGHT_WIND: float = 0.3
"""Weight of wind channel in composite hazard."""
FATIGUE_WEIGHT: float = 0.12
"""Logit coefficient for hours-awake fatigue term."""
CIRCADIAN_WEIGHT: float = 1.5
"""Logit coefficient for circadian vigilance term."""
CIRCADIAN_NADIR_HOUR_UTC: float = 3.0
"""UTC hour of minimum vigilance."""
ERROR_BIAS: float = 4.0
"""Bias term subtracted from crew-error logit."""
ARCHETYPE_MOD_VETERAN: float = -1.5
"""Crew archetype logit offset for veteran crews."""
ARCHETYPE_MOD_STANDARD: float = 0.0
"""Crew archetype logit offset for standard crews."""
ARCHETYPE_MOD_GREEN: float = 2.0
"""Crew archetype logit offset for inexperienced crews."""
PREP_FORECAST_ERROR_PENALTY: float = 1.5
"""Preparedness penalty scaling for forecast error."""
PREP_ARCHETYPE_PENALTY: float = 0.2
"""Preparedness penalty scaling for archetype modifier."""
SHORE_TRUST_DECAY_K: float = 5.0
"""Exponential decay constant for shore-trust adaptation."""
EVAC_HAZARD_WEIGHT: float = 4.0
"""Evacuation logit weight for blended hazard."""
EVAC_ERROR_PENALTY: float = 2.5
"""Evacuation logit penalty for forecast error."""
EVAC_PREP_WEIGHT: float = 1.8
"""Evacuation logit weight for preparedness."""
EVAC_BASELINE_BIAS: float = -2.8
"""Evacuation baseline bias to prevent immediate mass evacuation."""
RAFT_BASE_SUCCESS: float = 0.90
"""Baseline raft deployment success probability."""
RAFT_WEATHER_PENALTY: float = 0.60
"""Raft deployment penalty scaling for hazard."""
RAFT_PREP_BONUS: float = 0.25
"""Raft deployment bonus scaling for preparedness."""
HELICOPTER_SPEED_KN: float = 80.0
"""Rescue helicopter speed in knots."""
PATROL_VESSEL_SPEED_KN: float = 25.0
"""Rescue patrol vessel speed in knots."""
RESCUE_MOBILISATION_TICKS: int = 2
"""Delay in ticks before rescue asset starts moving."""
COLLISION_RADIUS_NM: float = 0.15
"""Collision detection radius around each vessel in nautical miles."""
