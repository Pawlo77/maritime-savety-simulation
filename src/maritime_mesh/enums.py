"""Enumerations used by maritime weather mesh simulation."""

from enum import Enum


class VesselState(Enum):
    """Lifecycle states of a vessel agent."""

    ACTIVE = "active"
    EVAC = "evac"
    SUNK = "sunk"
    RESCUED = "rescued"


class CrewArchetype(Enum):
    """Crew-experience archetypes controlling risk behaviour."""

    VETERAN = "veteran"
    STANDARD = "standard"
    GREEN = "green"


class RescueAssetType(Enum):
    """Rescue asset kinds dispatched from the coastal station."""

    HELICOPTER = "helicopter"
    PATROL_VESSEL = "patrol_vessel"


class MethodCondition(Enum):
    """Experiment method conditions for ablation matrix."""

    BASELINE_A = "baseline_a"
    BASELINE_B = "baseline_b"
    PROPOSED = "proposed"
