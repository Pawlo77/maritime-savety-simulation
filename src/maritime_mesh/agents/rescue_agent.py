"""Rescue asset agent implementation."""

from math import atan2, cos, dist, sin

import numpy as np

from maritime_mesh.agents.base_agent import AbstractMesaAgent
from maritime_mesh.constants import (
    HELICOPTER_SPEED_KN,
    MACRO_TICK_HOURS,
    PATROL_VESSEL_SPEED_KN,
    RESCUE_MOBILISATION_TICKS,
)
from maritime_mesh.enums import RescueAssetType, VesselState
from maritime_mesh.mesa_compat import Model

RESCUE_CAPTURE_RADIUS_NM = 0.4


class RescueAgent(AbstractMesaAgent):
    """Rescue asset travelling to a distress location."""

    def __init__(
        self,
        model: Model,
        unique_id: int,
        rng: np.random.Generator,
        asset_type: RescueAssetType,
        target_position: tuple[float, float],
        start_position: tuple[float, float],
    ) -> None:
        """Initialize rescue asset configuration."""
        super().__init__(model=model, rng=rng)
        self.unique_id = unique_id
        self.asset_type = asset_type
        self.speed_kn = (
            HELICOPTER_SPEED_KN
            if asset_type == RescueAssetType.HELICOPTER
            else PATROL_VESSEL_SPEED_KN
        )
        self.position = start_position
        self.target_position = target_position
        self.mobilisation_ticks_remaining = RESCUE_MOBILISATION_TICKS
        self.is_idle = False
        self.target_vessel_id: int | None = None

    def step(self) -> None:
        """Advance countdown or move toward target and rescue survivors."""
        if self.is_idle:
            return
        if self.mobilisation_ticks_remaining > 0:
            self.mobilisation_ticks_remaining -= 1
            return

        distance_to_target = dist(self.position, self.target_position)
        if distance_to_target <= RESCUE_CAPTURE_RADIUS_NM:
            self.position = self.target_position
            rescued_any = False
            for vessel in self.model.vessels:
                if (
                    vessel.state == VesselState.EVAC
                    and dist(vessel.position, self.target_position) <= RESCUE_CAPTURE_RADIUS_NM
                ):
                    vessel.state = VesselState.RESCUED
                    rescued_any = True
            # Mark completed rescue sortie (even if target vessel already terminal).
            self.is_idle = bool(rescued_any or self.target_vessel_id is not None)
            return

        local_hazard = self.model.weather_field.hazard_at(*self.position)
        hazard_speed_penalty = max(0.4, 1.0 - (0.35 * local_hazard))
        step_nm = min(distance_to_target, self.speed_kn * hazard_speed_penalty * MACRO_TICK_HOURS)
        angle = atan2(
            self.target_position[1] - self.position[1],
            self.target_position[0] - self.position[0],
        )
        self.position = (
            self.position[0] + (step_nm * cos(angle)),
            self.position[1] + (step_nm * sin(angle)),
        )
