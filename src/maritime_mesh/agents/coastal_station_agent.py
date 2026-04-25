"""Coastal station agent implementation."""

from collections import deque

import numpy as np

from maritime_mesh.agents.base_agent import AbstractMesaAgent
from maritime_mesh.communication.packet import SosPacket
from maritime_mesh.communication.shore_radio import ShoreRadioModel
from maritime_mesh.constants import SHORE_STATION_POSITION
from maritime_mesh.mesa_compat import Model
from maritime_mesh.weather.weather_field import WeatherField


class CoastalStationAgent(AbstractMesaAgent):
    """Stationary shore station broadcasting weather and handling SOS queue."""

    def __init__(
        self,
        model: Model,
        unique_id: int,
        rng: np.random.Generator,
        shore_radio: ShoreRadioModel,
        weather_field: WeatherField,
    ) -> None:
        """Initialize station dependencies and queue state."""
        super().__init__(model=model, rng=rng)
        self.unique_id = unique_id
        self.position = SHORE_STATION_POSITION
        self.sos_queue: deque[SosPacket] = deque()
        self.shore_radio = shore_radio
        self.weather_field = weather_field
        self.last_broadcast: float | None = None

    def step(self) -> None:
        """Broadcast weather and notify model for pending SOS dispatch."""
        true_hazard = self.weather_field.hazard_at(*self.position)
        self.last_broadcast = self.shore_radio.broadcast(true_hazard=true_hazard)
        while self.sos_queue:
            packet = self.sos_queue.popleft()
            self.model.dispatch_rescue(packet)

    def receive_sos(self, packet: SosPacket) -> None:
        """Queue distress packet for dispatch handling."""
        self.sos_queue.append(packet)
