"""Static land geometry and intersection helpers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LandRectangle:
    """Axis-aligned land rectangle in nautical-mile coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, point: tuple[float, float]) -> bool:
        """Return whether a world point lies inside rectangle bounds."""
        x_nm, y_nm = point
        return self.x0 <= x_nm <= self.x1 and self.y0 <= y_nm <= self.y1


class WorldLand:
    """Simple static landmass model used for simulation and rendering."""

    def __init__(self, rectangles: tuple[LandRectangle, ...]) -> None:
        """Store immutable set of axis-aligned land rectangles."""
        self.rectangles = rectangles

    @classmethod
    def default_for_world_size(cls, world_size_nm: float) -> "WorldLand":
        """Build canonical coastline and headland geometry."""
        coastal_strip = LandRectangle(
            x0=0.0,
            y0=0.0,
            x1=world_size_nm * 0.12,
            y1=world_size_nm,
        )
        offshore_island = LandRectangle(
            x0=world_size_nm * 0.72,
            y0=world_size_nm * 0.94,
            x1=world_size_nm * 0.86,
            y1=world_size_nm,
        )
        central_island = LandRectangle(
            x0=world_size_nm * 0.44,
            y0=world_size_nm * 0.46,
            x1=world_size_nm * 0.56,
            y1=world_size_nm * 0.58,
        )
        south_bank = LandRectangle(
            x0=world_size_nm * 0.62,
            y0=0.0,
            x1=world_size_nm * 0.74,
            y1=world_size_nm * 0.10,
        )
        return cls((coastal_strip, offshore_island, central_island, south_bank))

    def is_land(self, point: tuple[float, float]) -> bool:
        """Return whether point is inside any land rectangle."""
        return any(rectangle.contains(point) for rectangle in self.rectangles)

    @staticmethod
    def _segment_intersects_rectangle(
        start: tuple[float, float],
        end: tuple[float, float],
        rectangle: LandRectangle,
    ) -> bool:
        """Return whether a line segment intersects axis-aligned rectangle."""
        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0
        p = (-dx, dx, -dy, dy)
        q = (
            x0 - rectangle.x0,
            rectangle.x1 - x0,
            y0 - rectangle.y0,
            rectangle.y1 - y0,
        )
        u1 = 0.0
        u2 = 1.0
        for p_i, q_i in zip(p, q, strict=False):
            if p_i == 0.0:
                if q_i < 0.0:
                    return False
                continue
            t = q_i / p_i
            if p_i < 0.0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)
            if u1 > u2:
                return False
        return True

    def segment_intersects_land(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        samples: int = 24,
    ) -> bool:
        """Return whether line segment intersects any land area."""
        if self.is_land(start) or self.is_land(end):
            return True
        for rectangle in self.rectangles:
            if self._segment_intersects_rectangle(start, end, rectangle):
                return True
        for index in range(1, samples):
            ratio = index / samples
            sample = (
                start[0] + ((end[0] - start[0]) * ratio),
                start[1] + ((end[1] - start[1]) * ratio),
            )
            if self.is_land(sample):
                return True
        return False
