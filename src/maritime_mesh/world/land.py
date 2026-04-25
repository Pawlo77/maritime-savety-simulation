"""Static land geometry and intersection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist, hypot

Point = tuple[float, float]


@dataclass(frozen=True)
class LandRectangle:
    """Axis-aligned land rectangle in nautical-mile coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, point: Point) -> bool:
        """Return whether a world point lies inside rectangle bounds."""
        x_nm, y_nm = point
        return self.x0 <= x_nm <= self.x1 and self.y0 <= y_nm <= self.y1

    def distance_to_point(self, point: Point) -> float:
        """Return shortest distance from point to rectangle."""
        x_nm, y_nm = point
        dx = max(self.x0 - x_nm, 0.0, x_nm - self.x1)
        dy = max(self.y0 - y_nm, 0.0, y_nm - self.y1)
        return hypot(dx, dy)


@dataclass(frozen=True)
class LandPolygon:
    """Simple polygon land shape using ordered vertices."""

    points: tuple[Point, ...]

    @staticmethod
    def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
        """Return whether point lies on segment."""
        px, py = point
        x1, y1 = start
        x2, y2 = end
        cross = ((px - x1) * (y2 - y1)) - ((py - y1) * (x2 - x1))
        if abs(cross) > 1e-9:
            return False
        return (
            min(x1, x2) - 1e-9 <= px <= max(x1, x2) + 1e-9
            and min(y1, y2) - 1e-9 <= py <= max(y1, y2) + 1e-9
        )

    def contains(self, point: Point) -> bool:
        """Return whether point is within polygon bounds."""
        x_nm, y_nm = point
        if any(
            self._point_on_segment(point, edge_start, edge_end)
            for edge_start, edge_end in self.edges()
        ):
            return True
        inside = False
        for idx in range(len(self.points)):
            x1, y1 = self.points[idx]
            x2, y2 = self.points[(idx + 1) % len(self.points)]
            if (y1 > y_nm) == (y2 > y_nm):
                continue
            denominator = y2 - y1
            if abs(denominator) < 1e-12:
                denominator = 1e-12 if denominator >= 0.0 else -1e-12
            x_cross = ((x2 - x1) * (y_nm - y1) / denominator) + x1
            if x_nm < x_cross:
                inside = not inside
        return inside

    def edges(self) -> tuple[tuple[Point, Point], ...]:
        """Return polygon edges as point pairs."""
        return tuple(
            (self.points[idx], self.points[(idx + 1) % len(self.points)])
            for idx in range(len(self.points))
        )


class WorldLand:
    """Static landmass model used for simulation and rendering."""

    def __init__(
        self,
        rectangles: tuple[LandRectangle, ...],
        polygons: tuple[LandPolygon, ...] = (),
        profile: str = "legacy_rectangles",
    ) -> None:
        """Store immutable land geometry and profile metadata."""
        self.rectangles = rectangles
        self.polygons = polygons
        self.profile = profile

    @classmethod
    def default_for_world_size(
        cls,
        world_size_nm: float,
        profile: str = "natural_coast",
    ) -> WorldLand:
        """Build canonical coastline and island geometry by profile."""
        if profile == "legacy_rectangles":
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
            return cls(
                (coastal_strip, offshore_island, central_island, south_bank),
                profile=profile,
            )
        if profile != "natural_coast":
            raise ValueError(f"Unsupported land profile '{profile}'.")
        return cls(
            rectangles=(
                LandRectangle(
                    x0=world_size_nm * 0.66,
                    y0=0.0,
                    x1=world_size_nm * 0.74,
                    y1=world_size_nm * 0.09,
                ),
            ),
            polygons=(
                LandPolygon(
                    points=(
                        (0.0, 0.0),
                        (world_size_nm * 0.12, 0.0),
                        (world_size_nm * 0.11, world_size_nm * 0.16),
                        (world_size_nm * 0.15, world_size_nm * 0.30),
                        (world_size_nm * 0.10, world_size_nm * 0.42),
                        (world_size_nm * 0.13, world_size_nm * 0.56),
                        (world_size_nm * 0.09, world_size_nm * 0.74),
                        (world_size_nm * 0.12, world_size_nm * 0.88),
                        (world_size_nm * 0.10, world_size_nm),
                        (0.0, world_size_nm),
                    )
                ),
                LandPolygon(
                    points=(
                        (world_size_nm * 0.71, world_size_nm * 0.94),
                        (world_size_nm * 0.79, world_size_nm * 0.92),
                        (world_size_nm * 0.87, world_size_nm * 0.95),
                        (world_size_nm * 0.86, world_size_nm),
                        (world_size_nm * 0.72, world_size_nm),
                    )
                ),
                LandPolygon(
                    points=(
                        (world_size_nm * 0.42, world_size_nm * 0.46),
                        (world_size_nm * 0.48, world_size_nm * 0.44),
                        (world_size_nm * 0.54, world_size_nm * 0.47),
                        (world_size_nm * 0.55, world_size_nm * 0.54),
                        (world_size_nm * 0.50, world_size_nm * 0.58),
                        (world_size_nm * 0.44, world_size_nm * 0.54),
                    )
                ),
                LandPolygon(
                    points=(
                        (world_size_nm * 0.66, world_size_nm * 0.58),
                        (world_size_nm * 0.71, world_size_nm * 0.56),
                        (world_size_nm * 0.76, world_size_nm * 0.60),
                        (world_size_nm * 0.75, world_size_nm * 0.66),
                        (world_size_nm * 0.69, world_size_nm * 0.68),
                        (world_size_nm * 0.65, world_size_nm * 0.64),
                    )
                ),
                LandPolygon(
                    points=(
                        (world_size_nm * 0.24, world_size_nm * 0.80),
                        (world_size_nm * 0.29, world_size_nm * 0.78),
                        (world_size_nm * 0.34, world_size_nm * 0.82),
                        (world_size_nm * 0.33, world_size_nm * 0.88),
                        (world_size_nm * 0.27, world_size_nm * 0.90),
                        (world_size_nm * 0.23, world_size_nm * 0.86),
                    )
                ),
            ),
            profile=profile,
        )

    def is_land(self, point: Point) -> bool:
        """Return whether point is inside any land geometry."""
        return any(shape.contains(point) for shape in (*self.rectangles, *self.polygons))

    @staticmethod
    def _segment_intersects_rectangle(
        start: Point,
        end: Point,
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

    @staticmethod
    def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
        """Return shortest distance from point to segment."""
        px, py = point
        x1, y1 = start
        x2, y2 = end
        seg_len_sq = ((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1))
        if seg_len_sq == 0.0:
            return dist(point, start)
        proj = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / seg_len_sq
        proj = max(0.0, min(1.0, proj))
        nearest = (x1 + ((x2 - x1) * proj), y1 + ((y2 - y1) * proj))
        return dist(point, nearest)

    @staticmethod
    def _orientation(a: Point, b: Point, c: Point) -> float:
        """Return orientation cross product for triplet."""
        return ((b[0] - a[0]) * (c[1] - a[1])) - ((b[1] - a[1]) * (c[0] - a[0]))

    @staticmethod
    def _on_segment(point: Point, start: Point, end: Point) -> bool:
        """Return whether point lies on segment bounds."""
        return (
            min(start[0], end[0]) - 1e-9 <= point[0] <= max(start[0], end[0]) + 1e-9
            and min(start[1], end[1]) - 1e-9 <= point[1] <= max(start[1], end[1]) + 1e-9
        )

    @classmethod
    def _segments_intersect(
        cls,
        a_start: Point,
        a_end: Point,
        b_start: Point,
        b_end: Point,
    ) -> bool:
        """Return whether two segments intersect."""
        o1 = cls._orientation(a_start, a_end, b_start)
        o2 = cls._orientation(a_start, a_end, b_end)
        o3 = cls._orientation(b_start, b_end, a_start)
        o4 = cls._orientation(b_start, b_end, a_end)
        if ((o1 > 0.0) != (o2 > 0.0)) and ((o3 > 0.0) != (o4 > 0.0)):
            return True
        if abs(o1) < 1e-9 and cls._on_segment(b_start, a_start, a_end):
            return True
        if abs(o2) < 1e-9 and cls._on_segment(b_end, a_start, a_end):
            return True
        if abs(o3) < 1e-9 and cls._on_segment(a_start, b_start, b_end):
            return True
        return abs(o4) < 1e-9 and cls._on_segment(a_end, b_start, b_end)

    def distance_to_land(self, point: Point) -> float:
        """Return distance from point to nearest land boundary."""
        if self.is_land(point):
            return 0.0
        min_distance = float("inf")
        for rectangle in self.rectangles:
            min_distance = min(min_distance, rectangle.distance_to_point(point))
        for polygon in self.polygons:
            for edge_start, edge_end in polygon.edges():
                min_distance = min(
                    min_distance,
                    self._point_segment_distance(point, edge_start, edge_end),
                )
        return min_distance if min_distance != float("inf") else 0.0

    def _segment_intersects_polygon(
        self,
        start: Point,
        end: Point,
        polygon: LandPolygon,
        samples: int,
    ) -> bool:
        """Return whether a line segment enters polygon area."""
        if polygon.contains(start) or polygon.contains(end):
            return True
        for index in range(1, samples):
            ratio = index / samples
            sample = (
                start[0] + ((end[0] - start[0]) * ratio),
                start[1] + ((end[1] - start[1]) * ratio),
            )
            if polygon.contains(sample):
                return True
        return False

    def segment_intersects_land(
        self,
        start: Point,
        end: Point,
        samples: int = 24,
        clearance_nm: float = 0.0,
    ) -> bool:
        """Return whether line segment intersects land or buffered clearance."""
        if clearance_nm > 0.0:
            if (
                self.distance_to_land(start) <= clearance_nm
                or self.distance_to_land(end) <= clearance_nm
            ):
                return True
        elif self.is_land(start) or self.is_land(end):
            return True
        for rectangle in self.rectangles:
            if self._segment_intersects_rectangle(start, end, rectangle):
                return True
        for polygon in self.polygons:
            if self._segment_intersects_polygon(start, end, polygon, samples=max(samples, 40)):
                return True
        for index in range(1, samples):
            ratio = index / samples
            sample = (
                start[0] + ((end[0] - start[0]) * ratio),
                start[1] + ((end[1] - start[1]) * ratio),
            )
            if clearance_nm > 0.0:
                if self.distance_to_land(sample) <= clearance_nm:
                    return True
            elif self.is_land(sample):
                return True
        return False
