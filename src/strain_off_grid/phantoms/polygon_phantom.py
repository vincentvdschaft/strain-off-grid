from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from scipy.interpolate import CubicSpline

from .phantom import DynamicPhantom

_PLANE_AXES = [0, 2]  # the polygon's 2D coordinates map to the x and z axes
_ELEVATION_AXIS = 1  # y is the thin out-of-plane (elevation) direction


@dataclass
class PolygonPhantom(DynamicPhantom):
    """A 2D polygon (in the x–z plane) whose vertices move over time.

    The polygon is defined by keyframe vertices ``points`` of shape
    ``(n_times, n_points, 2)``.  At an arbitrary time t the vertices are
    obtained by cubic-spline interpolation of the keyframes over ``times``.
    Interior points are carried along with the moving boundary using mean value
    coordinates, so a point that starts inside the polygon keeps the same
    relative location as the outer vertices deform.

    The two polygon coordinates map to the x and z axes (the imaging plane); the
    y axis is a thin out-of-plane extent of size ``thickness``.
    """

    points: np.ndarray = field(default_factory=lambda: np.zeros((1, 3, 2)))
    keyframe_times: Optional[np.ndarray] = None
    thickness: float = 0e-3

    def set_keyframe_times_by_dt(self, dt: float) -> PolygonPhantom:
        """Sets the keyframe times to a uniform interval of dt."""
        self.keyframe_times = np.arange(self.points.shape[0]) * dt
        print(self.keyframe_times)
        return self

    @classmethod
    def from_csv(
        cls,
        path: Union[str, Path],
        thickness: float = 2e-3,
        scale: float = 1.0,
        center_x: bool = True,
    ) -> "PolygonPhantom":
        """Builds a phantom from a CSV with one row per polygon vertex.

        The CSV must have the columns ``shape_index,shape_type,t,z,y,x``. Rows
        sharing the same ``t`` form one keyframe polygon and ``t`` becomes the
        keyframe time. Every keyframe must have the same number of vertices. The
        x/z coordinates are multiplied by ``scale``.
        """
        rows = _read_csv_rows(path)
        times, points = _group_keyframes(rows)
        points = points * scale
        if center_x:
            points[..., 0] -= np.mean(points[..., 0])
        return cls(points=points, keyframe_times=times, thickness=thickness)

    def _keyframe_times(self) -> np.ndarray:
        """Returns the keyframe times, defaulting to a unit interval."""
        if self.keyframe_times is not None:
            return np.asarray(self.keyframe_times)
        return np.linspace(0.0, 1.0, self.points.shape[0])

    def _polygon_at_time(self, t: float) -> np.ndarray:
        """Cubic-spline interpolated polygon vertices at time t."""
        if self.points.shape[0] == 1:
            return self.points[0]
        spline = CubicSpline(self._keyframe_times(), self.points, axis=0)
        return spline(t)

    def _morph(
        self, positions_local: np.ndarray, source: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """Moves points from the source polygon to the target via mean value coords."""
        weights = _mean_value_weights(positions_local[:, _PLANE_AXES], source)
        return self._with_plane_coords(positions_local, weights @ target)

    def _with_plane_coords(
        self, positions_local: np.ndarray, plane_xy: np.ndarray
    ) -> np.ndarray:
        """Returns a copy of positions with the x/z coordinates replaced."""
        result = positions_local.copy()
        result[:, _PLANE_AXES] = plane_xy
        return result

    def _translate_to_time(self, positions_local: np.ndarray, t: float) -> np.ndarray:
        return self._morph(
            positions_local, self._polygon_at_time(0.0), self._polygon_at_time(t)
        )

    def _inverse_translate_to_time(
        self, positions_local: np.ndarray, t: float
    ) -> np.ndarray:
        return self._morph(
            positions_local, self._polygon_at_time(t), self._polygon_at_time(0.0)
        )

    def _embed_in_3d(self, plane_points: np.ndarray) -> np.ndarray:
        """Combines 2D plane points with a random out-of-plane elevation."""
        positions = np.zeros((plane_points.shape[0], 3))
        positions[:, _PLANE_AXES] = plane_points
        positions[:, _ELEVATION_AXIS] = (
            np.random.uniform(
                -self.thickness / 2, self.thickness / 2, plane_points.shape[0]
            )
            * 0.0
        )
        return positions

    def _sample_points(self, n_points: int) -> np.ndarray:
        """Samples n_points uniformly inside the polygon at time 0."""
        plane_points = _sample_inside_polygon(self._polygon_at_time(0.0), n_points)
        return self._embed_in_3d(plane_points)

    def _points_in_phantom(self, positions_local_t0: np.ndarray) -> np.ndarray:
        """Returns a boolean array indicating which positions are inside the phantom at time t."""
        print(positions_local_t0.shape)
        plane_points = positions_local_t0[:, _PLANE_AXES]
        return _points_in_polygon(plane_points, self._polygon_at_time(0.0))


def _read_csv_rows(path: Union[str, Path]) -> list:
    """Returns the CSV rows as dictionaries keyed by column name."""
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def _group_keyframes(rows: Sequence[dict]) -> tuple:
    """Groups vertex rows by time into (times, points) arrays, sorted by time."""
    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(row["t"], []).append(row)
    keyframes_by_time = sorted(groups.items(), key=lambda item: float(item[0]))
    keyframes = [_rows_to_vertices(rows) for _, rows in keyframes_by_time]
    _check_equal_vertex_counts(keyframes)
    times = np.array([float(t) for t, _ in keyframes_by_time])
    return times, np.stack(keyframes, axis=0)


def _rows_to_vertices(rows: Sequence[dict]) -> np.ndarray:
    """Returns the (x, z) vertices for one keyframe's rows."""
    return np.array([[float(row["x"]), float(row["z"])] for row in rows])


def _check_equal_vertex_counts(keyframes: Sequence[np.ndarray]) -> None:
    """Raises if the keyframe polygons do not all share one vertex count."""
    counts = {len(keyframe) for keyframe in keyframes}
    if len(counts) != 1:
        raise ValueError(f"Keyframes have differing vertex counts: {sorted(counts)}")


def _tan_half_angles(edge_vectors: np.ndarray) -> np.ndarray:
    """tan(angle/2) at the query point between each vertex and the next."""
    next_vectors = np.roll(edge_vectors, -1, axis=1)
    cross = np.cross(edge_vectors, next_vectors)
    dot = np.sum(edge_vectors * next_vectors, axis=2)
    return np.tan(np.arctan2(cross, dot) / 2.0)


def _mean_value_weights(query_xy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Mean value coordinates of query points w.r.t. a polygon (Floater 2003)."""
    edge_vectors = polygon[None, :, :] - query_xy[:, None, :]
    radii = np.maximum(np.linalg.norm(edge_vectors, axis=2), 1e-12)
    tan_half = _tan_half_angles(edge_vectors)
    weights = (np.roll(tan_half, 1, axis=1) + tan_half) / radii
    return weights / weights.sum(axis=1, keepdims=True)


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized even-odd ray-casting point-in-polygon test."""
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    for (x0, y0), (x1, y1) in zip(polygon, np.roll(polygon, -1, axis=0)):
        crosses = (y0 > y) != (y1 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_intersect = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (x < x_intersect)
    return inside


def _distances_to_segments(
    points: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    """Distance from each point to each segment, shape (n_points, n_segments)."""
    edges = ends - starts
    offsets = points[:, None, :] - starts[None, :, :]
    edge_lengths_squared = np.maximum(np.sum(edges**2, axis=1), 1e-12)
    projections = np.sum(offsets * edges[None, :, :], axis=2) / edge_lengths_squared
    clamped = np.clip(projections, 0.0, 1.0)
    closest = starts[None, :, :] + clamped[:, :, None] * edges[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=2)


def distances_to_edge(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Distance from each (n_points, 2) coordinate to the polygon boundary."""
    distances = _distances_to_segments(points, polygon, np.roll(polygon, -1, axis=0))
    return distances.min(axis=1)


def _sample_inside_polygon(polygon: np.ndarray, n_points: int) -> np.ndarray:
    """Rejection-samples n_points uniformly inside a simple polygon."""
    lower, upper = polygon.min(axis=0), polygon.max(axis=0)
    collected = []
    while sum(len(chunk) for chunk in collected) < n_points:
        candidates = np.random.uniform(lower, upper, size=(n_points * 2, 2))
        collected.append(candidates[_points_in_polygon(candidates, polygon)])
    return np.concatenate(collected, axis=0)[:n_points]
