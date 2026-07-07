from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline

from .phantom import DynamicPhantom

_PLANE_AXES = [0, 2]  # the chain's 2D coordinates map to the x and z axes
_ELEVATION_AXIS = 1  # y is the thin out-of-plane (elevation) direction

_INNER, _OUTER = 0, 1


@dataclass
class ChainedBoxesPhantom(DynamicPhantom):
    """A chain of quadrilateral boxes (in the x–z plane) that deform over time.

    The chain is described by ``box_endpoints`` of shape
    ``(n_times, n_segments, 2, 2)``: for each keyframe time and segment it holds
    an inner and an outer endpoint (``[..., _INNER, :]`` and
    ``[..., _OUTER, :]``). Consecutive segments span one box, so ``n_segments``
    endpoints define ``n_segments - 1`` boxes that share their width edges.

    An interior point is bound to the single box it falls in and is carried along
    by only the four vertices of that box, using bilinear (quad) coordinates. The
    principal axis at a point is the tangent of the centre line running lengthwise
    through the chain.
    """

    box_endpoints: np.ndarray = field(default_factory=lambda: np.zeros((1, 2, 2, 2)))
    keyframe_times: Optional[np.ndarray] = None
    thickness: float = 0e-3

    def _keyframe_times(self) -> np.ndarray:
        """Returns the keyframe times, defaulting to a unit interval."""
        if self.keyframe_times is not None:
            return np.asarray(self.keyframe_times)
        return np.linspace(0.0, 1.0, self.box_endpoints.shape[0])

    def _boxes_at_time(self, t: float) -> np.ndarray:
        """Cubic-spline interpolated box endpoints at time t."""
        if self.box_endpoints.shape[0] == 1:
            return self.box_endpoints[0]
        spline = CubicSpline(self._keyframe_times(), self.box_endpoints, axis=0)
        return spline(t)

    def _morph(
        self, positions_local: np.ndarray, source: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """Moves points from the source boxes to the target boxes per box."""
        plane = positions_local[..., _PLANE_AXES]
        box_index, u, v = _locate_in_boxes(plane.reshape(-1, 2), source)
        moved = _evaluate_boxes(target, box_index, u, v).reshape(plane.shape)
        return self._with_plane_coords(positions_local, moved)

    def _with_plane_coords(
        self, positions_local: np.ndarray, plane_xz: np.ndarray
    ) -> np.ndarray:
        """Returns a copy of positions with the x/z coordinates replaced."""
        result = positions_local.copy()
        result[..., _PLANE_AXES] = plane_xz
        return result

    def _translate_to_time(self, positions_local: np.ndarray, t: float) -> np.ndarray:
        return self._morph(
            positions_local, self._boxes_at_time(0.0), self._boxes_at_time(t)
        )

    def _inverse_translate_to_time(
        self, positions_local: np.ndarray, t: float
    ) -> np.ndarray:
        return self._morph(
            positions_local, self._boxes_at_time(t), self._boxes_at_time(0.0)
        )

    def _embed_in_3d(self, plane_points: np.ndarray) -> np.ndarray:
        """Combines 2D plane points with a random out-of-plane elevation."""
        positions = np.zeros((plane_points.shape[0], 3))
        positions[:, _PLANE_AXES] = plane_points
        positions[:, _ELEVATION_AXIS] = np.random.uniform(
            -self.thickness / 2, self.thickness / 2, plane_points.shape[0]
        )
        return positions

    def _sample_points(self, n_points: int) -> np.ndarray:
        """Samples n_points spread across the boxes at time 0."""
        plane_points = _sample_inside_boxes(self._boxes_at_time(0.0), n_points)
        return self._embed_in_3d(plane_points)

    def points_in_phantom(self, positions: np.ndarray, t: float) -> np.ndarray:
        """Returns which positions lie inside a box of the chain at time t.

        Tests membership directly against the boxes at time t rather than morphing
        back to time 0, so outside points are never snapped into the chain by the
        (nonlinear) bilinear extrapolation."""
        plane_points = self._to_local(positions)[..., _PLANE_AXES].reshape(-1, 2)
        mask = _points_in_any_box(plane_points, self._boxes_at_time(t))
        return mask.reshape(*positions.shape[:-1])

    def _points_in_phantom(self, positions_local_t0: np.ndarray) -> np.ndarray:
        """Returns which local positions lie inside any box at time 0."""
        plane_points = positions_local_t0[..., _PLANE_AXES].reshape(-1, 2)
        mask = _points_in_any_box(plane_points, self._boxes_at_time(0.0))
        return mask.reshape(*positions_local_t0.shape[:-1])

    def _principal_axis(self, positions_local_t0: np.ndarray) -> np.ndarray:
        return np.array([0.0, 1.0])
        # """Returns the chain centre-line tangent (x, z) at each local position."""
        # boxes = self._boxes_at_time(0.0)
        # plane = positions_local_t0[..., _PLANE_AXES]
        # box_index, _, _ = _locate_in_boxes(plane.reshape(-1, 2), boxes)
        # return _centerline_directions(boxes)[box_index].reshape(plane.shape)


def _box_corners(boxes: np.ndarray, box: int) -> tuple:
    """Returns the four corners (a, b, c, d) of a box, ordered around the quad."""
    a = boxes[box, _INNER]
    b = boxes[box + 1, _INNER]
    c = boxes[box + 1, _OUTER]
    d = boxes[box, _OUTER]
    return a, b, c, d


def _cross_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Scalar cross product of 2D vectors, broadcast over the leading axes."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _away_from_zero(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Nudges values away from zero to keep divisions finite."""
    return np.where(np.abs(values) < eps, eps, values)


def _unit_square_penalty(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Squared distance from (u, v) to the unit square; zero when inside."""
    return (u - np.clip(u, 0.0, 1.0)) ** 2 + (v - np.clip(v, 0.0, 1.0)) ** 2


def _candidate_vs(
    k0: np.ndarray, k1: np.ndarray, k2: np.ndarray, eps: float = 1e-9
) -> tuple:
    """Returns the two v-roots of the bilinear system (equal in the linear case)."""
    is_linear = np.abs(k2) < eps
    v_linear = -k0 / _away_from_zero(k1)
    discriminant = np.maximum(k1 * k1 - 4.0 * k0 * k2, 0.0)
    root = np.sqrt(discriminant)
    inverse = 0.5 / _away_from_zero(k2)
    v_minus = np.where(is_linear, v_linear, (-k1 - root) * inverse)
    v_plus = np.where(is_linear, v_linear, (-k1 + root) * inverse)
    return v_minus, v_plus


def _bilinear_u(
    h: np.ndarray, e: np.ndarray, f: np.ndarray, g: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Recovers u from a known v using the more stable coordinate axis."""
    numerator = h - f * v[:, None]
    denominator = e + g * v[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        u_x = numerator[:, 0] / _away_from_zero(denominator[:, 0])
        u_y = numerator[:, 1] / _away_from_zero(denominator[:, 1])
    use_x = np.abs(denominator[:, 0]) >= np.abs(denominator[:, 1])
    return np.where(use_x, u_x, u_y)


def _best_uv(
    h: np.ndarray,
    e: np.ndarray,
    f: np.ndarray,
    g: np.ndarray,
    v_minus: np.ndarray,
    v_plus: np.ndarray,
) -> tuple:
    """Picks the (u, v) root that lands closest to the unit square."""
    u_minus = _bilinear_u(h, e, f, g, v_minus)
    u_plus = _bilinear_u(h, e, f, g, v_plus)
    pick_minus = _unit_square_penalty(u_minus, v_minus) <= _unit_square_penalty(
        u_plus, v_plus
    )
    u = np.where(pick_minus, u_minus, u_plus)
    v = np.where(pick_minus, v_minus, v_plus)
    return u, v


def _inverse_bilinear(
    points: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> tuple:
    """Solves for the bilinear (u, v) coordinates of points in the quad a,b,c,d."""
    e, f, g = b - a, d - a, a - b + c - d
    h = points - a
    k2 = _cross_2d(g, f)
    k1 = _cross_2d(e, f) + _cross_2d(h, g)
    k0 = _cross_2d(h, e)
    v_minus, v_plus = _candidate_vs(k0, k1, k2)
    return _best_uv(h, e, f, g, v_minus, v_plus)


def _locate_in_boxes(points: np.ndarray, boxes: np.ndarray) -> tuple:
    """Assigns each point to its best-fitting box with its bilinear (u, v).

    The (u, v) are left un-clamped so that a point outside every box maps back to
    an outside position (identity when source and target boxes coincide), which
    keeps ``points_in_phantom`` from snapping outside points into the chain."""
    best_penalty = np.full(len(points), np.inf)
    best_index = np.zeros(len(points), dtype=int)
    best_u = np.zeros(len(points))
    best_v = np.zeros(len(points))
    for box in range(boxes.shape[0] - 1):
        u, v = _inverse_bilinear(points, *_box_corners(boxes, box))
        penalty = _unit_square_penalty(u, v)
        better = penalty < best_penalty
        best_penalty = np.where(better, penalty, best_penalty)
        best_index = np.where(better, box, best_index)
        best_u = np.where(better, u, best_u)
        best_v = np.where(better, v, best_v)
    return best_index, best_u, best_v


def _evaluate_boxes(
    boxes: np.ndarray, box_index: np.ndarray, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Bilinearly maps (u, v) inside each point's box to a plane position."""
    inner_start = boxes[box_index, _INNER]
    inner_end = boxes[box_index + 1, _INNER]
    outer_end = boxes[box_index + 1, _OUTER]
    outer_start = boxes[box_index, _OUTER]
    u, v = u[:, None], v[:, None]
    return (
        (1 - u) * (1 - v) * inner_start
        + u * (1 - v) * inner_end
        + u * v * outer_end
        + (1 - u) * v * outer_start
    )


def _points_in_any_box(
    points: np.ndarray, boxes: np.ndarray, tolerance: float = 1e-9
) -> np.ndarray:
    """Returns a boolean array marking points that fall inside some box."""
    inside = np.zeros(len(points), dtype=bool)
    for box in range(boxes.shape[0] - 1):
        u, v = _inverse_bilinear(points, *_box_corners(boxes, box))
        inside |= _unit_square_penalty(u, v) <= tolerance
    return inside


def _quad_area(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Shoelace area of the quadrilateral with corners a, b, c, d."""
    corners = np.stack([a, b, c, d])
    x, y = corners[:, 0], corners[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _box_areas(boxes: np.ndarray) -> np.ndarray:
    """Returns the area of every box in the chain."""
    return np.array(
        [_quad_area(*_box_corners(boxes, box)) for box in range(boxes.shape[0] - 1)]
    )


def _points_per_box(boxes: np.ndarray, n_points: int) -> np.ndarray:
    """Splits n_points across the boxes proportional to their area."""
    fractions = _box_areas(boxes) / _box_areas(boxes).sum()
    counts = np.floor(fractions * n_points).astype(int)
    counts[0] += n_points - counts.sum()
    return counts


def _sample_inside_box(boxes: np.ndarray, box: int, count: int) -> np.ndarray:
    """Uniformly samples count points in a box's bilinear (u, v) coordinates."""
    u = np.random.uniform(0.0, 1.0, count)
    v = np.random.uniform(0.0, 1.0, count)
    return _evaluate_boxes(boxes, np.full(count, box), u, v)


def _sample_inside_boxes(boxes: np.ndarray, n_points: int) -> np.ndarray:
    """Samples n_points across all boxes, weighted by box area."""
    counts = _points_per_box(boxes, n_points)
    samples = [
        _sample_inside_box(boxes, box, count) for box, count in enumerate(counts)
    ]
    return np.concatenate(samples, axis=0)


def _centerline_directions(boxes: np.ndarray) -> np.ndarray:
    """Unit tangents (x, z) of the chain centre line, one per box."""
    centers = boxes.mean(axis=1)
    directions = np.diff(centers, axis=0)
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)
