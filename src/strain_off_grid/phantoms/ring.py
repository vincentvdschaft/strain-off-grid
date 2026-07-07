from dataclasses import dataclass

import numpy as np

from strain_off_grid.phantoms import DynamicPhantom


@dataclass
class ShortAxisPhantom(DynamicPhantom):
    """A ring phantom simulating the short-axis view of a heart.

    The ring lies in the local x-z plane (y is the out-of-plane slice
    thickness). Its inner and outer diameter each oscillate independently
    between a min and max value over a cardiac cycle of duration 1 (in the
    same time units as `t`), and the ring is twisted about the long (y) axis
    to simulate cardiac torsion.
    """

    inner_diameter_min: float = 31.5e-3
    inner_diameter_max: float = 38.5e-3
    outer_diameter_min: float = 40.5e-3
    outer_diameter_max: float = 49.5e-3
    thickness_variation: float = 2e-3
    torsion_amplitude: float = np.deg2rad(10.0)  # peak twist angle [rad]

    def _radius_at(self, diameter_min: float, diameter_max: float, t: float) -> float:
        """Diameter/2, oscillating between min and max over a cycle of duration 1."""
        phase = np.sin(2 * np.pi * t)
        diameter = diameter_min + (phase + 1.0) / 2.0 * (diameter_max - diameter_min)
        return diameter / 2.0

    def _remap_radius(
        self,
        x: np.ndarray,
        z: np.ndarray,
        from_inner: float,
        from_outer: float,
        to_inner: float,
        to_outer: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rescales (x, z) so that the radial band [from_inner, from_outer] maps to [to_inner, to_outer]."""
        radius = np.maximum(np.hypot(x, z), 1e-12)
        frac = (radius - from_inner) / (from_outer - from_inner)
        target_radius = to_inner + frac * (to_outer - to_inner)
        scale = target_radius / radius
        return x * scale, z * scale

    def _twist(
        self, x: np.ndarray, z: np.ndarray, angle: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rotates (x, z) around the origin (i.e. around the y/long axis) by angle."""
        c, s = np.cos(angle), np.sin(angle)
        return x * c - z * s, x * s + z * c

    def _translate_to_time(self, positions_local: np.ndarray, t: float) -> np.ndarray:
        """Translates the given positions from time 0 to the specified time t."""
        inner0 = self._radius_at(self.inner_diameter_min, self.inner_diameter_max, 0.0)
        outer0 = self._radius_at(self.outer_diameter_min, self.outer_diameter_max, 0.0)
        inner_t = self._radius_at(self.inner_diameter_min, self.inner_diameter_max, t)
        outer_t = self._radius_at(self.outer_diameter_min, self.outer_diameter_max, t)

        x, y, z = (
            positions_local[..., 0],
            positions_local[..., 1],
            positions_local[..., 2],
        )
        x, z = self._remap_radius(x, z, inner0, outer0, inner_t, outer_t)
        twist_angle = self.torsion_amplitude * np.sin(2 * np.pi * t)
        x, z = self._twist(x, z, twist_angle)
        return np.stack([x, y, z], axis=-1)

    def _inverse_translate_to_time(
        self, positions_local: np.ndarray, t: float
    ) -> np.ndarray:
        """Translates the given positions from time t back to time 0."""
        inner0 = self._radius_at(self.inner_diameter_min, self.inner_diameter_max, 0.0)
        outer0 = self._radius_at(self.outer_diameter_min, self.outer_diameter_max, 0.0)
        inner_t = self._radius_at(self.inner_diameter_min, self.inner_diameter_max, t)
        outer_t = self._radius_at(self.outer_diameter_min, self.outer_diameter_max, t)

        x, y, z = (
            positions_local[..., 0],
            positions_local[..., 1],
            positions_local[..., 2],
        )
        twist_angle = self.torsion_amplitude * np.sin(2 * np.pi * t)
        x, z = self._twist(x, z, -twist_angle)
        x, z = self._remap_radius(x, z, inner_t, outer_t, inner0, outer0)
        return np.stack([x, y, z], axis=-1)

    def _sample_points(self, n_points: int) -> np.ndarray:
        """Samples n_points from the phantom at time 0."""
        inner0 = self._radius_at(self.inner_diameter_min, self.inner_diameter_max, 0.0)
        outer0 = self._radius_at(self.outer_diameter_min, self.outer_diameter_max, 0.0)

        total_n_points = 0
        positions_list = []
        while total_n_points < n_points:
            positions = np.random.uniform(-1, 1, size=(n_points * 2, 3))
            positions = (
                positions
                * (np.array([outer0, self.thickness_variation / 2, outer0]))[None, :]
            )
            radius = np.linalg.norm(positions[:, np.array([0, 2])], axis=1)
            mask = (radius >= inner0) & (radius <= outer0)
            positions = positions[mask]
            total_n_points += positions.shape[0]
            positions_list.append(positions)

        positions = np.concatenate(positions_list, axis=0)[:n_points]
        return positions

    def distances_to_edge(self, positions: np.ndarray, t: float) -> np.ndarray:
        """Distance from each (n_points, 3) coordinate to the phantom boundary at time t."""
        inner = self._radius_at(self.inner_diameter_min, self.inner_diameter_max, t)
        outer = self._radius_at(self.outer_diameter_min, self.outer_diameter_max, t)
        radius = np.linalg.norm(positions[:, np.array([0, 2])], axis=1)
        distance_to_inner = radius - inner
        distance_to_outer = outer - radius
        return np.minimum(distance_to_inner, distance_to_outer)

    def _points_in_phantom(self, positions_local_t0: np.ndarray) -> np.ndarray:
        inner = self._radius_at(self.inner_diameter_min, self.inner_diameter_max, 0.0)
        outer = self._radius_at(self.outer_diameter_min, self.outer_diameter_max, 0.0)
        radius = np.linalg.norm(positions_local_t0[..., np.array([0, 2])], axis=-1)
        return (radius >= inner) & (radius <= outer)

    def _principal_axis(self, positions_local_t0: np.ndarray) -> np.ndarray:
        """Computes the principal axis of the phantom at the given local positions."""
        angle = np.arctan2(positions_local_t0[..., -1], positions_local_t0[..., 0])
        return np.stack([np.sin(angle), -np.cos(angle)], axis=-1)
