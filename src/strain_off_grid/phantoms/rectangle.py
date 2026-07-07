from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .phantom import DynamicPhantom

_PLANE_AXES = [0, 2]  # the polygon's 2D coordinates map to the x and z axes
_ELEVATION_AXIS = 1  # y is the thin out-of-plane (elevation) direction


@dataclass
class RectanglePhantom(DynamicPhantom):
    width: float = 6e-3
    height: float = 4e-3
    max_vertical_strain: float = 0.2
    max_horizontal_strain: float = 0.2
    frequency: float = 1.0

    def _scale_x(self, t: float) -> float:
        """Returns the scaling factor in the x direction at time t."""
        return self._scale(t, max_strain=self.max_horizontal_strain)

    def _scale_y(self, t: float) -> float:
        """Returns the scaling factor in the y direction at time t."""
        return self._scale(t, max_strain=self.max_vertical_strain)

    def _scale(
        self, t: float, systolic_fraction: float = 0.35, max_strain: float = 2.0
    ) -> float:
        """Returns the scaling factor at time t.

        Contraction (systole) happens over `systolic_fraction` of the cycle and
        is fast; relaxation/filling (diastole) takes up the rest and is slow.
        """
        T = 1.0 / self.frequency
        phase = (t % T) / T  # normalized position in cycle, [0, 1)

        # theta rises slowly 0 -> pi over diastole (long), then falls
        # quickly pi -> 2*pi over systole (short).
        diastolic_fraction = 1 - systolic_fraction
        theta = np.where(
            phase < diastolic_fraction,
            np.pi * phase / diastolic_fraction,
            np.pi + np.pi * (phase - diastolic_fraction) / systolic_fraction,
        )

        return 1 + max_strain * (1 - np.cos(theta)) / 2

    def _translate_to_time(self, positions_local: np.ndarray, t: float) -> np.ndarray:
        """Translates the given positions from time 0 to the specified time t."""
        return (
            positions_local
            * np.array([self._scale_x(t), 1.0, self._scale_y(t)])[None, :]
        )

    def _inverse_translate_to_time(
        self, positions_local: np.ndarray, t: float
    ) -> np.ndarray:
        """Translates the given positions from time t back to time 0."""
        return (
            positions_local
            / np.array([self._scale_x(t), 1.0, self._scale_y(t)])[None, :]
        )

    def _sample_points(self, n_points: int) -> np.ndarray:
        """Samples n_points from the phantom at time 0."""
        # Sample points uniformly in the rectangle at time 0
        x = np.random.uniform(-self.width / 2, self.width / 2, size=n_points)
        y = np.zeros(n_points)  # y is the thin out-of-plane direction
        z = np.random.uniform(-self.height / 2, self.height / 2, size=n_points)
        return np.stack([x, y, z], axis=1)

    def points_in_phantom(self, positions: np.ndarray, t: float) -> np.ndarray:
        """Returns a boolean array indicating which positions are inside the phantom at time t."""
        positions_local_t = self._to_local(positions)
        positions_local_t0 = self._inverse_translate_to_time(positions_local_t, t)

        x_in = np.abs(positions_local_t0[:, 0]) <= self.width / 2
        z_in = np.abs(positions_local_t0[:, -1]) <= self.height / 2

        return x_in & z_in
