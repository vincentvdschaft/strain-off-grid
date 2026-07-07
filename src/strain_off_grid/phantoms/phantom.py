from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import final

import numpy as np

from ..velocities import Velocities3D
from .dataclass_saving import HDF5Mixin


@dataclass
class DynamicPhantom(HDF5Mixin, ABC):
    """Base class for dynamic phantoms that can translate points over time."""

    A: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3×3 linear part
    b: np.ndarray = field(default_factory=lambda: np.zeros(3))  # translation
    period: float = 1.0

    @final
    def _wrap_time(self, t: float) -> float:
        """Wraps the given time t to the range [0, period)."""
        return t % self.period

    @final
    def _to_world(self, positions_local: np.ndarray) -> np.ndarray:
        """Transforms the given positions from local to world coordinates."""
        return positions_local @ self.A.T + self.b[None, :]

    @final
    def sample_points(self, n_points: int) -> np.ndarray:
        """Samples n_points from the phantom at time t."""
        positions = self._sample_points(n_points)
        return self._to_world(positions)

    @final
    def points_in_phantom(self, positions: np.ndarray, t: float) -> np.ndarray:
        """Returns a boolean array indicating which positions are inside the phantom at time t."""
        positions_local_t = self._to_local(positions)
        positions_local_t0 = self._inverse_translate_to_time(positions_local_t, t)
        mask = self._points_in_phantom(positions_local_t0)
        return mask.reshape(*positions.shape[:-1])

    @final
    def _to_local(self, positions: np.ndarray) -> np.ndarray:
        """Transforms the given positions from world to local coordinates."""
        if positions.shape[-1] == 2:
            positions = np.stack(
                [
                    positions[..., 0],
                    np.zeros_like(positions[..., 0]),
                    positions[..., -1],
                ],
                axis=-1,
            )
        return (positions - self.b[None, :]) @ np.linalg.inv(self.A).T

    @final
    def compute_strain_rate(
        self, positions: np.ndarray, directions: np.ndarray, t: float, dt: float
    ) -> np.ndarray:
        """Computes the strain rate along specified directions at given positions and time.

        Parameters
        ----------
        positions : np.ndarray
            An array of shape (n_points, 3) representing the positions in world coordinates.
        directions : np.ndarray
            An array of shape (n_points, 3) representing the unit direction vectors along which to compute strain rates.
        t : float
            The time at which to compute the strain rate.
        dt : float
            The time increment for computing the strain rate.

        Returns
        -------
        np.ndarray
            An array of shape (n_points,) containing the computed strain rates along the specified directions.
        """
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

        distance = 1e-6
        positions_min = positions - distance / 2 * directions
        positions_max = positions + distance / 2 * directions
        positions_min_t2 = self.translate_from_time_to_time(positions_min, t, t + dt)
        positions_max_t2 = self.translate_from_time_to_time(positions_max, t, t + dt)
        distances_t2 = np.linalg.norm(positions_max_t2 - positions_min_t2, axis=1)
        strain_rate = (distances_t2 - distance) / (distance * dt)
        return strain_rate

    @final
    def principal_axis(self, positions: np.ndarray) -> np.ndarray:
        """Computes the principal axis of the phantom at the given positions."""
        positions_local = self._to_local(positions)
        positions_local_t0 = self._inverse_translate_to_time(positions_local, 0.0)
        return self._principal_axis(positions_local_t0)

    @final
    def translate_to_time(self, positions: np.ndarray, t: float) -> np.ndarray:
        """Translates the given positions from time 0 to the specified time t."""
        return self._to_world(self._translate_to_time(self._to_local(positions), t))

    @final
    def translate_from_time_to_time(
        self, positions: np.ndarray, t0: float, t1: float
    ) -> np.ndarray:
        """Translates the given positions from time t0 to time t1."""
        t0, t1 = self._wrap_time(t0), self._wrap_time(t1)
        return self.translate_to_time(self.inverse_translate_to_time(positions, t0), t1)

    @final
    def inverse_translate_to_time(self, positions: np.ndarray, t: float) -> np.ndarray:
        """Translates the given positions from time t back to time 0."""
        return self._to_world(
            self._inverse_translate_to_time(self._to_local(positions), t)
        )

    @final
    def get_offsets(self, positions: np.ndarray, t: float, dt: float) -> np.ndarray:
        translated_positions = self.translate_from_time_to_time(positions, t, t + dt)
        offsets = translated_positions - positions
        return offsets

    @final
    def get_velocities(
        self, positions: np.ndarray, t: float, dt: float
    ) -> Velocities3D:
        """Computes the velocities of the given positions at time t."""
        return Velocities3D(
            positions=positions,
            velocities=self.get_offsets(_2d_to_3d(positions), t, dt) / dt,
            timestamp=t,
        )

    @abstractmethod
    def _translate_to_time(self, positions_local: np.ndarray, t: float) -> np.ndarray:
        """Translates the given positions from time 0 to the specified time t."""
        return positions_local

    @abstractmethod
    def _inverse_translate_to_time(
        self, positions_local: np.ndarray, t: float
    ) -> np.ndarray:
        """Translates the given positions from time t back to time 0."""
        return positions_local

    @abstractmethod
    def _points_in_phantom(self, positions_local_t0: np.ndarray) -> np.ndarray:
        """Returns a boolean array indicating which positions are inside the phantom at time 0.

        Args:
            positions_local_t0: An array of shape (N, 3) representing the positions in local coordinates at time 0.

        Returns:
            A boolean array of shape (N,) indicating which positions are inside the phantom at time 0.
        """
        raise NotImplementedError("Subclasses must implement _points_in_phantom()")

    @abstractmethod
    def _sample_points(self, n_points: int) -> np.ndarray:
        """Samples n_points from the phantom at time 0."""
        return np.zeros((n_points, 3))

    def _principal_axis(self, positions_local_t0: np.ndarray) -> np.ndarray:
        """Computes the principal axis of the phantom at the given local positions."""
        raise NotImplementedError("Subclasses must implement _principal_axis()")


def _2d_to_3d(arr, axis: int = -1):
    """Converts an array of shape (..., 2, ...) to shape (..., 3, ...) by inserting a zero in the specified axis."""
    arr = np.asarray(arr)
    if arr.shape[axis] == 3:
        return arr
    slice_before = [slice(None)] * arr.ndim
    slice_after = [slice(None)] * arr.ndim
    slice_before[axis] = slice(0, 1)
    slice_after[axis] = slice(1, 2)
    slice_before, slice_after = tuple(slice_before), tuple(slice_after)

    return np.concatenate(
        [arr[slice_before], 0.0 * arr[slice_after], arr[slice_after]], axis=axis
    )


@dataclass
class StaticPhantom(DynamicPhantom):
    """A phantom consisting of a single point that does not move over time."""

    def _sample_points(self, n_points: int) -> np.ndarray:
        """Returns a single stationary point at the local origin."""
        return np.zeros((1, 3))
