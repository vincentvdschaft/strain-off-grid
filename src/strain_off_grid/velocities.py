from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from strain_off_grid.phantoms.dataclass_saving import HDF5Mixin

from .dimconvert import promote_2d_to_3d, reduce_3d_to_2d


@dataclass
class Velocities2D(HDF5Mixin):
    """A class to represent 2D velocities at a set of positions."""

    positions: np.ndarray  # shape (N, 2)
    velocities: np.ndarray  # shape (N, 2)
    timestamp: float = field(default=0.0)

    def __post_init__(self):
        self.positions = reduce_3d_to_2d(np.asarray(self.positions, dtype=float))
        self.velocities = reduce_3d_to_2d(np.asarray(self.velocities, dtype=float))
        if self.positions.shape != self.velocities.shape:
            raise ValueError(
                f"Positions and velocities must have the same shape, got {self.positions.shape} and {self.velocities.shape}"
            )

    def __getitem__(self, index) -> Velocities2D:
        """Returns a new Velocities2D object with the specified index."""
        return Velocities2D(
            positions=self.positions[index],
            velocities=self.velocities[index],
            timestamp=self.timestamp,
        )

    def __repr__(self) -> str:
        return f"Velocities2D(positions={self.positions}, velocities={self.velocities}, timestamp={self.timestamp})"

    def to_3d(self) -> Velocities3D:
        """Returns a Velocities3D object by promoting the 2D positions and velocities to 3D."""
        return Velocities3D(
            positions=promote_2d_to_3d(self.positions),
            velocities=promote_2d_to_3d(self.velocities),
        )

    def with_velocities(self, new_velocities: np.ndarray) -> Velocities2D:
        """Returns a new Velocities2D object with the same positions but new velocities."""
        new_velocities = reduce_3d_to_2d(np.asarray(new_velocities, dtype=float))
        if new_velocities.shape != self.velocities.shape:
            raise ValueError(
                f"New velocities must have the same shape as existing velocities, got {new_velocities.shape} and {self.velocities.shape}"
            )
        return Velocities2D(positions=self.positions, velocities=new_velocities)

    def __len__(self) -> int:
        """Returns the number of velocity vectors."""
        return self.positions.shape[0]


@dataclass
class Velocities3D(HDF5Mixin):
    """A class to represent 3D velocities at a set of positions."""

    positions: np.ndarray  # shape (N, 3)
    velocities: np.ndarray  # shape (N, 3)
    timestamp: float = field(default=0.0)

    def __post_init__(self):
        self.positions = promote_2d_to_3d(np.asarray(self.positions, dtype=float))
        self.velocities = promote_2d_to_3d(np.asarray(self.velocities, dtype=float))
        if self.positions.shape != self.velocities.shape:
            raise ValueError(
                f"Positions and velocities must have the same shape, got {self.positions.shape} and {self.velocities.shape}"
            )

    def __getitem__(self, index) -> Velocities3D:
        """Returns a new Velocities3D object with the specified index."""
        return Velocities3D(
            positions=self.positions[index],
            velocities=self.velocities[index],
            timestamp=self.timestamp,
        )

    def __repr__(self) -> str:
        return f"Velocities3D(positions={self.positions}, velocities={self.velocities}, timestamp={self.timestamp})"

    def to_2d(self) -> Velocities2D:
        """Returns a Velocities2D object by reducing the 3D positions and velocities to 2D."""
        return Velocities2D(
            positions=reduce_3d_to_2d(self.positions),
            velocities=reduce_3d_to_2d(self.velocities),
            timestamp=self.timestamp,
        )

    def with_velocities(self, new_velocities: np.ndarray) -> Velocities3D:
        """Returns a new Velocities3D object with the same positions but new velocities."""
        new_velocities = promote_2d_to_3d(np.asarray(new_velocities, dtype=float))
        if new_velocities.shape != self.velocities.shape:
            raise ValueError(
                f"New velocities must have the same shape as existing velocities, got {new_velocities.shape} and {self.velocities.shape}"
            )
        return Velocities3D(
            positions=self.positions,
            velocities=new_velocities,
            timestamp=self.timestamp,
        )

    def __len__(self) -> int:
        """Returns the number of velocity vectors."""
        return self.positions.shape[0]
