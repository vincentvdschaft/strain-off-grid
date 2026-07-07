"""Limits classes for storing image spatial extents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Union

import numpy as np


@dataclass(frozen=True)
class Limits:
    min: float
    max: float

    def __post_init__(self):
        true_min = float(min(self.min, self.max))
        true_max = float(max(self.min, self.max))
        object.__setattr__(self, "min", true_min)
        object.__setattr__(self, "max", true_max)

    def size(self) -> float:
        return self.max - self.min

    def __iter__(self):
        yield self.min
        yield self.max

    def __repr__(self):
        return f"Limits({self.min}, {self.max})"

    def __hash__(self):
        return hash((self.min, self.max))

    def __eq__(self, other):
        if not isinstance(other, Limits):
            other = Limits(*other)
        return self.min == other.min and self.max == other.max

    def __add__(self, other):
        if not isinstance(other, Limits):
            other = Limits(*other)
        return LimitsND((self, other))


LimitsLike = Union["Limits", tuple[float, float], Sequence[float]]
LimitsNDInput = Union[
    "LimitsND",
    Sequence[LimitsLike],  # tuple of tuples / list of Limits
    Sequence[float],  # flat tuple
    np.ndarray,  # (N, 2) or flat (2N,)
]


@dataclass(frozen=True)
class LimitsND:
    limits: tuple[Limits, ...] = field(default_factory=tuple)

    def __post_init__(self):
        raw = self.limits.limits if isinstance(self.limits, LimitsND) else self.limits

        # already a LimitsND-shaped list of Limits objects
        if all(isinstance(item, Limits) for item in raw):
            object.__setattr__(self, "limits", tuple(raw))
            return

        arr = np.asarray(raw, dtype=float)

        if arr.ndim == 1:
            if arr.size % 2 != 0:
                raise ValueError(
                    f"Flat input must have an even number of elements, got {arr.size}"
                )
            arr = arr.reshape(-1, 2)
        elif arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected shape (N, 2) or flat (2N,), got {arr.shape}")

        object.__setattr__(self, "limits", tuple(Limits(lo, hi) for lo, hi in arr))

    @property
    def ndim(self) -> int:
        return len(self.limits)

    def __getitem__(self, index) -> LimitsND | Limits:
        # Mirrors numpy's scalar-vs-array indexing rules for a 1D array: a bare
        # integer collapses to a single element, everything else stays an array.
        if isinstance(index, (int, np.integer)):
            return self.limits[index]
        if index is Ellipsis:
            return LimitsND(list(self.limits))
        if isinstance(index, slice):
            return LimitsND(self.limits[index])
        if isinstance(index, tuple):
            if len(index) == 1:
                return self[index[0]]
            raise IndexError(
                f"Too many indices for LimitsND: index {index} for {self.ndim} dimension(s)"
            )

        index_array = np.asarray(index)
        if index_array.dtype == bool:
            if index_array.shape != (len(self.limits),):
                raise IndexError(
                    f"Boolean index shape {index_array.shape} does not match "
                    f"LimitsND length ({len(self.limits)},)"
                )
            return LimitsND(
                [limit for limit, keep in zip(self.limits, index_array) if keep]
            )
        return LimitsND([self.limits[i] for i in index_array])

    def __setitem__(self, index, value: LimitsLike):
        # Mirrors __getitem__'s dispatch: an int sets a single element,
        # everything else selects a run of elements and broadcasts the same
        # LimitsLike value into each of them (each gets its own Limits
        # instance, never a shared one, since Limits is mutable).
        min_, max_ = value if isinstance(value, Limits) else Limits(*value)

        if isinstance(index, (int, np.integer)):
            self.limits[index] = Limits(min_, max_)
            return
        if index is Ellipsis:
            index = slice(None)
        if isinstance(index, tuple):
            if len(index) == 1:
                self[index[0]] = value
                return
            raise IndexError(
                f"Too many indices for LimitsND: index {index} for {self.ndim} dimension(s)"
            )

        if isinstance(index, slice):
            positions = range(len(self.limits))[index]
        else:
            index_array = np.asarray(index)
            if index_array.dtype == bool:
                if index_array.shape != (len(self.limits),):
                    raise IndexError(
                        f"Boolean index shape {index_array.shape} does not match "
                        f"LimitsND length ({len(self.limits)},)"
                    )
                positions = np.flatnonzero(index_array)
            else:
                positions = index_array

        for position in positions:
            self.limits[int(position)] = Limits(min_, max_)

    def sizes(self) -> np.ndarray:
        return np.array([limit.size() for limit in self.limits])

    @classmethod
    def from_extent(cls, extent: Extent) -> LimitsND:
        """Create a LimitsND object from a legacy Extent object."""
        limits = tuple(
            Limits(extent.start(dim), extent.end(dim)) for dim in range(extent.ndim)
        )
        return cls(limits)

    @classmethod
    def from_shape(cls, shape: tuple[int, ...]) -> LimitsND:
        """Create a LimitsND object from a shape tuple, where each dimension's limits are (0, size-1)."""
        limits = tuple(Limits(0, size - 1) for size in shape)
        return cls(limits)

    def __iter__(self):
        return iter(self.limits)

    def __len__(self):
        return len(self.limits)

    def origin(self) -> np.ndarray:
        """Return the origin (min values) of the limits as a numpy array."""
        return np.array([limit.min for limit in self.limits])

    def __hash__(self):
        return hash(tuple(self))

    def __add__(self, other):
        if not isinstance(other, LimitsND):
            other = LimitsND(other)
        return LimitsND(self.limits + other.limits)

    def make_grid(
        self, shape: tuple[int, ...] = None, pixel_sizes: Sequence[float] | float = None
    ) -> np.ndarray:
        """Create a meshgrid of coordinates for the given shape, using the limits.

        Returns a grid of shape (*shape, ndim), where the last dimension contains the coordinates for each axis.

        If the limits represent (z, y, x), then the shape should be (nz, ny, nx), and the output will be (nz, ny, nx, zyx).

        If `shape` is not provided, it will be computed from `pixel_sizes` if given.
        """
        if shape is not None and pixel_sizes is not None:
            raise ValueError("Provide either 'shape' or 'pixel_sizes', not both.")
        if shape is None:
            if pixel_sizes is None:
                raise ValueError("Either 'shape' or 'pixel_sizes' must be provided.")
            shape = self.get_shape_from_pixel_sizes(pixel_sizes)
        return self._make_grid(shape)

    def _make_grid(self, shape: tuple[int, ...]) -> np.ndarray:
        """Create a meshgrid of coordinates for the given shape, using the limits.

        Returns a grid of shape (*shape, ndim), where the last dimension contains the coordinates for each axis.

        If the limits represent (z, y, x), then the shape should be (nz, ny, nx), and the output will be (nz, ny, nx, zyx).
        """
        if len(shape) != self.ndim:
            raise ValueError(
                f"Shape length {len(shape)} does not match number of dimensions {self.ndim}"
            )
        grids = []
        for dim, (limit, size) in enumerate(zip(self.limits, shape)):
            grids.append(np.linspace(limit.min, limit.max, size))
        return np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)

    def fitted_to_pixel_sizes(self, pixel_sizes: Sequence[float] | float) -> LimitsND:
        """Adjust the limits to fit an integer number of pixels given the pixel sizes.

        The pixel sizes can be a single float (applied to all dimensions) or a sequence of floats (one per dimension).
        """
        if isinstance(pixel_sizes, (float, int)):
            pixel_sizes = [float(pixel_sizes)] * self.ndim
        elif len(pixel_sizes) != self.ndim:
            raise ValueError(
                f"Pixel sizes length {len(pixel_sizes)} does not match number of dimensions {self.ndim}"
            )

        new_limits = tuple(
            Limits(limit.min, limit.min + round(limit.size() / pixel_size) * pixel_size)
            for limit, pixel_size in zip(self.limits, pixel_sizes)
        )
        return LimitsND(new_limits)

    def get_shape_from_pixel_sizes(
        self, pixel_sizes: Sequence[float] | float
    ) -> tuple[int, ...]:
        """Compute the shape (number of pixels) for each dimension given the pixel sizes.

        The pixel sizes can be a single float (applied to all dimensions) or a sequence of floats (one per dimension).
        """
        if isinstance(pixel_sizes, (float, int)):
            pixel_sizes = [float(pixel_sizes)] * self.ndim
        elif len(pixel_sizes) != self.ndim:
            raise ValueError(
                f"Pixel sizes length {len(pixel_sizes)} does not match number of dimensions {self.ndim}"
            )

        return tuple(
            max(1, round(limit.size() / pixel_size))
            for limit, pixel_size in zip(self.limits, pixel_sizes)
        )

    @property
    def aspect(self) -> np.ndarray:
        """Return the aspect ratio (size of each dimension) as a numpy array."""
        if self.ndim < 2:
            raise ValueError("Aspect ratio is only defined for 2 or more dimensions.")

        return self[-1].size() / self[-2].size()


class Extent(tuple):
    """Legacy flat-tuple encoding of spatial limits: (dim0_min, dim0_max, dim1_min, dim1_max, ...).

    Kept only to decode HDF5 files written before the switch to LimitsND.
    """

    def __new__(cls, initializer):
        initializer = [float(value) for value in initializer]
        assert len(initializer) % 2 == 0, "Extent must have an even number of elements."

        return super(Extent, cls).__new__(cls, initializer)

    @property
    def ndim(self):
        return len(self) // 2

    def sort(self):
        """Sorts the extent such that dim0_min < dim0_max, dim1_min < dim1_max, ..."""
        initializer = []
        for val0, val1 in zip(self[::2], self[1::2]):
            initializer.append(min(val0, val1))
            initializer.append(max(val0, val1))

        return Extent(initializer)

    def start(self, dim):
        """Returns the start value of the given dimension."""
        return self[dim * 2]

    def end(self, dim):
        """Returns the end value of the given dimension."""
        return self[dim * 2 + 1]

    def __hash__(self):
        return hash(tuple(self))


def compute_limits_after_slicing(current_shape, limits: LimitsND, key) -> LimitsND:
    """Compute the new LimitsND after applying a numpy-style index key.

    Handles slices (including ``:``), integers (dimension removal),
    ``None`` / ``np.newaxis`` (new axis insertion), and ``Ellipsis``.
    """
    key = _expand_ellipsis(key, limits.ndim)

    new_limits = []
    original_dim = 0

    for key_element in key:
        if key_element is None:
            new_limits.append(Limits(0, 0))  # new axis carries no spatial meaning
        elif isinstance(key_element, int):
            original_dim += 1  # integer indexing removes this dimension
        else:
            indices = range(current_shape[original_dim])[key_element]
            new_limits.append(
                _limits_for_selected_indices(
                    limits[original_dim], current_shape[original_dim], indices
                )
            )
            original_dim += 1

    return LimitsND(tuple(new_limits))


def select_axis_values_after_slicing(values, key, fill) -> tuple:
    """Reorder a per-axis sequence to match a numpy-style index key.

    `values` has one entry per array dimension. Integer indexing drops the
    corresponding entry, ``None`` / ``np.newaxis`` inserts a new entry equal to
    ``fill``, and slices keep the entry unchanged.
    """
    key = _expand_ellipsis(key, len(values))
    result = []
    original_dim = 0
    for key_element in key:
        if key_element is None:
            result.append(fill)
        elif isinstance(key_element, int):
            original_dim += 1
        else:
            result.append(values[original_dim])
            original_dim += 1
    return tuple(result)


def _limits_for_selected_indices(dim_limits: Limits, num_pixels, indices) -> Limits:
    """Return the Limits spanning a sequence of selected pixel indices."""
    if len(indices) == 0:
        return Limits(dim_limits.min, dim_limits.min)
    return Limits(
        _index_to_coordinate(dim_limits, num_pixels, indices[0]),
        _index_to_coordinate(dim_limits, num_pixels, indices[-1]),
    )


def _expand_ellipsis(key, ndim):
    """Expand Ellipsis to explicit slice(None) objects.

    None (np.newaxis) items are preserved and do NOT count as consuming an array
    dimension — only slices and integers consume dimensions.
    """
    if not isinstance(key, tuple):
        key = (key,)

    n_consuming = sum(1 for k in key if k is not None and k is not Ellipsis)

    if Ellipsis not in key:
        return key + (slice(None),) * (ndim - n_consuming)

    ellipsis_pos = key.index(Ellipsis)
    n_missing = ndim - n_consuming
    return key[:ellipsis_pos] + (slice(None),) * n_missing + key[ellipsis_pos + 1 :]


def _index_to_coordinate(dim_limits: Limits, num_pixels, pixel_index) -> float:
    """Convert a pixel index to its physical coordinate (linear interpolation)."""
    if num_pixels <= 1:
        return dim_limits.min
    return dim_limits.min + pixel_index * dim_limits.size() / (num_pixels - 1)
