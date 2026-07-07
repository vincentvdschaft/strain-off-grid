from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.image
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter1d

from .clahe import clahe
from .dynamic_range import apply_dynamic_range_curve
from .extent import (
    Limits,
    LimitsND,
    LimitsNDInput,
    compute_limits_after_slicing,
    select_axis_values_after_slicing,
)
from .match_histograms import match_histograms
from .saving import load_hdf5_image, save_hdf5_image


class NDImage:
    def __init__(
        self,
        array,
        limits: LimitsNDInput | None = None,
        metadata=None,
        labels=None,
        units=None,
    ):
        self.array = np.asarray(array).copy()
        self.array.setflags(write=True)
        self._limits = (
            LimitsND(limits)
            if limits is not None
            else LimitsND.from_shape(self.array.shape)
        )
        _check_ndimage_initializers(self.array, self._limits)
        self._metadata = {}
        self._labels = self._get_initialized_labels(labels)
        self._units = self._get_initialized_units(units)
        if metadata is not None:
            self.update_metadata(metadata)

    @property
    def limits(self) -> LimitsND:
        return self._limits

    @property
    def labels(self) -> tuple:
        return self._labels

    @property
    def units(self) -> tuple:
        return self._units

    def __repr__(self) -> str:
        return (
            f"NDImage(array={self.shape}, limits={self.limits!r}, "
            f"labels={self.labels}, units={self.units})"
        )

    def _get_initialized_labels(self, labels=None) -> tuple:
        if labels is not None:
            assert len(labels) == self.ndim, (
                "The number of labels must match the number of dimensions. "
                f"Got {len(labels)} and {self.ndim}."
            )
            return tuple(labels)

        labels = [f"dim_{i}" for i in range(self.ndim)]
        labels[-1] = "x"
        if self.ndim > 1:
            labels[-2] = "y"
        return tuple(labels)

    def _get_initialized_units(self, units=None) -> tuple:
        if units is None:
            return tuple("" for _ in range(self.ndim))
        assert len(units) == self.ndim, (
            "The number of units must match the number of dimensions. "
            f"Got {len(units)} and {self.ndim}."
        )
        return tuple(units)

    # ==========================================================================
    # Numpy array interface
    # ==========================================================================
    # --- 1️⃣ Allow automatic conversion to numpy array ---
    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.array, dtype=dtype)

    # --- 2️⃣ Intercept numpy ufuncs like +, -, *, sin, etc. ---
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # unwrap NDImage objects to their arrays
        unwrapped = []
        for inp in inputs:
            if isinstance(inp, NDImage):
                unwrapped.append(inp.array)
            else:
                unwrapped.append(inp)

        # apply the ufunc
        result = getattr(ufunc, method)(*unwrapped, **kwargs)

        # if result is an ndarray (not scalar), wrap it back
        if isinstance(result, np.ndarray):
            return self.with_array(result)
        else:
            return result

    # --- 3️⃣ Optional: handle numpy high-level functions like np.concatenate ---
    def __array_function__(self, func, types, args, kwargs):
        # if NDImage not supported, fall back to ndarray
        if not all(issubclass(t, NDImage) for t in types):
            return NotImplemented

        if func is np.concatenate:
            arrays = [a.array for a in args[0]]
            new_array = np.concatenate(arrays, **kwargs)
            return self.with_array(new_array)
        return NotImplemented

    # ==========================================================================
    # Sizes and dimensions
    # ==========================================================================
    @property
    def size(self) -> int:
        """Returns the total number of pixels in the image."""
        return self.array.size

    def pixel_size(self, dim) -> float:
        """Returns the pixel size in the given dimension."""
        return float(self.pixel_sizes[dim])

    @property
    def pixel_sizes(self) -> np.ndarray:
        return compute_pixel_sizes(self.limits, self.shape)

    @property
    def pixel_scales(self) -> np.ndarray:
        """Returns the scaling to apply to convert pixel indices to physical coordinates."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.pixel_sizes > 0, self.pixel_sizes, 1e-6)

    @property
    def extent(self) -> tuple:
        """Returns the (x0, x1, y0, y1) extent for use with matplotlib's imshow.

        Uses the last two dimensions, which are the (y, x) image-plane axes.
        """
        return tuple([limit for limits in self.limits[::-1] for limit in limits])

    @property
    def extent_imshow(self) -> tuple:
        """Returns the (x0, x1, y0, y1) extent for use with matplotlib's imshow.

        Uses the last two dimensions, which are the (y, x) image-plane axes.
        """
        return get_limits_imshow(self.limits, self.shape)

    @property
    def translate(self) -> np.ndarray:
        """Returns the translation vector to convert pixel indices to physical coordinates."""
        return self.limits.origin()

    @property
    def scale(self) -> np.ndarray:
        """Returns the scaling to apply to convert pixel indices to physical coordinates."""
        return self.pixel_scales

    # ==========================================================================
    # Forward properties
    # ==========================================================================
    @property
    def shape(self) -> tuple:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    def T(self) -> NDImage:
        """Returns the transposed image."""
        axes = list(reversed(range(self.ndim)))
        return self.transpose(axes)

    # ==========================================================================
    # Metadata handling
    # ==========================================================================

    @property
    def metadata(self) -> dict:
        """Return metadata of image."""
        return self._metadata

    @metadata.setter
    def metadata(self, value) -> None:
        """Set metadata of image."""
        assert isinstance(value, dict), "Metadata must be a dictionary."
        self._metadata = value

    def add_metadata(self, key, value) -> NDImage:
        """Add metadata to image."""
        self._metadata[key] = value
        return self

    def update_metadata(self, metadata) -> NDImage:
        """Update metadata of image."""
        self._metadata.update(metadata)
        return self

    def append_metadata(self, key, value) -> NDImage:
        """Add metadata assuming the key is a list."""

        if key not in self.metadata:
            self._metadata[key] = []
        elif not isinstance(self.metadata[key], list):
            raise ValueError(f"Metadata key {key} is not a list.")

        self._metadata[key].append(value)
        return self

    def clear_metadata(self) -> NDImage:
        """Clear metadata of image."""
        self.metadata = {}
        return self

    # ==========================================================================
    # Grids
    # ==========================================================================
    def vals(self, dim) -> np.ndarray:
        """Returns the coordinate values along the given dimension."""
        dim_limits = self.limits[dim]
        return np.linspace(dim_limits.min, dim_limits.max, self.shape[dim])

    @property
    def grid(self) -> np.ndarray:
        """Returns the coordinate grid."""
        return np.stack(
            np.meshgrid(*[self.vals(dim) for dim in range(self.ndim)], indexing="ij"),
            axis=-1,
        )

    @property
    def flatgrid(self) -> np.ndarray:
        """Returns the flattened coordinate grid."""
        return self.grid.reshape(-1, self.ndim)

    def __getitem__(self, key) -> NDImage:
        """Slicing the image."""
        new_array = self.array[key]
        new_limits = compute_limits_after_slicing(self.shape, self.limits, key)
        new_labels, new_units = self._axis_metadata_after_slicing(key)
        return NDImage(
            new_array,
            new_limits,
            metadata=self.metadata,
            labels=new_labels,
            units=new_units,
        )

    def _axis_metadata_after_slicing(self, key) -> tuple:
        """Propagate per-axis labels and units through a numpy-style index key."""
        labels = select_axis_values_after_slicing(self._labels, key, "")
        units = select_axis_values_after_slicing(self._units, key, "")
        return labels, units

    def __setitem__(self, key, value) -> None:
        self.array[key] = value

    # ==========================================================================
    # Functions
    # ==========================================================================
    def save(
        self,
        path,
        group="/",
        append=False,
    ) -> NDImage:
        """Save image to HDF5 file.

        The image data is saved to a dataset called 'image' in the given group.
        The metadata is saved in the same group.

        Args:
            path: Path to the HDF5 file.
            group: Group within the HDF5 file. Default is root group.
            append: If True, append to an existing file. If False, overwrite the file.

        Returns:
            self: The NDImage instance (for chaining).
        """
        path = Path(path)
        assert path.suffix == ".hdf5", "File must be HDF5 format."

        save_hdf5_image(
            path=path,
            array=self.array,
            limits=self.limits,
            metadata=self.metadata,
            labels=self.labels,
            units=self.units,
            group=group,
            append=append,
        )
        return self

    def save_png(
        self,
        path,
        cmap="gray",
        vmin=None,
        vmax=None,
    ) -> NDImage:
        """Save image to PNG file."""
        path = Path(path)
        assert path.suffix == ".png", "File must be PNG format."
        matplotlib.image.imsave(path, self.array.T, cmap=cmap, vmin=vmin, vmax=vmax)
        return self

    @classmethod
    def load(cls, path, indices=slice(None), group="/") -> NDImage:
        """Load image from HDF5 file."""
        path = Path(path)
        assert path.suffix == ".hdf5", "File must be HDF5 format."
        return load_hdf5_image(path, indices=indices, group=group)

    def _rewrap(self, array, limits: LimitsNDInput | None = None) -> NDImage:
        """Create a new image carrying this image's metadata, labels and units.

        For operations that keep the number and order of dimensions. `limits`
        defaults to the current limits.
        """
        return NDImage(
            array,
            limits=self.limits if limits is None else limits,
            metadata=self.metadata,
            labels=self.labels,
            units=self.units,
        )

    def with_array(self, array) -> NDImage:
        return self._rewrap(array)

    def with_limits(self, limits: LimitsNDInput | None) -> NDImage:
        return self._rewrap(self.array, limits)

    def map_range(self, new_min, new_max, old_min=None, old_max=None) -> NDImage:
        """Map the image values to a new range [new_min, new_max]."""
        if old_min is None:
            old_min = np.min(self.array)
        if old_max is None:
            old_max = np.max(self.array)
        scaled = (self.array - old_min) / (old_max - old_min)
        mapped = scaled * (new_max - new_min) + new_min
        return self.with_array(mapped)

    def to_pixels(self) -> NDImage:
        """Convert the image to pixel values in the range [0, 1]."""
        return self.map_range(0, 1)

    def clip(self, min=None, max=None) -> NDImage:
        """Clip the image values to the given range."""
        return self.with_array(np.clip(self.array, min, max))

    def resample(
        self, shape, limits: LimitsNDInput | None = None, method="linear", fill_value=0
    ) -> NDImage:
        """Resample image to a new shape."""
        limits = LimitsND(limits) if limits is not None else self.limits

        all_vals = [self.vals(dim) for dim in range(self.ndim)]

        interpolator = RegularGridInterpolator(
            all_vals,
            self.array,
            bounds_error=False,
            fill_value=fill_value,
            method=method,
        )
        new_all_vals = [
            np.linspace(limits[dim].min, limits[dim].max, shape[dim])
            for dim in range(len(shape))
        ]

        new_data = interpolator(np.meshgrid(*new_all_vals, indexing="ij")).reshape(
            shape
        )

        return self._rewrap(new_data, limits)

    def transpose(self, axes=None) -> NDImage:
        """Transpose the image."""
        if axes is None:
            axes = list(reversed(range(self.ndim)))
        new_array = np.transpose(self.array, axes)
        new_limits = LimitsND([self.limits[axis] for axis in axes])
        new_labels = tuple(self._labels[axis] for axis in axes)
        new_units = tuple(self._units[axis] for axis in axes)
        return NDImage(
            new_array,
            limits=new_limits,
            metadata=self.metadata,
            labels=new_labels,
            units=new_units,
        )

    def flip(self, dim) -> NDImage:
        """Returns a copy of the image flipped along the given dimension."""
        key = [slice(None)] * self.ndim
        key[dim] = slice(None, None, -1)
        return self[tuple(key)]

    def square_pixels(self) -> NDImage:
        """Resample so that all dimensions share the smallest pixel size."""
        new_pixel_size = min(size for size in self.pixel_sizes if size > 0)
        new_shape, new_limits = self._square_pixel_geometry(new_pixel_size)
        return self.resample(shape=new_shape, limits=new_limits, method="nearest")

    def _square_pixel_geometry(self, pixel_size) -> tuple[list, LimitsND]:
        new_shape = []
        new_limits = []
        for dim in range(self.ndim):
            dim_limits = self.limits[dim]
            n_pixels = int(dim_limits.size() / pixel_size) + 1
            new_shape.append(n_pixels)
            new_limits.append(
                Limits(dim_limits.min, dim_limits.min + (n_pixels - 1) * pixel_size)
            )
        return new_shape, LimitsND(new_limits)

    def resample_scale(self, factor, axes=None) -> NDImage:
        """Scale the image by a given factor."""
        factors = [
            factor if (axes is None or dim in axes) else 1 for dim in range(self.ndim)
        ]
        new_shape = [
            max(1, int(dim_size * factor_for_dim))
            for dim_size, factor_for_dim in zip(self.shape, factors)
        ]
        new_limits = _collapse_limits_for_single_element_dims(self.limits, new_shape)
        return self.resample(shape=new_shape, limits=new_limits, method="linear")

    def get_window(self, limits) -> NDImage:
        """Returns a new image restricted to a physical-coordinate window.

        Args:
            limits: One entry per dimension, each either a (min, max) pair
                (in image units, not pixel indices) or None to leave that
                dimension unchanged.
        """
        assert len(limits) == self.ndim, "limits must have one entry per dimension."
        slices = tuple(
            self._window_slice(dim, entry) for dim, entry in enumerate(limits)
        )
        return self[slices]

    def _window_slice(self, dim, entry) -> slice:
        if entry is None:
            return slice(None)
        low, high = entry
        return slice(
            self._coordinate_to_clipped_index(dim, low),
            self._coordinate_to_clipped_index(dim, high),
        )

    def _coordinate_to_clipped_index(self, dim, coordinate) -> int:
        dim_limits = self.limits[dim]
        size = dim_limits.size() or 1
        fraction = (coordinate - dim_limits.min) / size
        index = int(np.ceil(fraction * (self.shape[dim] - 1)))
        return int(np.clip(index, 0, self.shape[dim]))

    def match_histogram(self, other) -> NDImage:
        """Match the histogram of the image to another image."""

        array = match_histograms(self.array, other.array)
        return self.with_array(array)

    def apply_dynamic_range_curve(self, curve: np.ndarray) -> NDImage:
        """Apply a dynamic range curve to the image data."""
        data = apply_dynamic_range_curve(curve, self.array)
        return self.with_array(data)

    def log_compress(self) -> NDImage:
        """Log-compress image data with 20*log10(image)."""

        data = np.where(self.array > 0, self.array, 1e-12)
        data = 20 * np.log10(data)

        return self.with_array(data)

    def log_expand(self) -> NDImage:
        """Log-expand image data."""

        array = np.power(10, self.array / 20)
        array = np.where(self.array <= -240, 0, array)

        return self.with_array(array)

    def symlog_compress(self, threshold=1.0) -> NDImage:
        """Symmetric log-compress image data."""

        data = np.where(
            np.abs(self.array) > threshold,
            np.sign(self.array)
            * (threshold + np.log10(np.abs(self.array) / threshold)),
            self.array,
        )

        return self.with_array(data)

    def symlog_expand(self, threshold=1.0) -> NDImage:
        """Symmetric log-expand image data."""

        array = np.where(
            np.abs(self.array) > threshold,
            np.sign(self.array)
            * threshold
            * np.power(10, (np.abs(self.array) - threshold) / threshold),
            self.array,
        )

        return self.with_array(array)

    def abs(self) -> NDImage:
        """Returns the absolute value of the image."""
        return self.with_array(np.abs(self.array))

    def normalize(self, normval=None) -> NDImage:
        """Normalize image data by dividing by the max or normval."""

        if normval is None:
            normval = self.array.max()

        if normval == 0.0:
            warnings.warn("Warning: normval is 0. Returning original image.")
            return self

        return self / normval

    def normalize_db(self, normval=None) -> NDImage:
        """Normalize image data by adding the max or normval."""
        if normval is None:
            normval = self.array.max()

        return self - np.array(normval)

    def normalize_percentile(self, percentile=99) -> NDImage:
        """Normalize image data to the given percentile value."""
        normval = np.percentile(self.array, percentile)
        return self.normalize(normval)

    def copy(self) -> NDImage:
        """Returns a copy of the image."""
        return self._rewrap(self.array.copy())

    def max(self, **kwargs) -> float:
        """Returns the maximum value of the image."""
        return np.max(self.array, **kwargs)

    def min(self, **kwargs) -> float:
        """Returns the minimum value of the image."""
        return np.min(self.array, **kwargs)

    def mean(self, **kwargs) -> float:
        """Returns the mean value of the image."""
        return np.mean(self.array, **kwargs)

    def fft(self, axes=None) -> NDImage:
        """Returns the FFT of the image.

        The spectrum is shifted so that the zero frequency component is in the center
        of the spectrum.

        The limits are updated to reflect the spatial frequency range."""
        data = np.fft.fftn(self.array, axes=axes)

        new_limits = list(self.limits.limits)

        for axis in axes or range(self.ndim):
            data = np.fft.fftshift(data, axes=axis)
            spatial_sampling_interval = self.pixel_size(axis)
            spatial_freqs = np.fft.fftfreq(
                self.shape[axis], d=spatial_sampling_interval
            )
            new_limits[axis] = Limits(np.min(spatial_freqs), np.max(spatial_freqs))

        return self._rewrap(data, LimitsND(new_limits))

    def moving_average(self, ax, window_size) -> NDImage:
        """Apply a moving average filter along the given axis."""
        smoothed = uniform_filter1d(
            self.array, size=window_size, axis=ax, mode="constant", cval=0
        )
        return self.with_array(smoothed)

    def normalize_moving_average(self, ax, window_size, eps=1e-6) -> NDImage:
        """Computes the moving average along the given axis and normalizes the image by dividing by the moving average."""

        moving_avg = np.abs(self.moving_average(ax, window_size).array) + eps

        all_axes = tuple(set(range(self.ndim)) - set([ax % self.ndim]))
        moving_avg = np.mean(moving_avg, axis=all_axes)
        dummy_dim_tuple = [None] * self.ndim
        dummy_dim_tuple[ax] = slice(None)
        moving_avg = moving_avg[tuple(dummy_dim_tuple)]
        normalized = np.where(moving_avg > 0, self.array / moving_avg, 0)
        return self.with_array(normalized)

    def sample(self, positions) -> np.ndarray:
        """Sample image values at the given spatial positions without interpolation.

        Args:
            positions: np.ndarray of shape (N, D) Spatial coordinates to sample. D
            must match the image dimensionality.

        Returns:
            values: Image values at the nearest pixel for each position of shape (N,) .
        """
        indices = self.coordinates_to_indices(positions)
        return self.array[tuple(indices[:, dim] for dim in range(self.ndim))]

    def coordinates_to_indices(self, coordinates) -> np.ndarray:
        """Convert coordinates to pixel indices."""
        assert coordinates.ndim == 2
        assert coordinates.shape[1] == self.ndim
        indices_total = []
        for dim in range(self.ndim):
            indices = (coordinates[:, dim] - self.limits[dim].min) / self.pixel_size(
                dim
            )
            indices_rounded = np.round(indices).astype(int)
            indices_rounded = np.clip(indices_rounded, 0, self.shape[dim] - 1)
            indices_total.append(indices_rounded)
        return np.stack(indices_total, axis=-1)

    def indices_to_coordinates(self, indices) -> np.ndarray:
        """Convert pixel indices to physical coordinates.

        Args:
            indices: Pixel indices (integer or float) to convert. D must match the image
            dimensionality. Float indices yield sub-pixel coordinates.

        Returns:
            coordinates: Physical coordinates corresponding to each index of shape (N, D).
        """
        indices = np.asarray(indices)
        assert indices.ndim == 2
        assert indices.shape[1] == self.ndim
        coords = []
        for dim in range(self.ndim):
            coords.append(self.limits[dim].min + indices[:, dim] * self.pixel_size(dim))
        return np.stack(coords, axis=-1)

    def clahe(
        self,
        clip_limit: float = 0.01,
        tile_grid_size: tuple = (8, 8),
        axes: tuple = (0, 1),
    ) -> NDImage:
        """Apply CLAHE to the image."""
        if self.ndim < 2:
            raise ValueError("CLAHE requires at least 2D images.")
        data = clahe(self.array, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        return self.with_array(data)

    # ==========================================================================
    # Dunder methods
    # ==========================================================================

    def __add__(self, other) -> NDImage:
        """Add two images together."""
        if isinstance(other, (int, float, np.number)):
            return self.with_array(self.array + other)

        if isinstance(other, NDImage):
            assert self.limits == other.limits
            return self.with_array(self.array + other.array)

        return self.with_array(self.array + np.array(other))

    def __mul__(self, other) -> NDImage:
        """Multiply image."""
        if isinstance(other, NDImage):
            other = other.array

        return self.with_array(self.array * other)

    def __rmul__(self, other) -> NDImage:
        """Multiply image."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Divide image."""
        if isinstance(other, NDImage):
            other = other.array

        return self.with_array(self.array / other)

    def __sub__(self, other) -> NDImage:
        """Subtract two images."""
        return self + (other * -1)

    def __eq__(self, other) -> bool:
        """Check if two images are equal."""
        if not isinstance(other, NDImage):
            return False

        if self.limits != other.limits:
            return False

        return np.allclose(self.array, other.array)

    @classmethod
    def test_image(cls) -> NDImage:
        """Returns a test image."""

        limits = LimitsND([(-10, 0), (0, 20)])
        dim0_vals = np.linspace(limits[0].min, limits[0].max, 128)
        dim1_vals = np.linspace(limits[1].min, limits[1].max, 256)
        grid0, grid1 = np.meshgrid(dim0_vals, dim1_vals, indexing="ij")
        array = (
            np.exp(-((grid0 + 5) ** 2 + (grid1 - 10) ** 2) / 20)
            * np.sin(2 * np.pi * grid0)
            * np.cos(2 * np.pi * grid1 * 2)
        )
        return NDImage(array, limits=limits)

    @classmethod
    def from_png(cls, path, limits: LimitsNDInput | None = None) -> NDImage:
        """Load image from PNG file."""
        array = np.mean(matplotlib.image.imread(path), axis=2).T  # convert to grayscale
        return NDImage(array, limits=limits)


# ==============================================================================
# Helper functions
# ==============================================================================
def _collapse_limits_for_single_element_dims(limits: LimitsND, shape) -> LimitsND:
    new_limits = []
    for dim, dim_limits in enumerate(limits):
        if shape[dim] == 1:
            center = (dim_limits.min + dim_limits.max) / 2
            new_limits.append(Limits(center, center))
        else:
            new_limits.append(dim_limits)
    return LimitsND(new_limits)


def _check_ndimage_initializers(array: np.ndarray, limits: LimitsND):
    assert array.ndim == limits.ndim, (
        "The array and limits must have the same number of dimensions. "
        f"Got {array.ndim} and {limits.ndim}."
    )
    for dim in range(array.ndim):
        if array.shape[dim] == 1 and not limits[dim].size() == 0.0:
            raise ValueError(
                "Dimensions with one element must have zero size in the limits"
            )


def get_limits_imshow(limits: LimitsND, shape: tuple) -> tuple:
    """Compute the (x0, x1, y0, y1) extent for matplotlib's imshow.

    Uses the last two dimensions of `limits`/`shape`, the (y, x) image-plane
    axes under the zyx convention.
    """
    y_edges = _padded_edges(limits[-2], shape[-2])
    x_edges = _padded_edges(limits[-1], shape[-1])
    return (*x_edges, *y_edges)


def _padded_edges(dim_limits: Limits, n_pixels: int) -> tuple:
    """Return (min, max) padded by half a pixel on each side."""
    pixel_size = dim_limits.size() / (n_pixels - 1) if n_pixels > 1 else 0.0
    return dim_limits.min - pixel_size / 2, dim_limits.max + pixel_size / 2


def compute_pixel_sizes(limits: LimitsND, shape) -> np.ndarray:
    pixel_sizes = []
    for dim in range(limits.ndim):
        pixel_size = limits[dim].size() / (shape[dim] - 1) if shape[dim] > 1 else 0.0
        pixel_sizes.append(pixel_size)
    return np.array(pixel_sizes)


def stack(
    images: list[NDImage], axis: int = 0, limits: Limits | tuple | None = None
) -> NDImage:
    """Stack a list of NDImage objects along a new axis.

    Args:
        images: List of NDImage objects to stack.
        axis: Axis along which to stack the images. Default is 0.
        limits: Optional LimitsND object for the new stacked image. If None, the limits
            will be set to (0, len(images)-1).

    Returns:
        NDImage: A new NDImage object representing the stacked images.
    """
    arrays = [img.array for img in images]
    stacked_array = np.stack(arrays, axis=axis)

    if limits is None:
        limits = Limits(0, len(images) - 1)
    else:
        limits = Limits(*limits)

    limits_list = []
    index_in_dims = 0
    for dim in range(stacked_array.ndim):
        if dim == axis:
            limits_list.append(limits)
        else:
            limits_list.append(images[0].limits[index_in_dims])
            index_in_dims += 1

    return NDImage(stacked_array, limits=LimitsND(limits_list))
