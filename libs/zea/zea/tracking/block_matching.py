"""Block matching tracker.

Tracks points by searching, for each point, the displacement within a search
window that best matches a reference block extracted from the previous frame.
Supports 2D and 3D data, multiple matching metrics, and subpixel refinement.
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
from keras import ops

from zea.func.tensor import translate

from .base import BaseTracker


class BlockMatchingTracker(BaseTracker):
    """Block matching tracker.

    For each point a reference block is extracted from the previous frame and
    compared against candidate blocks in the next frame over an exhaustive
    search window. The best matching displacement is selected and optionally
    refined to subpixel accuracy with a parabolic fit.

    Available metrics:
    - ``"ssd"``: Sum of squared differences (default).
    - ``"sad"``: Sum of absolute differences.
    - ``"ncc"``: Normalized cross-correlation (higher is better, so the negative is minimized).

    Args:
        block_size: Block size (height, width) for 2D or (depth, height, width) for 3D.
        search_range: Maximum displacement searched per dimension. An int is used
            for all dimensions, or a tuple matching ``block_size``.
        metric: Matching cost, one of ``"ssd"``, ``"sad"`` or ``"ncc"``.
        subpixel: Whether to refine the best match to subpixel accuracy.
        extent: Physical extent of the image as ``(x0, x1, y0, y1)`` for 2D or
            ``(x0, x1, y0, y1, z0, z1)`` for 3D, where each pair gives the
            coordinates of the centers of the outer pixels along that axis. When
            provided, ``points`` are interpreted and returned in physical
            coordinates; otherwise they are in pixel coordinates.
        **kwargs: Additional parameters.

    Example:
        .. doctest::

            >>> from zea.tracking import BlockMatchingTracker
            >>> import numpy as np

            >>> tracker = BlockMatchingTracker(block_size=(21, 21), search_range=10)
            >>> frame1 = np.random.rand(100, 100).astype("float32")
            >>> frame2 = np.random.rand(100, 100).astype("float32")
            >>> points = np.array([[50.5, 55.2], [60.1, 65.8]], dtype="float32")
            >>> new_points = tracker.track(frame1, frame2, points)
            >>> new_points.shape
            (2, 2)
    """

    def __init__(
        self,
        block_size: Tuple[int, ...] = (21, 21),
        search_range: Union[int, Tuple[int, ...]] = 10,
        metric: str = "ssd",
        subpixel: bool = True,
        extent: Optional[Sequence[float]] = None,
        **kwargs,
    ):
        """Initialize the block matching tracker."""
        self.ndim = len(block_size)
        super().__init__(ndim=self.ndim, **kwargs)

        if metric not in ("ssd", "sad", "ncc"):
            raise ValueError(f"Unknown metric '{metric}', expected 'ssd', 'sad' or 'ncc'")

        self.block_size = block_size
        self.metric = metric
        self.subpixel = subpixel
        self.half_block = tuple(size // 2 for size in block_size)
        self.search_range = self._prepare_search_range(search_range)
        self.search_shape = tuple(2 * radius + 1 for radius in self.search_range)
        self.extent = extent
        self.extent_pairs = self._parse_extent(extent)

    def _parse_extent(
        self, extent: Optional[Sequence[float]]
    ) -> Optional[Tuple[Tuple[float, float], ...]]:
        """Group the flat extent into a (low, high) pair per point dimension."""
        if extent is None:
            return None
        if len(extent) != 2 * self.ndim:
            raise ValueError(f"extent must have {2 * self.ndim} values, got {len(extent)}")
        spatial_pairs = [(extent[2 * i], extent[2 * i + 1]) for i in range(self.ndim)]
        return tuple(reversed(spatial_pairs))

    def _prepare_search_range(self, search_range) -> Tuple[int, ...]:
        """Expand a scalar search range to one radius per dimension."""
        if isinstance(search_range, int):
            return (search_range,) * self.ndim
        return tuple(search_range)

    def track(self, prev_frame, next_frame, points):
        """Track points from prev_frame to next_frame.

        Args:
            prev_frame: Previous frame/volume, shape (H, W) for 2D or (D, H, W) for 3D.
            next_frame: Next frame/volume, shape (H, W) for 2D or (D, H, W) for 3D.
            points: Points to track, shape (N, ndim) in (x, y) or (x, y, z) format.
                In pixel coordinates, or physical coordinates when ``extent`` is set.

        Returns:
            new_points: Tracked points as tensor, shape (N, ndim).
        """
        prev_norm = translate(prev_frame, range_to=(0, 1))
        next_norm = translate(next_frame, range_to=(0, 1))
        frame_shape = prev_norm.shape

        axis_order_points = ops.flip(points, axis=-1)
        pixel_points = self._physical_to_pixels(axis_order_points, frame_shape)
        n_points = int(points.shape[0])
        tracked = [
            self._track_point(prev_norm, next_norm, pixel_points[i]) for i in range(n_points)
        ]
        axis_order_tracked = self._pixels_to_physical(ops.stack(tracked), frame_shape)
        return ops.flip(axis_order_tracked, axis=-1)

    def _pixel_scales(self, frame_shape):
        """Physical size of one pixel along each point dimension."""
        pairs_and_sizes = zip(self.extent_pairs, frame_shape)
        scales = [(high - low) / (size - 1) for (low, high), size in pairs_and_sizes]
        return ops.convert_to_tensor(scales, dtype="float32")

    def _pixel_origins(self):
        """Physical coordinate of the first pixel along each point dimension."""
        return ops.convert_to_tensor([low for low, _ in self.extent_pairs], dtype="float32")

    def _physical_to_pixels(self, points, frame_shape):
        """Convert physical coordinates to pixel coordinates if an extent is set."""
        if self.extent_pairs is None:
            return points
        return (points - self._pixel_origins()) / self._pixel_scales(frame_shape)

    def _pixels_to_physical(self, points, frame_shape):
        """Convert pixel coordinates back to physical coordinates if an extent is set."""
        if self.extent_pairs is None:
            return points
        return points * self._pixel_scales(frame_shape) + self._pixel_origins()

    def _track_point(self, prev_frame, next_frame, point):
        """Find the best matching displacement for a single point."""
        template = self._sample_blocks(prev_frame, ops.reshape(point, (1, self.ndim)))[0]
        offsets = self._candidate_offsets()
        candidate_centers = ops.expand_dims(point, 0) + offsets
        blocks = self._sample_blocks(next_frame, candidate_centers)

        costs = self._match_costs(template, blocks)
        best = int(ops.argmin(costs))

        displacement = offsets[best]
        if self.subpixel:
            displacement = displacement + self._subpixel_offset(costs, best)
        return point + displacement

    def _candidate_offsets(self):
        """Integer search offsets, shape (num_candidates, ndim)."""
        axes = [ops.arange(-radius, radius + 1, dtype="float32") for radius in self.search_range]
        grids = ops.meshgrid(*axes, indexing="ij")
        return ops.stack([ops.reshape(grid, [-1]) for grid in grids], axis=1)

    def _block_grid(self):
        """Within-block coordinate offsets, shape (ndim, *block_size)."""
        axes = [ops.arange(2 * half + 1, dtype="float32") - half for half in self.half_block]
        grids = ops.meshgrid(*axes, indexing="ij")
        return ops.stack(grids, axis=0)

    def _sample_blocks(self, image, centers):
        """Sample a block around each center, shape (num_centers, *block_size)."""
        block_grid = ops.expand_dims(self._block_grid(), 1)
        centers_per_dim = ops.transpose(centers)
        centers_per_dim = ops.reshape(centers_per_dim, (self.ndim, -1) + (1,) * self.ndim)
        coords = block_grid + centers_per_dim
        return ops.image.map_coordinates(image, coords, order=1, fill_mode="constant")

    def _match_costs(self, template, blocks):
        """Matching cost per candidate block (lower is better)."""
        block_axes = tuple(range(1, self.ndim + 1))
        if self.metric == "sad":
            return ops.sum(ops.abs(blocks - template), axis=block_axes)
        if self.metric == "ncc":
            return -self._normalized_cross_correlation(template, blocks, block_axes)
        return ops.sum(ops.square(blocks - template), axis=block_axes)

    def _normalized_cross_correlation(self, template, blocks, block_axes):
        """Normalized cross-correlation between the template and each block."""
        centered_template = template - ops.mean(template)
        centered_blocks = blocks - ops.mean(blocks, axis=block_axes, keepdims=True)
        numerator = ops.sum(centered_template * centered_blocks, axis=block_axes)
        template_energy = ops.sum(ops.square(centered_template))
        block_energy = ops.sum(ops.square(centered_blocks), axis=block_axes)
        return numerator / (ops.sqrt(template_energy * block_energy) + 1e-8)

    def _subpixel_offset(self, costs, best):
        """Parabolic subpixel refinement of the best displacement."""
        cost_grid = np.asarray(ops.convert_to_numpy(ops.reshape(costs, self.search_shape)))
        best_index = np.unravel_index(best, self.search_shape)
        deltas = [self._parabolic_delta(cost_grid, best_index, axis) for axis in range(self.ndim)]
        return ops.convert_to_tensor(deltas, dtype="float32")

    def _parabolic_delta(self, cost_grid, best_index, axis):
        """Subpixel shift along one axis from a parabola through three samples."""
        position = best_index[axis]
        if position <= 0 or position >= self.search_shape[axis] - 1:
            return 0.0
        center = float(cost_grid[best_index])
        lower = float(cost_grid[self._neighbor_index(best_index, axis, -1)])
        upper = float(cost_grid[self._neighbor_index(best_index, axis, 1)])
        denominator = lower - 2 * center + upper
        if abs(denominator) < 1e-8:
            return 0.0
        return 0.5 * (lower - upper) / denominator

    def _neighbor_index(self, best_index, axis, step):
        """Index tuple shifted by step along one axis."""
        neighbor = list(best_index)
        neighbor[axis] += step
        return tuple(neighbor)

    def __repr__(self):
        """String representation."""
        return (
            f"BlockMatchingTracker(block_size={self.block_size}, "
            f"search_range={self.search_range}, metric='{self.metric}', "
            f"subpixel={self.subpixel}, extent={self.extent})"
        )
