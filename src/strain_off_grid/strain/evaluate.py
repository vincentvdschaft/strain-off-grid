from pathlib import Path

import numpy as np
from imagelib import Image, Limits, LimitsND, stack
from scipy.spatial import KDTree

from strain_off_grid import console
from strain_off_grid.phantoms import DynamicPhantom, load_dataclass
from strain_off_grid.phantoms.block_matching import (
    get_baseline_tracking_velocities,
)
from strain_off_grid.strain.strain_rate_map import compute_strain_rate_map
from strain_off_grid.velocities import Velocities2D, Velocities3D

PRINCIPAL_AXIS = 0
OTHOGONAL_AXIS = 1


def compute_strain_maps_multi(
    paths: list[str | Path],
    map_pixel_size: float = 5e-4,
    map_limits_yx: LimitsND = LimitsND((30e-3, 100e-3, -20e-3, 20e-3)),
    direction=PRINCIPAL_AXIS,
    kernel_size: float = 2e-3,
) -> tuple[Image, Image, Image]:
    maps_solver = []
    maps_ground_truth = []
    maps_baseline = []
    for path in paths:
        map_solver, map_ground_truth, map_baseline = solution_to_strain_rate_map(
            path,
            map_limits_yx=map_limits_yx,
            map_pixel_size=map_pixel_size,
            direction=direction,
        )
        maps_solver.append(map_solver)
        maps_ground_truth.append(map_ground_truth)
        maps_baseline.append(map_baseline)

    t0 = maps_solver[0].metadata["timestamp"]
    t1 = maps_solver[-1].metadata["timestamp"]
    limits = Limits(t0, t1)

    return (
        stack(maps_solver, limits=limits),
        stack(maps_ground_truth, limits=limits),
        stack(maps_baseline, limits=limits),
    )


def solution_to_strain_rate_map(
    path: str | Path,
    map_pixel_size: float = 5e-4,
    map_limits_yx: LimitsND = LimitsND((30e-3, 100e-3, -20e-3, 20e-3)),
    direction: np.ndarray | int = PRINCIPAL_AXIS,
) -> tuple[Image, Image, Image]:
    image, phantom = load_image_and_phantom(path)
    map_limits_yx = LimitsND(map_limits_yx).fitted_to_pixel_sizes(map_pixel_size)
    target_grid = map_limits_yx.make_grid(pixel_sizes=map_pixel_size)[..., ::-1]

    velocities_2d_solver = _get_solver_velocities(image=image)

    velocities_2d_true = _get_ground_truth_velocities(
        phantom=phantom, timestamp=velocities_2d_solver.timestamp
    )

    velocities_2d_baseline = get_baseline_tracking_velocities(
        image0=image[0],
        image1=image[-1],
        positions0=velocities_2d_solver.positions,
        dt=image.metadata["timestamps"].ravel()[-1]
        - image.metadata["timestamps"].ravel()[0],
        block_size=(21, 21),
    )
    direction = _get_directions(
        phantom=phantom,
        positions=target_grid,
        direction=direction,
    ).reshape(-1, 2)
    strain_rate_map = compute_strain_rate_map(
        velocities=velocities_2d_solver,
        directions=direction,
        target_grid=target_grid,
        max_distance=5e-3,
        kernel_size=5e-3,
    )
    strain_rate_map_ground_truth = compute_strain_rate_map(
        velocities=velocities_2d_true,
        directions=direction,
        target_grid=target_grid,
        max_distance=5e-3,
        kernel_size=5e-3,
    )
    strain_rate_map_baseline = compute_strain_rate_map(
        velocities=velocities_2d_baseline,
        directions=direction,
        target_grid=target_grid,
        max_distance=5e-3,
        kernel_size=5e-3,
    )

    mask = ~phantom.points_in_phantom(target_grid, t=velocities_2d_solver.timestamp)

    strain_rate_map[mask] = np.nan
    strain_rate_map_ground_truth[mask] = np.nan
    strain_rate_map_baseline[mask] = np.nan

    return (
        Image(
            strain_rate_map,
            limits=map_limits_yx,
            metadata={"timestamp": velocities_2d_solver.timestamp},
        ),
        Image(
            strain_rate_map_ground_truth,
            limits=map_limits_yx,
            metadata={"timestamp": velocities_2d_true.timestamp},
        ),
        Image(
            strain_rate_map_baseline,
            limits=map_limits_yx,
            metadata={"timestamp": velocities_2d_baseline.timestamp},
        ),
    )


def _get_directions(
    phantom: DynamicPhantom, positions: np.ndarray, direction: np.ndarray | None
) -> np.ndarray:
    """Computes the directions of the phantom at the given positions."""
    if isinstance(direction, np.ndarray):
        return direction

    is_orthogonal = direction == OTHOGONAL_AXIS
    direction = phantom.principal_axis(positions)
    if is_orthogonal:
        direction = np.stack([-direction[..., 1], direction[..., 0]], axis=-1)
    return direction


def _get_ground_truth_velocities(
    phantom: DynamicPhantom, timestamp: float
) -> Velocities2D:
    positions = phantom.sample_points(500)
    velocities_3d_true = phantom.get_velocities(positions, t=timestamp, dt=1e-6)
    velocities_2d_true = velocities_3d_true.to_2d()
    return velocities_2d_true


def _read_timestamps(paths):
    timestamps = []
    for path in paths:
        image = Image.load(path)
        print(f"Read timestamps from {path}: {image.metadata['timestamps'].ravel()}")
        timestamps.append(image.metadata["timestamps"].ravel()[0])
    return np.array(timestamps)


def load_image_and_phantom(path: str | Path) -> tuple[Image, DynamicPhantom]:
    image = Image.load(path).abs().normalize().log_compress().clip(-60, 0).to_pixels()
    phantom: DynamicPhantom = load_dataclass(str(path), group="/custom/phantom")
    return image, phantom


def stack_maps(maps: list[Image]) -> Image:
    arrays = [m.array for m in maps]
    stacked_array = np.stack(arrays, axis=0)
    return Image(
        stacked_array,
        limits=LimitsND([Limits(0, len(maps) - 1), *maps[0].limits]),
    )


def strain_rate_make_grid_compute_map(
    velocities: Velocities2D,
    directions: np.ndarray,
    map_shape: tuple[int, int],
    map_limits: LimitsND,
    phantom: DynamicPhantom,
    max_distance: float = 5e-3,
    kernel_size: float = 2e-3,
    t: float = 0.0,
) -> Image:
    """Computes the strain rate map from the given velocities.

    Args:
        velocities: The velocities to compute the strain rate map from.
        directions: The directions to compute the strain rate map in.
        map_shape: The shape of the map to compute.
        map_limits: The limits of the map to compute (ylims, xlims).
        phantom: The phantom to compute the strain rate map in.
        max_distance: The maximum distance to consider for each point in the target grid.
        t: The time at which to compute the strain rate map.

    Returns:
        The strain rate map as a 1D array of shape (n_points,).
    """

    target_grid = map_limits.make_grid(map_shape).reshape(-1, map_limits.ndim)[
        ..., ::-1
    ]

    strain_rate_map = compute_strain_rate_map(
        velocities=velocities,
        directions=np.array([0.0, 1.0]),
        target_grid=target_grid,
        max_distance=max_distance,
        kernel_size=kernel_size,
    )
    strain_rate_map[
        ~phantom.points_in_phantom(target_grid.reshape(-1, target_grid.shape[-1]), t=t)
    ] = np.nan

    return Image(
        strain_rate_map.reshape(map_shape), limits=map_limits, metadata={"t": t}
    )


def _get_solver_velocities(image: Image) -> Velocities2D:
    """
    Computes the velocities from the solver's output. The velocity is computed from frame 0 to frame -1.
    The timestamp is set to the timestamp of frame 0.


    """
    positions_all_frames = image.metadata["scat_pos"]
    timestamps = image.metadata["timestamps"].ravel()
    dt = timestamps[-1] - timestamps[0]
    positions_frame0 = positions_all_frames[:, 0]
    velocities_2d_solver = Velocities3D(
        positions=positions_frame0,
        velocities=np.diff(positions_all_frames[:, [0, -1]], axis=1)[:, 0] / dt,
        timestamp=timestamps[0],
    ).to_2d()

    intensities = image.metadata["scat_amp"][:, 0]
    intensities = intensities / np.max(intensities)
    velocities_2d_solver = _filter_solver_velocities(
        velocities_2d_solver,
        intensities=intensities,
        min_intensity=0.1,
        max_velocity_magnitude=0.05,
    )
    console.log("Filtered solver velocities: %d points" % len(velocities_2d_solver))

    return velocities_2d_solver


def _filter_solver_velocities(
    velocities: Velocities2D,
    intensities: np.ndarray,
    min_intensity: float = 0.1,
    max_velocity_magnitude: float = 0.1,
) -> Velocities2D:
    if intensities is not None:
        mask_intensity = intensities > min_intensity
    else:
        mask_intensity = np.ones(velocities.positions.shape[0], dtype=bool)

    velocity_magnitudes = np.linalg.norm(velocities.velocities, axis=1)
    mask_velocity = velocity_magnitudes < max_velocity_magnitude

    mask = mask_intensity & mask_velocity
    return velocities[mask]


def _filter_solver_velocities_harmonic(
    velocities: Velocities2D,
    intensities: np.ndarray,
    neighborhood_radius: float = 6e-3,
    min_relative_intensity: float = 0.1,
    max_velocity_magnitude: float = 0.1,
) -> Velocities2D:
    """Filters solver velocities using a spatially-local intensity threshold.

    Harmonic imaging intensity can vary strongly over the field of view, so a
    single global intensity threshold (as in `_filter_solver_velocities`) either
    discards valid points in dim regions or keeps noise in bright ones. Instead,
    a point is kept only if its intensity is at least `min_relative_intensity`
    times the maximum intensity found within `neighborhood_radius` of it.
    """
    if intensities is not None:
        local_max_intensity = _find_max_in_neighborhood(
            velocities.positions, intensities, radius=neighborhood_radius
        )
        mask_intensity = intensities > min_relative_intensity * local_max_intensity
    else:
        mask_intensity = np.ones(velocities.positions.shape[0], dtype=bool)

    velocity_magnitudes = np.linalg.norm(velocities.velocities, axis=1)
    mask_velocity = velocity_magnitudes < max_velocity_magnitude

    mask = mask_intensity & mask_velocity
    return velocities[mask]


def _find_max_in_neighborhood(
    positions: np.ndarray, intensities: np.ndarray, radius: float
) -> np.ndarray:
    """For each point, finds the max intensity among points within `radius` of it."""
    tree = KDTree(positions)
    neighbor_indices = tree.query_ball_point(positions, r=radius)
    return np.array([intensities[indices].max() for indices in neighbor_indices])
