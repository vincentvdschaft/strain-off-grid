import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from strain_off_grid.velocities import Velocities2D


def compute_strain_rate_map(
    velocities: Velocities2D,
    directions: np.ndarray,
    target_grid: np.ndarray,
    max_distance: float,
    kernel_size: float | None = None,
) -> np.ndarray:
    """Compute a directional strain rate map from scattered velocity vectors.

    Estimates local strain rate tensors around the source points, interpolates
    them onto `target_grid`, and projects them onto `directions`.

    Args:
        velocities: Source positions (M, xy) and velocities (M, xy).
        directions: Direction(s) to project onto, shape (2,) or (M, xy).
        target_grid: Positions to evaluate the map at, shape (..., xy).
        max_distance: Neighbourhood radius for local strain rate estimation.
        kernel_size: Standard deviation of the gaussian kernel used to
            interpolate strain rates onto `target_grid`. Defaults to
            `max_distance`.

    Returns:
        strain_rate_map: Normal strain rate per grid point, shape (...,).
    """
    strain_rates, centroids = estimate_local_strain_rates(velocities, max_distance)
    kernel_size = kernel_size or max_distance

    grid_shape = target_grid.shape[:-1]
    target_grid = target_grid.reshape(-1, target_grid.shape[-1])

    interpolated = interpolate_strain_rates(
        strain_rates, centroids, target_grid, kernel_size
    )

    directions = np.broadcast_to(directions, target_grid.shape)
    return project_strain_rates(interpolated, directions).reshape(grid_shape)


def estimate_local_strain_rates(
    velocities: Velocities2D,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate strain rate tensors from a local neighbourhood around each point.

    For every point, all points within `max_distance` are collected and their
    velocities are fitted to a local affine field to estimate the strain rate
    tensor at the neighbourhood centroid.

    Args:
        velocities: Positions and [vx, vz] velocity at each point.
        max_distance: Radius used to collect neighbours around each point.

    Returns:
        strain_rates: Strain rate tensors, shape (K, 2, 2).
        centroids: Centroid positions of each neighbourhood, shape (K, 2).
    """
    neighbourhoods = _collect_neighbourhoods(velocities.positions, max_distance)

    strain_rates = []
    centroids = []
    for indices in neighbourhoods:
        result = _try_estimate_strain_rate(velocities[indices])
        if result is not None:
            strain_rate, centroid = result
            strain_rates.append(strain_rate)
            centroids.append(centroid)

    return np.array(strain_rates), np.array(centroids)


def _collect_neighbourhoods(
    positions: np.ndarray, max_distance: float
) -> list[np.ndarray]:
    tree = KDTree(positions)
    return [np.array(indices) for indices in tree.query_ball_tree(tree, max_distance)]


def _try_estimate_strain_rate(
    velocities: Velocities2D,
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        strain_rate = estimate_strain_rate_2d(velocities)
    except ValueError:
        return None
    return strain_rate, velocities.positions.mean(axis=0)


def interpolate_strain_rates(
    strain_rates: np.ndarray,
    centroids: np.ndarray,
    query_positions: np.ndarray,
    kernel_bandwidth: float,
) -> np.ndarray:
    """Interpolate strain rate tensors from centroids to new query positions.

    Uses a Gaussian kernel density estimate: each centroid contributes a
    gaussian-weighted vote to every query position, and the votes are
    normalized by the total gaussian weight at that position.

    Args:
        strain_rates: Strain rate tensors at the centroids, shape (K, 2, 2).
        centroids: Centroid positions, shape (K, 2).
        query_positions: Positions to interpolate to, shape (M, 2).
        kernel_bandwidth: Standard deviation of the gaussian kernel.

    Returns:
        interpolated: Strain rate tensors at the query positions, shape (M, 2, 2).
    """
    _check_array_is_n_by_2(centroids, "centroids")
    _check_array_is_n_by_2(query_positions, "query_positions")

    components = strain_rates.reshape(len(strain_rates), 4)
    weights = _gaussian_weights(query_positions, centroids, kernel_bandwidth)
    weighted_components = weights @ components
    total_weight = weights.sum(axis=1, keepdims=True)
    interpolated = weighted_components / total_weight
    return interpolated.reshape(len(query_positions), 2, 2)


def _gaussian_weights(
    query_positions: np.ndarray, centroids: np.ndarray, kernel_bandwidth: float
) -> np.ndarray:
    distances = cdist(query_positions, centroids)
    return np.exp(-0.5 * (distances / kernel_bandwidth) ** 2)


def project_strain_rates(
    strain_rates: np.ndarray, directions: np.ndarray
) -> np.ndarray:
    """Project strain rate tensors onto directions to get normal strain rates.

    Args:
        strain_rates: Strain rate tensors, shape (M, 2, 2).
        directions: Direction vectors, shape (M, 2). Need not be normalized.

    Returns:
        strain_rates_along_directions: Normal strain rate per point, shape (M,).
    """
    _check_array_is_n_by_2(directions, "directions")
    normals = _normalize_rows(np.asarray(directions, dtype=float))
    return np.einsum("mi,mij,mj->m", normals, strain_rates, normals)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def estimate_strain_rate_2d(velocities: Velocities2D) -> np.ndarray:
    """Estimate the 2D strain rate tensor from velocities at scattered points.

    Fits a local affine velocity field
        v_x = a0 + a1*dx + a2*dz
        v_z = b0 + b1*dx + b2*dz
    (dx, dz measured relative to the point centroid) via least squares,
    then reads the strain rate tensor off the fitted gradients:
        exx = a1, ezz = b2, exz = 0.5*(a2 + b1)

    Args:
        velocities: Positions (N >= 3, non-collinear) and [vx, vz] velocities.

    Returns:
        strain_rate: Symmetric strain rate tensor [[exx, exz], [exz, ezz]], evaluated at the centroid of the positions.
    """
    positions = velocities.positions
    velocity_vectors = velocities.velocities

    if positions.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 points to fit a 2D velocity gradient, got {positions.shape[0]}"
        )

    centroid = positions.mean(axis=0)
    d = positions - centroid  # (N, 2), relative coordinates

    # design matrix: [1, dx, dz]
    A = np.column_stack([np.ones(len(d)), d[:, 0], d[:, 1]])

    if np.linalg.matrix_rank(A) < 3:
        raise ValueError(
            "Points are collinear (or coincident) — cannot resolve a unique "
            "2D velocity gradient from this configuration."
        )

    coeffs_x, *_ = np.linalg.lstsq(A, velocity_vectors[:, 0], rcond=None)
    coeffs_z, *_ = np.linalg.lstsq(A, velocity_vectors[:, 1], rcond=None)

    _, a1, a2 = coeffs_x  # dv_x/dx, dv_x/dz
    _, b1, b2 = coeffs_z  # dv_z/dx, dv_z/dz

    exx = a1
    ezz = b2
    exz = 0.5 * (a2 + b1)

    return np.array([[exx, exz], [exz, ezz]])


def _check_array_is_n_by_2(arr: np.ndarray, name: str) -> None:
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must be a 2D array of shape (N, 2), got {arr.shape}")


def strain_rate_in_direction(strain_rate: np.ndarray, direction: np.ndarray) -> float:
    """Normal strain rate along an arbitrary direction, given the 2D strain rate tensor.

    Parameters
    ----------
    strain_rate : (2, 2) array
        Symmetric strain rate tensor, e.g. from estimate_strain_rate_2d.
    direction : (2,) array
        Direction vector [nx, nz]. Does not need to be pre-normalized.

    Returns
    -------
    float
        Normal strain rate along `direction`.
    """
    n = np.asarray(direction, dtype=float)
    n = n / np.linalg.norm(n)
    return n @ strain_rate @ n
