import numpy as np
from imagelib import Image

from strain_off_grid.phantoms import DynamicPhantom


def compute_rate_strain_curve(
    strain_rate_map: Image,
    phantom: DynamicPhantom,
    position_at_frame_0: np.ndarray,
):
    """

    Computes the strain rate curve from the given strain rate map.

    Args:
        strain_rate_map: Strain rate map, shape (time, y, x).

    Returns:
        strain_rate_curve: Strain rate curve, shape (time,).
    """
    time_limits = strain_rate_map.limits[0]
    timestamps = np.linspace(time_limits.min, time_limits.max, strain_rate_map.shape[0])
    strain_rate_curve = np.zeros_like(timestamps)
    for n, t in enumerate(timestamps):
        positions = phantom.translate_from_time_to_time(
            position_at_frame_0, t0=time_limits.min, t1=t
        )
        current_strain_rate_map = strain_rate_map[n]
        indices = current_strain_rate_map.coordinates_to_indices(positions[:, [-1, 0]])[
            0
        ]
        strain_rate_curve[n] = current_strain_rate_map.array[indices[0], indices[1]]
    return strain_rate_curve
