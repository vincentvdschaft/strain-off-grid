import numpy as np
from imagelib import Image
from zea.tracking import BlockMatchingTracker

from strain_off_grid.dimconvert import reduce_3d_to_2d
from strain_off_grid.velocities import Velocities2D


def get_baseline_tracking_positions(
    image0: Image,
    image1: Image,
    positions0: np.ndarray,
    dt: float,
    block_size: tuple[int, int] = (5, 5),
) -> np.ndarray:
    positions0 = reduce_3d_to_2d(positions0)

    tracker = BlockMatchingTracker(extent=image0.extent, block_size=block_size)

    return tracker.track(
        prev_frame=image0.array,
        next_frame=image1.array,
        points=positions0,
    )


def get_baseline_tracking_velocities(
    image0: Image,
    image1: Image,
    positions0: np.ndarray,
    dt: float,
    block_size: tuple[int, int] = (5, 5),
) -> Velocities2D:
    positions1 = get_baseline_tracking_positions(
        image0=image0,
        image1=image1,
        positions0=positions0,
        dt=dt,
        block_size=block_size,
    )

    return Velocities2D(
        positions=positions0,
        velocities=(positions1 - positions0) / dt,
    )
