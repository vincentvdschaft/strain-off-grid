import numpy as np


def reduce_3d_to_2d(arr: np.ndarray) -> np.ndarray:
    """Reduces an (N, 3) array to 2D by taking the x and z components.

    An (N, 2) array is already 2D and is returned unchanged.
    """
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(
            f"Input array must be of shape (N, 2) or (N, 3), got {arr.shape}"
        )
    if arr.shape[1] == 2:
        return arr
    return arr[:, [0, 2]]


def promote_2d_to_3d(arr: np.ndarray) -> np.ndarray:
    """Promotes an (N, 2) array to 3D by adding a zero y-component.

    An (N, 3) array is already 3D and is returned unchanged.
    """
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(
            f"Input array must be of shape (N, 2) or (N, 3), got {arr.shape}"
        )
    if arr.shape[1] == 3:
        return arr
    return np.column_stack([arr[:, 0], np.zeros(arr.shape[0]), arr[:, 1]])
