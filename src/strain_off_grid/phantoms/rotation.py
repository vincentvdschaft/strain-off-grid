import numpy as np


def rotation_matrix_x(theta: float) -> np.ndarray:
    """Returns the rotation matrix for a rotation around the x-axis by angle theta (in radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_matrix_y(theta: float) -> np.ndarray:
    """Returns the rotation matrix for a rotation around the y-axis by angle theta (in radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_matrix_z(theta: float) -> np.ndarray:
    """Returns the rotation matrix for a rotation around the z-axis by angle theta (in radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
