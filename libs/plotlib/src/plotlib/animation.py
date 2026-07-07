import numpy as np


def linear_clipped(t, factor=0.9):
    t = (t - 0.5) / factor + 0.5

    return np.clip(t, 0, 1)


def smooth(t: float) -> float:
    """Smooth transition function for animations, copied from manim.

    Zero first and second derivatives at t=0 and t=1.
    Equivalent to bezier([0, 0, 0, 1, 1, 1])
    """
    s = 1 - t
    return (t**3) * (10 * s * s + 5 * s * t + t * t)


def map_range(t, start, end, init_start=0, init_end=1):
    return start + (end - start) * (t - init_start) / (init_end - init_start)


def smooth_range(t, start, end):
    return map_range(smooth(t), start, end)


def smooth_segment(start: np.ndarray, end: np.ndarray, n_steps: int) -> np.ndarray:
    t_values = np.linspace(0, 1, n_steps, endpoint=False).reshape(-1, 1)
    return smooth_range(t_values, start, end)


def smooth_position_loop(positions: np.ndarray, n_steps: int) -> np.ndarray:
    """Smoothly interpolate through positions in a loop, returning to the first point.

    Args:
        positions: Array of shape (n_points, n_dim).
        n_steps: Number of interpolation steps per segment.

    Returns:
        Array of shape (n_points * n_steps, n_dim).
    """
    n_points = len(positions)
    segments = [
        smooth_segment(positions[i], positions[(i + 1) % n_points], n_steps)
        for i in range(n_points)
    ]
    return np.concatenate(segments, axis=0)
