import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

# from imagelib import Image


def apply_dynamic_range_curve(curve: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Apply a dynamic range curve to values in [0, 1].

    Parameters
    ----------
    curve : (n,) np.ndarray
        Control points sampled uniformly over x in [0, 1]. For example,
        curve[0] is the output at x=0, curve[-1] at x=1.
        Values are expected to be in [0, 1].
    values : np.ndarray
        Array of any shape with values in [0, 1] to which the curve is applied.

    Returns
    -------
    np.ndarray
        Array of same shape as `values`, transformed by the spline and clipped to [0, 1].
    """
    curve = np.asarray(curve, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if curve.ndim != 1 or curve.size < 2:
        raise ValueError("curve must be a 1D array with at least 2 points")

    cmin, cmax = values.min(), values.max()

    # x-positions for the uniformly spaced control points in [0, 1]
    x = np.linspace(cmin, cmax, num=curve.size)

    # shape-preserving cubic spline (monotone where data are monotone)
    spline = PchipInterpolator(x, curve, extrapolate=True)

    # evaluate spline at the input values
    transformed = spline(values)

    # ensure the result stays in [0, 1]
    transformed = np.clip(transformed, cmin, cmax)

    # preserve input dtype if it is floating, otherwise return float32
    if np.issubdtype(values.dtype, np.floating):
        return transformed.astype(values.dtype, copy=False)
    return transformed.astype(np.float32, copy=False)


def plot_curve(setpoints: np.ndarray) -> None:
    """
    Plot a dynamic range curve defined by control points.

    Parameters
    ----------
    setpoints : (n,) np.ndarray
        Control points sampled uniformly over x in [0, 1]. For example,
        setpoints[0] is the output at x=0, setpoints[-1] at x=1.
        Values are expected to be in [0, 1].
    """

    if setpoints.ndim != 1 or setpoints.size < 2:
        raise ValueError("setpoints must be a 1D array with at least 2 points")

    x = np.linspace(0.0, 1.0, num=setpoints.size)
    spline = PchipInterpolator(x, setpoints, extrapolate=True)

    x_fine = np.linspace(0.0, 1.0, num=100)
    y_fine = spline(x_fine)

    plt.plot(x_fine, y_fine, label="Dynamic Range Curve")
    plt.scatter(x, setpoints, color="red", label="Control Points")
    plt.title("Dynamic Range Curve")
    plt.xlabel("Input Value")
    plt.ylabel("Output Value")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.legend()
    plt.show()
