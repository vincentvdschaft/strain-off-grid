import matplotlib.patheffects as pe
import numpy as np


def add_ruler(
    ax,
    start,
    end,
    formatter,
    color="black",
    linewidth=1.5,
    label_side="above",
    label_offset=0.05,
    fontsize=None,
    black_edge=True,
):
    """Add a ruler (scale bar) to an axes.

    Draws a line between two 2D points with a centered label showing the scale.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the ruler on.
    start : (float, float)
        (x, y) start point of the ruler in data coordinates.
    end : (float, float)
        (x, y) end point of the ruler in data coordinates.
    formatter : callable
        Called with the Euclidean length of the ruler and must return the label
        string. Example: ``lambda d: f"{d:.1f} mm"``.
    color : str or color, optional
        Line and text color. Default is ``"black"``.
    linewidth : float, optional
        Width of the ruler line. Default is ``1.5``.
    label_side : {"above", "below"}, optional
        Which side of the line to place the label, where "above" is the
        left-normal direction. Default is ``"above"``.
    label_offset : float, optional
        Distance between the line and the label in data coordinates.
        Default is ``0.05``.
    """
    x0, y0 = start
    x1, y1 = end

    ax.plot([x0, x1], [y0, y1], color=color, linewidth=linewidth, solid_capstyle="butt")

    length = np.hypot(x1 - x0, y1 - y0)
    label = formatter(length)

    x_mid = (x0 + x1) / 2
    y_mid = (y0 + y1) / 2

    dx, dy = x1 - x0, y1 - y0
    nx, ny = -dy, dx
    norm = np.hypot(nx, ny)
    if norm > 0:
        nx, ny = nx / norm, ny / norm

    sign = 1 if label_side == "above" else -1
    text_handle = ax.text(
        x_mid + sign * nx * label_offset,
        y_mid + sign * ny * label_offset,
        label,
        color=color,
        ha="center",
        va="bottom" if label_side == "above" else "top",
        fontsize=fontsize,
    )
    if black_edge:
        text_handle.set_path_effects(
            [
                pe.withStroke(linewidth=3, foreground="black"),
                pe.Normal(),
            ]
        )
