from typing import Tuple, TypeAlias

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from plotlib import (
    DimensionsGrid,
    DimensionsSingle,
    DimensionsSingleBesidesGrid,
)
from plotlib.plotlib import MPLFigure

AxesArray: TypeAlias = npt.NDArray[np.object_]


def quickfig_single(dimensions: DimensionsSingle) -> Tuple[MPLFigure, Axes]:
    """Create a quick figure with single dimensions.

    Args:
        dimensions (DimensionsSingle): The single dimensions to use.

    Returns:
        MPLFigure: The created figure.
        Axes: The created axis.
    """
    assert isinstance(dimensions, DimensionsSingle)

    fig = MPLFigure(figsize=dimensions.figsize)
    ax = fig.add_ax(
        x=dimensions.margins.left,
        y=dimensions.margins.top,
        width=dimensions.axis_size.width,
        height=dimensions.axis_size.height,
    )
    return fig, ax


def quickfig_grid(dimensions: DimensionsGrid) -> Tuple[MPLFigure, AxesArray]:
    """Create a quick figure with grid dimensions.

    Args:
        dimensions_grid (DimensionsGrid): The grid dimensions to use.

    Returns:
        MPLFigure: The created figure.
        AxesArray: The created axes.
    """
    assert isinstance(dimensions, DimensionsGrid)

    fig = MPLFigure(figsize=dimensions.figsize)

    axes = fig.add_axes_grid(
        n_rows=dimensions.grid_shape.n_rows,
        n_cols=dimensions.grid_shape.n_cols,
        x=dimensions.margins.left,
        y=dimensions.margins.top,
        width=dimensions.axis_size.width,
        height=dimensions.axis_size.height,
        spacing=dimensions.grid_spacing,
    )
    return fig, axes


def quickfig_single_besides_grid(
    dimensions: DimensionsSingleBesidesGrid, grid_on_right: bool = True
) -> Tuple[MPLFigure, Axes, AxesArray]:
    """Create a quick figure with single beside grid dimensions.

    Args:
        dimensions (DimensionsSingleBesidesGrid): The single beside grid dimensions to use.

    Returns:
        MPLFigure: The created figure.
        Axes: The created single axis.
        AxesArray: The created grid axes.
    """
    assert isinstance(dimensions, DimensionsSingleBesidesGrid)

    fig = MPLFigure(figsize=dimensions.figsize)

    single_y = dimensions.margins.top

    if grid_on_right:
        single_x = dimensions.margins.left
    else:
        single_x = (
            dimensions.margins.left
            + dimensions.grid_total_size.width
            + dimensions.middle_spacing
        )

    if grid_on_right:
        grid_x = (
            dimensions.margins.left
            + dimensions.single_axis_shape.width
            + dimensions.middle_spacing
        )
    else:
        grid_x = dimensions.margins.left
    grid_y = dimensions.margins.top

    ax_single = fig.add_ax(
        x=single_x,
        y=single_y,
        width=dimensions.single_axis_shape.width,
        height=dimensions.single_axis_shape.height,
    )

    axes_grid = fig.add_axes_grid(
        n_rows=dimensions.grid_shape.n_rows,
        n_cols=dimensions.grid_shape.n_cols,
        x=grid_x,
        y=grid_y,
        width=dimensions.grid_axis_shape.width,
        height=dimensions.grid_axis_shape.height,
        spacing=dimensions.grid_spacing,
    )

    return fig, ax_single, axes_grid
