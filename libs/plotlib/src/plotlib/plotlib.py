from collections.abc import Generator, Iterable, Sequence

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.widgets
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import ArrowStyle, ConnectionPatch, FancyArrowPatch, Rectangle
from matplotlib.transforms import Bbox

from plotlib.constants import IEEE_COLUMN_WIDTH

from .boxconnection import get_bbox_connections


class MPLFigure:
    """Figure class. Wraps a matplotlib figure object."""

    def __init__(self, figsize: tuple[float, float] = (IEEE_COLUMN_WIDTH, 3.0)) -> None:
        self.fig: Figure = plt.figure(figsize=figsize)
        self.figsize: np.ndarray = self.fig.get_size_inches()
        self.axes: list[Axes] = []
        self.grids: list[np.ndarray] = []
        self.cbar_axes: list[Axes] = []
        self.colorbars: list[matplotlib.colorbar.Colorbar] = []
        self.buttons: list[matplotlib.widgets.Button] = []
        self.widget_axes: list[Axes] = []

    @property
    def width(self) -> float:
        """Returns the width of the figure in inches."""
        return self.figsize[0]

    @property
    def height(self) -> float:
        """Returns the height of the figure in inches."""
        return self.figsize[1]

    def add_ax(
        self,
        x: float,
        y: float,
        width: float | None = None,
        height: float | None = None,
        aspect: float | Sequence[float] | None = None,
    ) -> Axes:
        """Add an axis to the figure with the given bounding box in inches."""
        width, height = interpret_width_height_aspect(
            width=width, height=height, aspect=aspect
        )
        bbox_inches = Bbox.from_bounds(x, y, width, height)
        return self._add_ax(bbox_inches)

    def _add_ax(self, bbox_inches: Bbox) -> Axes:
        """Add an axis to the figure with the given bounding box in inches."""
        bbox = self.bbox_norm(bbox_inches)
        ax = self.fig.add_axes(bbox)
        self.axes.append(ax)

        return ax

    def add_button(
        self,
        label: str,
        x: float,
        y: float,
        width: float | None = None,
        height: float | None = None,
        aspect: float | Sequence[float] | None = None,
        **kwargs,
    ) -> matplotlib.widgets.Button:
        """Add a button to the figure."""
        width, height = interpret_width_height_aspect(
            width=width, height=height, aspect=aspect
        )
        bbox_inches = Bbox.from_bounds(x, y, width, height)
        button = self._add_button(label, bbox_inches, **kwargs)
        return button

    def _add_button(
        self, label: str, bbox_inches: Bbox, **kwargs
    ) -> matplotlib.widgets.Button:
        """Add a button to the figure."""
        bbox = self.bbox_norm(bbox_inches)
        ax = self.fig.add_axes(bbox)
        button = matplotlib.widgets.Button(ax=ax, label=label, **kwargs)
        self.buttons.append(button)
        self.widget_axes.append(ax)
        return button

    def add_textbox(
        self,
        x: float,
        y: float,
        width: float | None = None,
        height: float | None = None,
        aspect: float | Sequence[float] | None = None,
        **kwargs,
    ) -> matplotlib.widgets.TextBox:
        """Add a textbox to the figure."""
        width, height = interpret_width_height_aspect(
            width=width, height=height, aspect=aspect
        )
        bbox_inches = Bbox.from_bounds(x, y, width, height)
        textbox = self._add_textbox(bbox_inches, **kwargs)
        return textbox

    def _add_textbox(self, bbox_inches: Bbox, **kwargs) -> matplotlib.widgets.TextBox:
        """Add a textbox to the figure."""
        bbox = self.bbox_norm(bbox_inches)
        ax = self.fig.add_axes(bbox)
        textbox = matplotlib.widgets.TextBox(ax=ax, **kwargs)
        self.widget_axes.append(ax)
        return textbox

    def bbox_norm(self, bbox_inches: Bbox) -> Bbox:
        """Normalizes a bounding box in inches to a bounding box in normalized figure coordinates."""
        x0 = bbox_inches.x0 / self.figsize[0]
        y0 = 1.0 - bbox_inches.y1 / self.figsize[1]
        x1 = bbox_inches.x1 / self.figsize[0]
        y1 = 1.0 - bbox_inches.y0 / self.figsize[1]

        return Bbox(((x0, y0), (x1, y1)))

    def bbox_norm_inv(self, bbox_norm: Bbox) -> Bbox:
        """Inverse of bbox_norm."""
        x0 = bbox_norm.x0 * self.figsize[0]
        y0 = (1.0 - bbox_norm.y1) * self.figsize[1]
        x1 = bbox_norm.x1 * self.figsize[0]
        y1 = (1.0 - bbox_norm.y0) * self.figsize[1]
        return Bbox(((x0, y0), (x1, y1)))

    def coord_norm(self, x: float, y: float) -> tuple[float, float]:
        """Converts a coordinate in inches to a coordinate in normalized figure coordinates."""
        x_norm = x / self.figsize[0]
        y_norm = 1.0 - y / self.figsize[1]
        return x_norm, y_norm

    def coord_norm_inv(self, x_norm: float, y_norm: float) -> tuple[float, float]:
        """Inverse of coord_norm."""
        x = x_norm * self.figsize[0]
        y = (1.0 - y_norm) * self.figsize[1]
        return x, y

    def width_norm(self, width: float) -> float:
        """Converts a width in inches to a width in normalized figure coordinates."""
        return width / self.figsize[0]

    def height_norm(self, height: float) -> float:
        """Converts a height in inches to a height in normalized figure coordinates."""
        return height / self.figsize[1]

    def get_ax_bbox(self, ax: Axes) -> Bbox:
        bbox = ax.get_position()
        return self.bbox_norm_inv(bbox)

    def add_axes_grid(
        self,
        n_rows: int,
        n_cols: int,
        x: float,
        y: float,
        spacing: float | Sequence[float],
        width: float | None = None,
        height: float | None = None,
        aspect: float | Sequence[float] | None = None,
    ) -> np.ndarray:
        """Add a grid of axes to the figure. The bottom left corner of the grid is at (x0, y0) in inches."""

        if isinstance(spacing, (float, int)):
            spacing_horizontal = spacing
            spacing_vertical = spacing
        else:
            spacing_horizontal = spacing[0]
            spacing_vertical = spacing[1]

        width, height = interpret_width_height_aspect(
            width=width, height=height, aspect=aspect
        )

        axes_array = np.empty((n_rows, n_cols), dtype=object)
        for row in range(n_rows):
            for col in range(n_cols):
                bbox_inches = Bbox.from_bounds(
                    x + (width + spacing_horizontal) * col,
                    y + (height + spacing_vertical) * row,
                    width,
                    height,
                )
                ax = self._add_ax(bbox_inches)
                axes_array[row, col] = ax
                self.axes.append(ax)
        self.grids.append(axes_array)

        return axes_array

    def get_total_bbox(self, margin: float = 0.2) -> Bbox:
        """Get the total bounding box of the figure in inches."""
        bbox0 = self.axes[0].get_position()
        x0 = bbox0.x0
        y0 = bbox0.y0
        x1 = bbox0.x1
        y1 = bbox0.y1
        for ax in self.axes:
            bbox = ax.get_position()
            x0 = min(x0, bbox.x0)
            y0 = min(y0, bbox.y0)
            x1 = max(x1, bbox.x1)
            y1 = max(y1, bbox.y1)

        # ----------------------------------------------------------------------
        # Add the margin
        # ----------------------------------------------------------------------
        bbox = Bbox(((x0, y0), (x1, y1)))
        print(f"bbox before: {bbox}")
        bbox = add_margin_to_bbox(bbox, margin)
        print(f"bbox after: {bbox}")

        bbox = Bbox(((bbox.x0, 1.0 - bbox.y1), (bbox.x1, 1.0 - bbox.y0)))
        return self.bbox_norm_inv(bbox)

    def add_text(self, x: float, y: float, text: str, **kwargs) -> matplotlib.text.Text:
        """Add text to the figure."""
        x_norm = x / self.figsize[0]
        y_norm = 1.0 - y / self.figsize[1]
        return self.fig.text(x_norm, y_norm, text, **kwargs)

    def savefig(self, *args, margin: float | None = None, **kwargs) -> None:
        """Save the figure."""
        if "bbox_inches" not in kwargs:
            if margin is not None:
                kwargs["bbox_inches"] = self.get_total_bbox(margin=margin)
        self.fig.savefig(*args, **kwargs)

    def get_ax_width(self, ax: Axes) -> float:
        """Returns the width of the Axes in inches.

        Returns
        -------
        width : float
            The width of the figure in inches.
        """
        bbox = ax.get_position()
        width = bbox.x1 - bbox.x0
        width *= self.figsize[0]
        return width

    def get_ax_height(self, ax: Axes) -> float:
        """Returns the height of the Axes in inches.

        Returns
        -------
        height : float
            The height of the figure in inches.
        """
        bbox = ax.get_position()
        height = bbox.y1 - bbox.y0
        height *= self.figsize[1]
        return height

    def get_ax_position(self, ax: Axes) -> tuple[float, float]:
        """Returns the position of the Axes in figure coordinates.

        Returns
        -------
        x0, y0 : float, float
            The position of the Axes in figure coordinates.
        """
        bbox = ax.get_position()
        bbox = self.bbox_norm_inv(bbox)
        return bbox.x0, bbox.y0

    def data_to_figure_coords(
        self, ax: Axes, x: float, y: float
    ) -> tuple[float, float]:
        """"""
        return data_to_figure_coords(fig=self.fig, ax=ax, x=x, y=y)

    def add_inset_plot(
        self,
        ax: Axes,
        width: float | None = None,
        height: float | None = None,
        aspect: float | Sequence[float] | None = None,
        position: str = "top left",
        margin: float = 0.2,
    ) -> Axes:
        """"""
        parent_x, parent_y = self.get_ax_position(ax)
        parent_width = self.get_ax_width(ax)
        parent_height = self.get_ax_height(ax)
        width, height = interpret_width_height_aspect(
            width=width, height=height, aspect=aspect
        )
        if position == "top left":
            x = parent_x + margin
            y = parent_y + margin
        elif position == "top right":
            x = parent_x + parent_width - width - margin
            y = parent_y + margin
        elif position == "bottom left":
            x = parent_x + margin
            y = parent_y + parent_height - height - margin
        elif position == "bottom right":
            x = parent_x + parent_width - width - margin
            y = parent_y + parent_height - height - margin
        else:
            raise ValueError(f"Invalid position: {position}")

        ax = self.add_ax(x=x, y=y, width=width, height=height)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def add_arrow(
        self,
        ax: Axes,
        data_x: float,
        data_y: float,
        angle_deg: float,
        length_inches: float = 1,
        color: str = "C0",
        **kwargs,
    ) -> FancyArrowPatch:
        """Add an arrow to the figure."""
        fig_x, fig_y = self.data_to_figure_coords(ax, data_x, data_y)
        dx = length_inches * np.cos(angle_deg * np.pi / 180)
        dy = length_inches * np.sin(angle_deg * np.pi / 180)

        dx = self.width_norm(dx)
        dy = self.height_norm(dy)

        kwargs["facecolor"] = color
        kwargs["edgecolor"] = color

        return self.fig.add_artist(
            FancyArrowPatch(
                (fig_x + dx, fig_y + dy),
                (fig_x, fig_y),
                arrowstyle=ArrowStyle.Simple(
                    head_length=1.8,
                    head_width=1.8,
                    tail_width=0.6,
                ),
                mutation_scale=length_inches * 10,
                **kwargs,
            ),
        )

    def add_rectangle_ax(
        self,
        ax: Axes,
        data_x: float,
        data_y: float,
        data_width: float,
        data_height: float,
        *args,
        **kwargs,
    ) -> Rectangle:
        """Add a rectangle to the figure in the data coordinates of the Axes."""
        fig_x0, fig_y0 = self.data_to_figure_coords(ax, data_x, data_y)
        fig_x1, fig_y1 = self.data_to_figure_coords(
            ax, data_x + data_width, data_y + data_height
        )
        fig_width = fig_x1 - fig_x0
        fig_height = fig_y1 - fig_y0

        return self.fig.add_artist(
            Rectangle(
                (fig_x0, fig_y0),
                fig_width,
                fig_height,
                *args,
                **kwargs,
            )
        )

    def draw_bbox_connection(self, bbox0: Bbox, bbox1: Bbox, *args, **kwargs) -> None:
        bbox0 = self.bbox_norm(bbox0)
        bbox1 = self.bbox_norm(bbox1)
        lines, hull = get_bbox_connections(bbox0, bbox1)

        # Draw the lines
        for line in lines:
            line = list(line)
            self.fig.add_artist(
                ConnectionPatch(line[0], line[1], "figure fraction", *args, **kwargs)
            )

    def add_colorbar(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        cmap: str | matplotlib.colors.Colormap,
        vmin: float,
        vmax: float,
        ticks: list[float],
        orientation: str = "vertical",
        **kwargs,
    ) -> matplotlib.colorbar.Colorbar:
        """Adds a colorbar to the figure."""
        ax_cbar = self.add_ax(
            x=x,
            y=y,
            width=width,
            height=height,
        )

        colorbar = matplotlib.colorbar.ColorbarBase(
            ax_cbar,
            cmap=plt.get_cmap(cmap),
            norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
            orientation=orientation,
            ticks=ticks,
            **kwargs,
        )
        self.cbar_axes.append(ax_cbar)
        self.colorbars.append(colorbar)
        return colorbar

    # def __del__(self):
    #     plt.close(self.fig)

    def add_legend(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        labels: list[str] | None = None,
        handles: list[matplotlib.artist.Artist] | None = None,
        ax: Axes | None = None,
        **kwargs,
    ) -> matplotlib.legend.Legend:
        assert ax is not None or (labels is not None and handles is not None), (
            "Either ax or labels and handles should be provided."
        )
        if ax is not None:
            labels = [line.get_label() for line in ax.get_lines()]
            handles = [line for line in ax.get_lines()]

            # Also collect the labels and handles from the scatter plots
            labels += [line.get_label() for line in ax.collections]
            handles += [line for line in ax.collections]

        legend_ax = self.add_ax(x, y, width, height)
        bbox = Bbox.from_bounds(x, y, width, height)
        bbox = self.bbox_norm(bbox)
        print(f"bbox: {bbox}")
        legend = self.fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=bbox,
            mode="expand",
            borderaxespad=0.0,
            **kwargs,
        )
        legend_ax.set_frame_on(False)
        legend_ax.set_xticks([])
        legend_ax.set_yticks([])
        return legend


def add_ruler(
    ax: Axes,
    length: float,
    position: tuple[float, float] = (0.1, 0.1),
    orientation: str = "horizontal",
    label: str | None = None,
    **kwargs,
) -> None:
    """Adds a ruler to the given axis.

    Parameters
    ----------
    ax : Axes
        The axis to add the ruler to.
    length : float
        The length of the ruler in axis coordinates (0 to 1).
    position : tuple of float
        The position of the ruler in axis coordinates (0 to 1).
    orientation : str
        The orientation of the ruler ('horizontal' or 'vertical').
    kwargs : dict
        Additional keyword arguments to pass to the plot function.
    """
    is_horizontal = orientation.lower() in ("horizontal", "h", "hor")
    if not is_horizontal:
        assert orientation.lower() in (
            "vertical",
            "v",
            "ver",
            "vert",
        ), "Orientation must be 'horizontal' or 'vertical'."

    if is_horizontal:
        x0 = position[0]
        y0 = position[1]
        x1 = position[0] + length
        y1 = position[1]
    else:
        x0 = position[0]
        y0 = position[1]
        x1 = position[0]
        y1 = position[1] + length

    ax.plot(
        [x0, x1],
        [y0, y1],
        transform=ax.transAxes,
        **kwargs,
    )
    if label is not None:
        if is_horizontal:
            ax.text(
                (x0 + x1) / 2,
                y0 - 0.05,
                label,
                ha="center",
                va="top",
                transform=ax.transAxes,
            )
        else:
            ax.text(
                x0 - 0.05,
                (y0 + y1) / 2,
                label,
                ha="right",
                va="center",
                rotation=90,
                transform=ax.transAxes,
            )


def interpret_width_height_aspect(
    width: float | None = None,
    height: float | None = None,
    aspect: float | Sequence[float] | None = None,
) -> tuple[float, float]:
    """Interprets the width, height, and aspect parameters to form just a width and
    height. If aspect is provided, either as a float or and extent, one of the other
    two parameters can be inferred.

    Parameters
    ----------
    width : float
        The width of the ax. May be ommitted if height and aspect are provided.
    height : float
        The height of the ax. May be ommitted if width and aspect are provided.
    aspect : float or size 4 tuple/list/array
        The aspect ratio (delta y)/(delta x) or extent from which the aspect ratio can
        be computed.

    Returns
    -------
    width, height : float, float
        The width and height of the ax.
    """
    if width is not None and height is not None:
        return width, height

    try:
        aspect = float(aspect)
    except TypeError:
        extent_width = max(aspect[0], aspect[1]) - min(aspect[0], aspect[1])
        extent_height = max(aspect[2], aspect[3]) - min(aspect[2], aspect[3])
        aspect = extent_height / extent_width

    if width is None:
        assert height is not None, "Either width or height should be specified."
        width = height / aspect
    else:
        height = width * aspect

    return float(width), float(height)


def remove_internal_ticks(grid: np.ndarray) -> None:
    """Remove internal ticks from a grid of axes."""
    grid = np.atleast_2d(grid)
    n_rows, n_cols = grid.shape
    for row in range(n_rows):
        for col in range(n_cols):
            ax = grid[row, col]
            if row != n_rows - 1:
                ax.set_xticks([])
            if col != 0:
                ax.set_yticks([])


def remove_internal_labels(grid: np.ndarray) -> None:
    """Remove internal labels from a grid of axes."""
    grid = np.atleast_2d(grid)
    n_rows, n_cols = grid.shape
    for row in range(n_rows):
        for col in range(n_cols):
            ax = grid[row, col]
            if row != n_rows - 1:
                ax.set_xlabel("")
            if col != 0:
                ax.set_ylabel("")


def remove_internal_titles(grid: np.ndarray) -> None:
    """Remove internal titles from a grid of axes."""
    grid = np.atleast_2d(grid)
    n_rows, n_cols = grid.shape
    for row in range(1, n_rows):
        for col in range(n_cols):
            ax = grid[row, col]
            ax.set_title("")


def remove_internal_ticks_labels(grid: np.ndarray) -> None:
    """Remove internal ticks and labels from a grid of axes."""
    remove_internal_labels(grid)
    remove_internal_ticks(grid)


def remove_internal_last_ticks_grid(axes_grid: np.ndarray) -> None:
    """Remove the last ticks of the axes in a closely spaced grid."""
    axes_grid = np.atleast_2d(axes_grid)
    n_rows, n_cols = axes_grid.shape
    for row in range(n_rows - 1):
        ax = axes_grid[row, 0]
        ax.get_yticklabels()[0].set_visible(False)
    for col in range(n_cols - 1):
        ax = axes_grid[-1, col]
        ax.get_xticklabels()[-1].set_visible(False)


def box_fit_tick_labels(ax: Axes | Iterable, x: bool = True, y: bool = False) -> None:
    """Adjust the box of the axis to fit the tick labels."""
    for ax in _flat_iterate(ax):
        if x:
            _box_fit_tick_xlabels(ax)
        if y:
            _box_fit_tick_ylabels(ax)


def _box_fit_tick_xlabels(ax: Axes) -> None:
    labels = ax.get_xticklabels()
    for i, label in enumerate(labels):
        if i == 0:
            label.set_ha("left")
        elif i == len(labels) - 1:
            label.set_ha("right")
        else:
            label.set_ha("center")


def _box_fit_tick_ylabels(ax: Axes) -> None:
    labels = ax.get_yticklabels()
    for i, label in enumerate(labels):
        if i == 0:
            label.set_va("bottom")
        elif i == len(labels) - 1:
            label.set_va("top")
        else:
            label.set_va("center")


def data_to_figure_coords(
    fig: Figure,
    ax: Axes,
    x: float,
    y: float,
) -> tuple[float, float]:
    coords = x, y
    # Transform (x, y) from data coordinates to display coordinates
    # coords = ax.transAxes.transform(coords)
    coords = ax.transData.transform(coords)

    # Transform display coordinates to figure coordinates
    fig_x, fig_y = fig.transFigure.inverted().transform(coords)

    return fig_x, fig_y


def add_margin_to_bbox(bbox: Bbox, margin: float | Sequence[float] | Bbox) -> Bbox:
    """Adds a margin to a bounding box.

    Parameters
    ----------
    bbox : Bbox
        The bounding box.
    margin : float or Bbox
        The margin to add to the bounding box. If a float, the same margin is added to
        all sides. If a Bbox, the margin is added to each side separately.

    Returns
    -------
    bbox : Bbox
        The bounding box with the margin added.
    """
    if isinstance(margin, (float, int)):
        margin = Bbox([[margin, margin], [margin, margin]])
    elif isinstance(margin, (list, tuple, np.ndarray)):
        margin = Bbox(margin)
    x0 = bbox.x0 - margin.x0
    y0 = bbox.y0 - margin.y0
    x1 = bbox.x1 + margin.x1
    y1 = bbox.y1 + margin.y1
    return Bbox([[x0, y0], [x1, y1]])


def mmplot(ax: Axes, decimals: int = 0) -> None:
    """Configures a plot to have millimeter units on the axes."""
    formatter = plt.FuncFormatter(lambda x, _: f"{x * 1e3:.{decimals}f}")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")


def mm_formatter_ax(
    ax: Axes | Iterable,
    x: bool = True,
    y: bool = True,
    decimals: int = 0,
) -> None:
    """Configures an axis to have millimeter units on the axes."""
    for ax in _flat_iterate(ax):
        _mm_formatter_ax(ax, x=x, y=y, decimals=decimals)


def _mm_formatter_ax(
    ax: Axes, x: bool = True, y: bool = True, decimals: int = 0
) -> None:
    """Configures an axis to have millimeter units on the axes."""
    formatter = plt.FuncFormatter(lambda x, _: f"{x * 1e3:.{decimals}f}")
    if x:
        ax.xaxis.set_major_formatter(formatter)
    if y:
        ax.yaxis.set_major_formatter(formatter)


def remove_axes(axes: Axes | Iterable) -> None:
    """Removes the axes from the figure."""

    for ax in _flat_iterate(axes):
        axes.axis("off")
        axes.set_xticks([])
        axes.set_yticks([])


def remove_ticks(axes: Axes | Iterable) -> None:
    """Removes the ticks from the axes."""
    for ax in _flat_iterate(axes):
        ax.set_xticks([])
        ax.set_yticks([])


def remove_ticks_labels(axes: Axes | Iterable) -> None:
    """Removes the ticks and labels from the axes."""
    remove_ticks(axes)
    for ax in _flat_iterate(axes):
        ax.set_xlabel("")
        ax.set_ylabel("")


def _flat_iterate(
    axes: Axes | Iterable,
) -> Generator[Axes, None, None]:
    """Iterate over axes in a possibly nested structure."""
    if not isinstance(axes, Axes):
        for ax in axes:
            yield from _flat_iterate(ax)
    else:
        yield axes


def flip_ylims(ax: Axes) -> None:
    """Flip the y-limits of the given axis."""
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[1], ylim[0])


def flip_xlims(ax: Axes) -> None:
    """Flip the x-limits of the given axis."""
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[1], xlim[0])
