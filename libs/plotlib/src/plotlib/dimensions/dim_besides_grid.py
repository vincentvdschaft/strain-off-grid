from typing import Tuple, TypeAlias

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from ..plotlib import MPLFigure
from .aspect import extent_to_aspect_if_needed
from .margins import Margins
from .shape import FloatShape, IntShape
from .spacing import Spacing

AxesArray: TypeAlias = npt.NDArray[np.object_]


class DimensionsSingleBesidesGrid:
    def __init__(
        self,
        margins: Margins,
        grid_shape: IntShape,
        figsize: FloatShape,
        grid_spacing: Spacing,
        middle_spacing: float,
        single_axis_shape: FloatShape,
    ):
        assert isinstance(margins, Margins)
        assert isinstance(grid_spacing, Spacing)
        self._margins = margins
        self._grid_shape = IntShape(grid_shape[0], grid_shape[1])
        self._figsize = FloatShape(figsize[0], figsize[1])
        self._grid_spacing = Spacing(grid_spacing[0], grid_spacing[1])
        self._single_axis_shape = FloatShape(single_axis_shape[0], single_axis_shape[1])
        self._middle_spacing = float(middle_spacing)

    @property
    def margins(self):
        return self._margins.copy()

    @property
    def grid_shape(self):
        return self._grid_shape

    @property
    def figsize(self):
        return self._figsize

    @property
    def single_axis_shape(self):
        return self._single_axis_shape

    @property
    def middle_spacing(self):
        return self._middle_spacing

    @property
    def grid_spacing(self):
        return self._grid_spacing

    @property
    def grid_axis_shape(self):
        grid_width = (
            self._figsize.width
            - self._margins.width
            - self._single_axis_shape.width
            - self._middle_spacing
            - self._grid_spacing.horizontal * (self._grid_shape.n_cols - 1)
        ) / self._grid_shape.n_cols

        grid_height = (
            self._figsize.height
            - self._margins.height
            - self._grid_spacing.vertical * (self._grid_shape.n_rows - 1)
        ) / self._grid_shape.n_rows

        return FloatShape(grid_width, grid_height)

    @property
    def grid_total_size(self):
        grid_width = (
            self._figsize.width
            - self._margins.width
            - self._single_axis_shape.width
            - self._middle_spacing
        )
        grid_height = self._figsize.height - self._margins.height
        return FloatShape(width=grid_width, height=grid_height)

    @classmethod
    def from_solve(
        cls,
        grid_shape: IntShape | Tuple[float, float],
        fig_width=None,
        fig_height=None,
        margins_left=None,
        margins_right=None,
        margins_top=None,
        margins_bottom=None,
        single_axis_width=None,
        single_axis_height=None,
        grid_axis_width=None,
        grid_axis_height=None,
        grid_horizontal_spacing=None,
        grid_vertical_spacing=None,
        middle_spacing=None,
        single_axis_aspect=None,
        grid_axis_aspect=None,
        grid_spacings_equal=True,
        all_spacings_equal=False,
        margin_left_right_equal=False,
    ):
        # 0  fig_width
        # 1  fig_height
        # 2  margins_left
        # 3  margins_right
        # 4  margins_top
        # 5  margins_bottom
        # 6  single_axis_width
        # 7  single_axis_height
        # 8  grid_axis_width
        # 9  grid_axis_height
        # 10 grid_horizontal_spacing
        # 11 grid_vertical_spacing
        # 12 middle_spacing
        grid_axis_aspect = extent_to_aspect_if_needed(grid_axis_aspect)
        single_axis_aspect = extent_to_aspect_if_needed(single_axis_aspect)
        grid_shape = IntShape(grid_shape[0], grid_shape[1])

        system_matrix_rows = []
        if fig_width is not None:
            system_matrix_rows.append(
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, fig_width]
            )
        if fig_height is not None:
            system_matrix_rows.append(
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, fig_height]
            )
        if margins_left is not None:
            system_matrix_rows.append(
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, margins_left]
            )
        if margins_right is not None:
            system_matrix_rows.append(
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, margins_right]
            )
        if margins_top is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, margins_top]
            )
        if margins_bottom is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, margins_bottom]
            )
        if single_axis_width is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, single_axis_width]
            )
        if single_axis_height is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, single_axis_height]
            )

        if grid_axis_width is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, grid_axis_width]
            )
        if grid_axis_height is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, grid_axis_height]
            )
        if grid_horizontal_spacing is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, grid_horizontal_spacing]
            )
        if grid_vertical_spacing is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, grid_vertical_spacing]
            )
        if middle_spacing is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, middle_spacing]
            )

        # ======================================================================
        # Ensure correct aspect ratios
        # ======================================================================
        if single_axis_aspect is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 0, 0, -single_axis_aspect, 1, 0, 0, 0, 0, 0, 0]
            )
        if grid_axis_aspect is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 0, 0, 0, 0, -grid_axis_aspect, 1, 0, 0, 0, 0]
            )

        # ======================================================================
        # Further constraints
        # ======================================================================
        if grid_spacings_equal is not None:
            system_matrix_rows.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0])

        if all_spacings_equal:
            system_matrix_rows.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0])
            system_matrix_rows.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0])

        if margin_left_right_equal:
            system_matrix_rows.append([0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # Ensure the height is equal to the single axis height + margins
        system_matrix_rows.append([0, -1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0])

        # Ensure the total grid height is equal to the single axis height
        system_matrix_rows.append(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -1,
                0,
                grid_shape.n_rows,
                0,
                grid_shape.n_rows - 1,
                0,
                0,
            ]
        )
        system_matrix_rows.append(
            [
                -1,
                0,
                1,
                1,
                0,
                0,
                1,
                0,
                grid_shape.n_cols,
                0,
                grid_shape.n_cols - 1,
                0,
                1,
                0,
            ]
        )

        system_matrix_rows.append([0, -1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0])

        system_matrix_rows = [np.array(row) for row in system_matrix_rows]
        system_matrix = np.vstack(system_matrix_rows)
        target_vector = system_matrix[:, -1]
        coeff_matrix = system_matrix[:, :-1]
        # Check if there is a solution
        if np.linalg.matrix_rank(coeff_matrix) < np.linalg.matrix_rank(
            np.column_stack((coeff_matrix, target_vector))
        ):
            print(
                "No solution found for the given constraints. Providing least squares solution."
            )
            solution = np.linalg.lstsq(coeff_matrix, target_vector, rcond=None)[0]
        else:
            solution = np.linalg.lstsq(coeff_matrix, target_vector, rcond=None)[0]

            if np.any(solution < 0):
                print("Negative value found in solution for the given constraints.")

        return cls(
            margins=Margins(
                left=solution[2],
                right=solution[3],
                top=solution[4],
                bottom=solution[5],
            ),
            figsize=FloatShape(width=solution[0], height=solution[1]),
            grid_shape=grid_shape,
            grid_spacing=Spacing(horizontal=solution[10], vertical=solution[11]),
            middle_spacing=solution[12],
            single_axis_shape=FloatShape(width=solution[6], height=solution[7]),
        )

    def initialize_figure(self):
        fig = MPLFigure(figsize=self.figsize)
        grid_total_width = (
            self.figsize.width
            - self.margins.width
            - self.single_axis_shape.width
            - self.middle_spacing
        )
        grid_total_height = self.figsize.height - self.margins.height

        axes_grid = fig.add_axes_grid(
            n_rows=self.grid_shape.n_rows,
            n_cols=self.grid_shape.n_cols,
            x=self.margins.left + self.single_axis_shape.width + self.middle_spacing,
            y=self.margins.top,
            width=(
                grid_total_width
                - (self.grid_spacing.horizontal * (self.grid_shape.n_cols - 1))
            )
            / self.grid_shape.n_cols,
            height=(
                grid_total_height
                - (self.grid_spacing.vertical * (self.grid_shape.n_rows - 1))
            )
            / self.grid_shape.n_rows,
            spacing=self.grid_spacing,
        )

        ax_single = fig.add_ax(
            x=self.margins.left,
            y=self.margins.top,
            width=self.single_axis_shape.width,
            height=self.single_axis_shape.height,
        )

        return fig, axes_grid, ax_single

    def quickfig(self, grid_on_right: bool = True) -> Tuple[MPLFigure, Axes, AxesArray]:
        """Create a quick figure with single beside grid dimensions.

        Args:
            grid_on_right (bool, optional): Whether the grid should be on the right side of the single axis. Defaults to True.

        Returns:
            MPLFigure: The created figure.
            Axes: The created single axis.
            AxesArray: The created grid axes.
        """
        dimensions = self

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


def _any_is_none(iterable):
    return any(x is None for x in iterable)
