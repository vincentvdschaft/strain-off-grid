import numpy as np

from ..plotlib import MPLFigure
from .aspect import extent_to_aspect_if_needed
from .margins import Margins
from .shape import FloatShape


class DimensionsSingle:
    def __init__(self, margins: Margins, figsize: FloatShape | tuple[float, float]):
        assert isinstance(margins, Margins)

        self._margins = margins
        self._figsize = FloatShape(figsize[0], figsize[1])

    @property
    def margins(self):
        return self._margins.copy()

    @property
    def figsize(self):
        return self._figsize

    @property
    def axis_size(self):
        axis_width = self._figsize[0] - self._margins.width
        axis_height = self._figsize[1] - self._margins.height
        return FloatShape(axis_width, axis_height)

    @classmethod
    def from_no_height(cls, margins, fig_width, axis_aspect):
        axis_width = fig_width - margins.width
        axis_height = axis_width * axis_aspect
        figsize = FloatShape(fig_width, axis_height + margins.height)
        return cls(margins, figsize)

    @classmethod
    def from_no_width(cls, margins, fig_height, axis_aspect):
        axis_height = fig_height - margins.height
        axis_width = axis_height / axis_aspect
        figsize = FloatShape(axis_width + margins.width, fig_height)
        return cls(margins, figsize)

    @classmethod
    def from_solve(
        cls,
        fig_width=None,
        fig_height=None,
        margins_left=None,
        margins_right=None,
        margins_top=None,
        margins_bottom=None,
        axis_aspect=None,
    ):
        axis_aspect = extent_to_aspect_if_needed(axis_aspect)
        system_matrix_rows = []
        row_ax_width = np.array([1, 0, -1, -1, 0, 0, 0])
        row_ax_height = np.array([0, 1, 0, 0, -1, -1, 0])

        if fig_width is not None:
            system_matrix_rows.append([1, 0, 0, 0, 0, 0, fig_width])
        if fig_height is not None:
            system_matrix_rows.append([0, 1, 0, 0, 0, 0, fig_height])
        if margins_left is not None:
            system_matrix_rows.append([0, 0, 1, 0, 0, 0, margins_left])
        if margins_right is not None:
            system_matrix_rows.append([0, 0, 0, 1, 0, 0, margins_right])
        if margins_top is not None:
            system_matrix_rows.append([0, 0, 0, 0, 1, 0, margins_top])
        if margins_bottom is not None:
            system_matrix_rows.append([0, 0, 0, 0, 0, 1, margins_bottom])
        if axis_aspect is not None:
            system_matrix_rows.append(row_ax_width * -axis_aspect + row_ax_height)

        system_matrix_rows = [np.array(row) for row in system_matrix_rows]
        system_matrix = np.vstack(system_matrix_rows)
        target_vector = system_matrix[:, -1]
        coeff_matrix = system_matrix[:, :-1]

        rank = np.linalg.matrix_rank(coeff_matrix)
        if rank < coeff_matrix.shape[1]:
            print("Underdetermined system: Consider providing more constraints.")
        elif rank > coeff_matrix.shape[1]:
            print(
                "Overdetermined system: No exact solution found for the given "
                "constraints. Providing least squares solution."
            )

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
        )

    def initialize_figure(self):
        fig = MPLFigure(figsize=self.figsize)
        ax = fig.add_ax(
            x=self.margins.left,
            y=self.margins.top,
            width=self.figsize.width - self.margins.left - self.margins.right,
            height=self.figsize.height - self.margins.top - self.margins.bottom,
        )
        return fig, ax

    def __repr__(self):
        return f"DimensionsSingle(margins={self.margins}, figsize={self.figsize})"
