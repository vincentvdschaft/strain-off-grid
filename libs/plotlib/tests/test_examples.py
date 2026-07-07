from pathlib import Path

import pytest
import matplotlib.pyplot as plt
import numpy as np

from plotlib import *


def _create_example_lineplot(title, x_vals, y_vals):
    fig = MPLFigure()
    ax = fig.add_ax(0, 0, 4, 2)

    for curve in y_vals:
        ax.plot(x_vals, curve)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage [V]")
    ax.set_title(title)
    ax.legend(loc="upper right")

    return fig


def test_example_lineplots(tmpdir, fixture_x_vals, fixture_y_vals):

    for style in ALLOWED_STYLES:
        use_style(style)
        fig = _create_example_lineplot(
            f"Style: {STYLE_NAMES[style]}", fixture_x_vals, fixture_y_vals
        )
        fig.savefig(
            tmpdir / f"style_{STYLE_NAMES[style]}.png",
            bbox_inches="tight",
            dpi=600,
        )


# if __name__ == "__main__":
# test_example_lineplots(Path("."),
