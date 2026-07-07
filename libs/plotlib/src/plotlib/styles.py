import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from plotlib.constants import *


def _get_style_dir():
    return Path(__file__).parent / "styles"


def use_dark_style():
    """Available for backward compatibility. Use `use_style(STYLE_DARK)` instead."""
    path = _get_style_dir() / "dark.mplstyle"
    plt.style.use(str(path))

    print("use_dark_style() is deprecated. Use use_style(STYLE_DARK) instead.")


def use_style(style):
    """Set the style of the plots.

    Examples
    --------
    ```python
    use_style(STYLE_DARK)
    ```
    """
    if style not in ALLOWED_STYLES:
        raise ValueError(f"Unknown style {style}. Choose from {ALLOWED_STYLES}")

    # Get style file
    filename = STYLE_NAMES[style] + ".mplstyle"
    path = _get_style_dir() / filename

    # Reset to default style
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    # Apply new style
    plt.style.use(str(path))
