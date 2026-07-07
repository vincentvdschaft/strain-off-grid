# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run a single test
uv run pytest tests/test_examples.py::test_example_lineplots

# Build the package
uv build
```

## Architecture

`plotlib` is a matplotlib wrapper for creating publication-quality figures with **inch-based, top-left-origin coordinates**. This is the key design difference from matplotlib's normalized/bottom-left convention.

### Coordinate system

All axes positions are specified in inches with **(0, 0) at the top-left** of the figure. `MPLFigure` converts to matplotlib's normalized bottom-left coordinates internally via `bbox_norm()`.

### Core modules

- **`plotlib.py`** — `MPLFigure` class: the main API. Wraps a matplotlib figure and exposes `add_ax(x, y, width, height)`, `add_axes_grid()`, `add_colorbar()`, `add_legend()`, `add_arrow()`, etc., all using inch coordinates.
- **`styles.py`** — `use_style(STYLE_*)` applies one of four `.mplstyle` files (`light`, `dark`, `paper`, `poster`). Always resets to `rcParamsDefault` before applying a new style.
- **`constants.py`** — `IEEE_COLUMN_WIDTH = 3.5`, `IEEE_DOUBLE_COLUMN_WIDTH = 7.16`, and style constants.
- **`animation.py`** — Easing/interpolation helpers (`smooth`, `map_range`, `smooth_range`) for animations.
- **`boxconnection.py`** — Computes connecting line segments between two bounding boxes using convex hull.
- **`quicksfigs.py`** — Convenience factory functions (`quickfig_single`, `quickfig_grid`, `quickfig_single_besides_grid`) that take `Dimensions*` dataclasses and return `(fig, ax)` or `(fig, axes)`.
- **`dimensions.py`** — `DimensionsSingle`, `DimensionsGrid`, `DimensionsSingleBesidesGrid`, `Margins`, `Spacing`, `FloatShape`, `IntShape` dataclasses for declarative layout specification. (Imported by `__init__.py`; this file is under active development.)

### Typical usage pattern

```python
from plotlib import MPLFigure, use_style, STYLE_PAPER

use_style(STYLE_PAPER)
fig = MPLFigure(figsize=(3.5, 2.5))
ax = fig.add_ax(x=0.5, y=0.4, width=2.8, height=1.8)
ax.plot(x, y)
fig.savefig("output.pdf", bbox_inches="tight")
```

### Style files

Located in `src/plotlib/styles/`. The `paper` style uses serif fonts, compact tick/label sizes, and is tuned for IEEE column-width figures.
