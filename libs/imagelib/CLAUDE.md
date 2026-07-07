# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_image.py

# Run a single test
pytest tests/test_image.py::test_initialize_image

# Install the package in editable mode (from .venv)
.venv/bin/pip install -e .
```

The virtual environment is at `.venv/` (Python 3.12). Activate with `source .venv/bin/activate` or prefix commands with `.venv/bin/`.

## Architecture

`imagelib` is a Python library providing an `NDImage` class — a numpy array bundled with physical spatial limits metadata, so dimensions stay synchronized through transformations. The API is dimension-agnostic: methods take a `dim` index rather than having per-axis variants (no `xflip`, `x0`, etc.) — this keeps the same code working regardless of how many dimensions an image has or what they represent.

### Core classes

**`Limits`** / **`LimitsND`** (`imagelib/extent.py`) — `Limits` is a `(min, max)` pair for one dimension (auto-sorted on construction). `LimitsND` wraps a list of `Limits`, one per array dimension, and accepts flexible input (list of pairs, flat tuple, `(N, 2)` array, or another `LimitsND`). Indexed by dimension: `limits[dim].min` / `.max` / `.size()`.

**`Extent`** (`imagelib/extent.py`) — legacy flat-tuple encoding `(dim0_min, dim0_max, dim1_min, dim1_max, ...)`. Kept only to decode HDF5 files written before the switch to `LimitsND`; not part of the public API.

**`NDImage`** (`imagelib/ndimage.py`) — the main class, exposed as `Image` from the package. Wraps a numpy array with a `LimitsND`. Key design principles:
- All operations return a new `NDImage` (immutable-style)
- Implements `__array_ufunc__` and `__array_function__` so numpy ufuncs (e.g. `np.sin(image)`) work transparently, preserving limits
- Slicing via `__getitem__` recomputes limits from the physical coordinate grid, so `image[:65]` correctly updates the limits
- `flip(dim)` flips the array along a dimension without changing its limits
- `extent_imshow` returns the `(x0, x1, y0, y1)` tuple for the *last two* dimensions, adjusted by half a pixel, for use with `matplotlib.imshow` (which treats extent as pixel edges, not centers)

**Per-axis metadata (`limits`, `labels`, `units`).** Each carries exactly one entry per array dimension (invariant enforced at construction). They travel together through every transformation: integer indexing drops the entry, `np.newaxis` inserts an empty one, `transpose` permutes them, and shape-preserving ops (ufuncs, arithmetic, `log_compress`, …) pass them through unchanged. All three are persisted to and restored from HDF5. This threading is centralized: axis-preserving methods build results via `self._rewrap(array, limits=None)` (or the thin `with_array` / `with_limits` wrappers), and axis-restructuring uses `select_axis_values_after_slicing` (extent.py) so labels/units follow the exact same drop/insert logic as limits. When adding a method that returns a new image, use `_rewrap` rather than calling `NDImage(...)` directly, or the labels/units will be silently reset to defaults.

**`saving.py`** — HDF5 serialization via h5py. Saves limits under the `limits` attribute; loading falls back to the legacy `extent` attribute (via `Extent`) for old files. Supports nested dict metadata with list-to-numbered-dict round-trip encoding.

### Public API (from `imagelib import *`)

- `Image` — the main class (`NDImage`)
- `save_hdf5_image`, `load_hdf5_image`, `check_hdf5_image_hash` — standalone HDF5 I/O

### Coordinate convention

Array axes are ordered zyx: the last axis is x, the second-to-last is y, and any leading axes are additional (e.g. z, time). Default dimension labels reflect this (`labels[-1] == "x"`, `labels[-2] == "y"`). When plotting a 2D image with `imshow`, use `image.array.T` and `extent=image.extent_imshow` with `origin="lower"`.
