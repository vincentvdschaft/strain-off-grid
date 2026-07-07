import dataclasses

import numpy as np
import pytest

from imagelib import Limits, LimitsND
from imagelib.extent import Extent


def test_limits_sorts_min_and_max():
    """Constructing with min > max swaps them so min <= max always holds."""
    limits = Limits(10, 5)
    assert limits.min == 5
    assert limits.max == 10


def test_limits_coerces_to_float():
    """Integer inputs are stored as floats."""
    limits = Limits(0, 10)
    assert isinstance(limits.min, float)
    assert isinstance(limits.max, float)


def test_limits_size():
    """size() returns max - min."""
    assert Limits(2, 7).size() == 5


def test_limits_iter():
    """Unpacking a Limits yields (min, max)."""
    lo, hi = Limits(2, 7)
    assert (lo, hi) == (2, 7)


def test_limits_repr():
    """repr() shows the class name and both values."""
    assert repr(Limits(2, 7)) == "Limits(2.0, 7.0)"


def test_limits_equality():
    """Limits compares equal to another Limits or a plain (min, max) pair."""
    assert Limits(2, 7) == Limits(2, 7)
    assert Limits(2, 7) == (2, 7)
    assert Limits(2, 7) != Limits(2, 8)


def test_limits_hash_consistent_with_equality():
    """Equal Limits objects hash the same, so they work correctly in sets/dicts."""
    assert hash(Limits(2, 7)) == hash(Limits(2, 7))
    assert len({Limits(2, 7), Limits(2, 7)}) == 1


def test_limits_is_immutable():
    """Limits is a frozen dataclass; attribute assignment must fail."""
    limits = Limits(2, 7)
    with pytest.raises(dataclasses.FrozenInstanceError):
        limits.min = 0


def test_limits_addition_builds_limits_nd():
    """Adding two Limits produces a LimitsND combining both."""
    result = Limits(0, 10) + Limits(5, 15)
    assert isinstance(result, LimitsND)
    assert result.limits == (Limits(0, 10), Limits(5, 15))


def test_limits_nd_addition():
    limits1 = Limits(0, 10)
    limits2 = Limits(5, 15)
    limits3 = Limits(20, 30)

    nd1 = LimitsND((limits1, limits2))
    nd2 = LimitsND((limits3,))

    result = nd1 + nd2

    assert isinstance(result, LimitsND)
    assert result.limits == (limits1, limits2, limits3)


def test_limits_nd_from_limits_objects():
    """LimitsND accepts a sequence of Limits objects directly."""
    nd = LimitsND([Limits(-1, 1), Limits(0, 3)])
    assert nd.limits == (Limits(-1, 1), Limits(0, 3))


def test_limits_nd_from_pairs():
    """LimitsND accepts a list of (min, max) pairs."""
    nd = LimitsND([(-1, 1), (0, 3)])
    assert nd.limits == (Limits(-1, 1), Limits(0, 3))


def test_limits_nd_from_flat_tuple():
    """LimitsND accepts a flat (2N,) tuple."""
    nd = LimitsND((-1, 1, 0, 3))
    assert nd.limits == (Limits(-1, 1), Limits(0, 3))


def test_limits_nd_from_array():
    """LimitsND accepts an (N, 2) numpy array."""
    nd = LimitsND(np.array([[-1, 1], [0, 3]]))
    assert nd.limits == (Limits(-1, 1), Limits(0, 3))


def test_limits_nd_from_another_limits_nd():
    """LimitsND can be constructed from another LimitsND (copy-like)."""
    original = LimitsND([(-1, 1), (0, 3)])
    copy = LimitsND(original)
    assert copy.limits == original.limits


def test_limits_nd_rejects_odd_length_flat_input():
    """An odd number of flat elements cannot be paired into (min, max)."""
    with pytest.raises(ValueError):
        LimitsND((-1, 1, 0))


def test_limits_nd_rejects_wrong_shape():
    """Anything that isn't (N, 2) or flat (2N,) is rejected."""
    with pytest.raises(ValueError):
        LimitsND(np.array([[-1, 1, 2], [0, 3, 4]]))


def test_limits_nd_is_immutable():
    """LimitsND is a frozen dataclass; attribute assignment must fail."""
    nd = LimitsND([(-1, 1), (0, 3)])
    with pytest.raises(dataclasses.FrozenInstanceError):
        nd.limits = ()


def test_limits_nd_ndim():
    """ndim reflects the number of Limits entries."""
    assert LimitsND([(-1, 1), (0, 3)]).ndim == 2


def test_limits_nd_getitem():
    """Indexing returns the Limits for that dimension."""
    nd = LimitsND([(-1, 1), (0, 3)])
    assert nd[0] == Limits(-1, 1)
    assert nd[1] == Limits(0, 3)


def test_limits_nd_sizes():
    """sizes() returns an array of per-dimension sizes."""
    nd = LimitsND([(-1, 1), (0, 3)])
    np.testing.assert_array_equal(nd.sizes(), [2, 3])


def test_limits_nd_from_extent():
    """from_extent decodes a legacy flat Extent into per-dimension Limits."""
    extent = Extent((-1, 1, 0, 3))
    nd = LimitsND.from_extent(extent)
    assert nd.limits == (Limits(-1, 1), Limits(0, 3))


def test_limits_nd_from_shape():
    """from_shape gives each dimension limits (0, size - 1)."""
    nd = LimitsND.from_shape((4, 5))
    assert nd.limits == (Limits(0, 3), Limits(0, 4))


def test_limits_nd_iter():
    """Iterating a LimitsND yields its Limits in order."""
    nd = LimitsND([(-1, 1), (0, 3)])
    assert list(nd) == [Limits(-1, 1), Limits(0, 3)]


def test_limits_nd_len():
    """len() reflects the number of dimensions."""
    assert len(LimitsND([(-1, 1), (0, 3)])) == 2


def test_limits_nd_origin():
    """origin() returns the min value of each dimension as an array."""
    nd = LimitsND([(-1, 1), (0, 3)])
    np.testing.assert_array_equal(nd.origin(), [-1, 0])


def test_limits_nd_hash_consistent_with_equality():
    """Equal LimitsND objects hash the same."""
    nd1 = LimitsND([(-1, 1), (0, 3)])
    nd2 = LimitsND([(-1, 1), (0, 3)])
    assert hash(nd1) == hash(nd2)


def test_limits_nd_make_grid():
    """make_grid builds a coordinate grid with shape (*shape, ndim)."""
    nd = LimitsND([(0, 1), (0, 2)])
    grid = nd.make_grid((2, 3))

    assert grid.shape == (2, 3, 2)
    np.testing.assert_allclose(grid[:, 0, 0], [0, 1])
    np.testing.assert_allclose(grid[0, :, 1], [0, 1, 2])


def test_limits_nd_make_grid_rejects_mismatched_shape():
    """make_grid requires one size per dimension."""
    nd = LimitsND([(0, 1), (0, 2)])
    with pytest.raises(ValueError):
        nd.make_grid((2, 3, 4))


def test_limits_nd_fitted_to_pixel_size_scalar():
    """A single pixel size is applied to every dimension."""
    nd = LimitsND([(0, 10), (0, 10)])
    fitted = nd.fitted_to_pixel_sizes(3.0)
    assert fitted.limits == (Limits(0, 9), Limits(0, 9))


def test_limits_nd_fitted_to_pixel_size_per_dimension():
    """A sequence of pixel sizes is applied one per dimension."""
    nd = LimitsND([(0, 10), (0, 9)])
    fitted = nd.fitted_to_pixel_sizes([3.0, 2.0])
    assert fitted.limits == (Limits(0, 9), Limits(0, 8))


def test_limits_nd_fitted_to_pixel_size_rejects_mismatched_length():
    """The pixel size sequence must have one entry per dimension."""
    nd = LimitsND([(0, 10), (0, 9)])
    with pytest.raises(ValueError):
        nd.fitted_to_pixel_sizes([3.0, 2.0, 1.0])
