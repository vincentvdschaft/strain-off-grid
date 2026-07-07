import pytest

from imagelib.extent import Extent


def test_extent_initialize_from_tuple(fixture_extent_tuple):
    """Initializes an extent object from a tuple."""
    extent = Extent(fixture_extent_tuple)
    assert extent == fixture_extent_tuple


@pytest.mark.parametrize(
    "input_tuple, expected",
    [
        ((-1, 1, 0, 2), (-1, 1, 0, 2)),
        ((1, -1, 2, 0), (-1, 1, 0, 2)),
        ((1, 1, 2, 2), (1, 1, 2, 2)),
        ((1, -1, 0, 2), (-1, 1, 0, 2)),
        ((1, -1, 2, 0), (-1, 1, 0, 2)),
    ],
)
def test_extent_sort(input_tuple, expected):
    """Tests the sort method of the extent object."""
    extent = Extent(input_tuple)
    assert extent.sort() == expected


def test_extent_start_end(fixture_extent):
    """Tests the start/end accessors of the extent object."""
    assert fixture_extent.start(0) == fixture_extent[0]
    assert fixture_extent.end(0) == fixture_extent[1]
    assert fixture_extent.start(1) == fixture_extent[2]
    assert fixture_extent.end(1) == fixture_extent[3]


def test_extent_ndim(fixture_extent):
    """Tests the ndim property of the extent object."""
    assert fixture_extent.ndim == 2
