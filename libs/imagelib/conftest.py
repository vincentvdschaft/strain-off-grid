import numpy as np
from pytest import fixture

from imagelib import Image
from imagelib.extent import Extent


@fixture
def fixture_image_data():
    """Produces a random 2D array."""
    return np.random.rand(100, 100)


@fixture
def fixture_limits():
    """Produces a limits specification: one (min, max) pair per dimension."""
    return [(-1, 1), (0, 3)]


@fixture
def fixture_extent():
    """Produces a legacy Extent object, used to test backward-compatible HDF5 loading."""
    return Extent((-1, 1, 0, 3))


@fixture
def fixture_extent_tuple():
    """Produces a size 4 tuple."""
    return (-1, 1, 0, 3)


@fixture
def fixture_metadata():
    """Produces a metadata object."""
    return {
        "name": "test",
        "date": "2020-01-01",
        "n_samples": 3,
        "names": ["a", "b", "c"],
    }


@fixture
def fixture_image():
    """Produces an Image object."""
    return Image(array=np.random.rand(100, 100), limits=[(-1, 1), (0, 3)])


@fixture
def fixture_image_with_metadata(fixture_image_data, fixture_limits, fixture_metadata):
    """Produces an Image object with metadata."""
    return Image(
        array=fixture_image_data, limits=fixture_limits, metadata=fixture_metadata
    )


@fixture
def fixture_list_of_images(fixture_image_data, fixture_limits):
    """Produces a list of Image objects."""
    list = []
    for n in range(5):
        fixture_image_data += np.ones(fixture_image_data.shape)
        list.append(Image(array=fixture_image_data, limits=fixture_limits))

    return list


@fixture
def fixture_image_sequence(fixture_list_of_images):
    """Produces an ImageSequence object."""
    return Image(
        array=np.stack(fixture_list_of_images),
        limits=[(0, len(fixture_list_of_images) - 1), *fixture_list_of_images[0].limits],
    )
