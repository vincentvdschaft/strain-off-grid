from pathlib import Path

import h5py
import numpy as np
import pytest

from imagelib import Image


def _dict_equal(dict1, dict2):
    for key, value1 in dict1.items():
        value2 = dict2[key]
        if isinstance(value1, dict):
            if not _dict_equal(value1, value2):
                return False
        elif isinstance(value1, np.ndarray):
            if not np.allclose(value1, value2):
                return False
        elif value1 != value2:
            return False
    return True


def test_initialize_image(fixture_image_data, fixture_limits):
    """Initializes an image object."""
    image = Image(array=fixture_image_data, limits=fixture_limits)
    assert np.allclose(image.array, fixture_image_data)
    for dim, (lo, hi) in enumerate(fixture_limits):
        assert image.limits[dim].min == lo
        assert image.limits[dim].max == hi
    assert image.shape == fixture_image_data.shape


def test_mul_scalar(fixture_image):
    """Tests multiplication of an image by a scalar."""
    image = fixture_image * 2
    assert np.allclose(image.array, fixture_image.array * 2)


def test_mul_ndarray(fixture_image):
    """Tests multiplication of an image by an ndarray."""
    multiplier = np.arange(fixture_image.size).reshape(fixture_image.shape)
    image = fixture_image * multiplier
    print(image)
    assert np.allclose(fixture_image.array * multiplier, image.array), "Data not equal."


def test_mul_image(fixture_image):
    """Tests multiplication of an image by an ndarray."""
    multiplier = np.arange(fixture_image.size).reshape(fixture_image.shape)
    image = fixture_image * multiplier
    assert isinstance(image, Image), "Not an Image object."
    assert np.allclose(fixture_image.array * multiplier, image.array), "Data not equal."


def test_rmul_scalar(fixture_image):
    """Tests multiplication of a scalar by an image."""
    image = 2 * fixture_image
    assert np.allclose(image.array, fixture_image.array * 2)


def test_rmul_ndarray(fixture_image):
    """Tests multiplication of an ndarray by an image."""
    multiplier = np.arange(fixture_image.size).reshape(fixture_image.shape)
    image = multiplier * fixture_image
    assert np.allclose(fixture_image.array * multiplier, image.array), "Data not equal."


def test_rmul_image(fixture_image):
    """Tests multiplication of an image by an image."""
    image = fixture_image * fixture_image
    assert isinstance(image, Image), "Not an Image object."
    assert np.allclose(fixture_image.array * fixture_image.array, image.array), (
        "Data not equal."
    )


def test_add(fixture_image):
    """Tests addition of an image by a scalar."""
    image = fixture_image + 2
    assert np.allclose(image.array, fixture_image.array + 2)


def test_sub(fixture_image):
    """Tests subtraction of an image by a scalar."""
    image = fixture_image - 2
    assert np.allclose(image.array, fixture_image.array - 2)


def test_resample(fixture_image):
    """Tests resampling of an image."""
    new_shape = (20, 50)
    new_limits = [(0, 1), (0, 1)]
    image = fixture_image.resample(shape=new_shape, limits=new_limits, method="nearest")
    assert image.shape == new_shape
    assert image.limits[0].min == 0 and image.limits[0].max == 1
    assert image.limits[1].min == 0 and image.limits[1].max == 1


def test_square_pixels(fixture_image):
    """Tests the square_pixels method."""
    assert fixture_image.pixel_size(0) != fixture_image.pixel_size(1)
    image = fixture_image.square_pixels()
    assert image.pixel_size(0) == image.pixel_size(1)


@pytest.mark.parametrize("shape", [(1, 100), (100, 1)])
def test_size_one_nonzero_width(shape, fixture_limits):
    """Ensures that an error is raised when a dimension has size 1 but non-zero width."""

    with pytest.raises(ValueError):
        Image(array=np.ones(shape), limits=fixture_limits)


@pytest.mark.parametrize(
    "slice_x, slice_y",
    [
        (0, slice(5, 10, None)),
        (slice(5, 10, None), 0),
        (slice(55, 80, None), slice(5, 10, None)),
        (slice(-10, -1, None), slice(-10, -1, None)),
    ],
)
def test_getitem_slice(fixture_image, slice_x, slice_y):
    """Tests slicing of an image."""

    image = fixture_image[slice_x, slice_y]
    data_sliced_by_numpy = fixture_image.array[slice_x, slice_y].reshape(image.shape)
    assert np.allclose(image.array, data_sliced_by_numpy)
    assert image.limits != fixture_image.limits

    expected_limits = []
    for dim, key in enumerate((slice_x, slice_y)):
        if isinstance(key, int):
            continue
        coords = fixture_image.vals(dim)[key]
        expected_limits.append((coords[0], coords[-1]))
    for dim, limits in enumerate(expected_limits):
        assert np.allclose(
            (image.limits[dim].min, image.limits[dim].max), sorted(limits)
        )


def test_limits_after_slicing():
    """Tests the limits of an image after slicing."""
    array = np.random.rand(101, 101)
    limits = [(-5, 5), (0, 2)]
    image = Image(array=array, limits=limits)
    image_sliced = image[0:51, 0:51]
    assert image_sliced.limits[0].min == -5 and image_sliced.limits[0].max == 0
    assert image_sliced.limits[1].min == 0 and image_sliced.limits[1].max == 1


def test_save_hdf5(fixture_image_with_metadata, tmp_path):
    """Tests saving an image to an HDF5 file."""
    fixture_image_with_metadata.save(tmp_path / "test.hdf5")
    image = Image.load(tmp_path / "test.hdf5")
    assert np.allclose(image.array, fixture_image_with_metadata.array)
    assert image.limits == fixture_image_with_metadata.limits
    assert _dict_equal(image.metadata, fixture_image_with_metadata.metadata)


def test_load_legacy_extent_format(tmp_path):
    """Tests that HDF5 files written with the old 'extent' attribute still load."""
    array = np.random.rand(10, 10)
    path = tmp_path / "legacy.hdf5"
    with h5py.File(path, "w") as dataset:
        dataset.create_dataset("image", data=array)
        dataset["image"].attrs["extent"] = np.array([-1.0, 1.0, 0.0, 3.0])

    image = Image.load(path)
    assert np.allclose(image.array, array)
    assert image.limits[0].min == -1 and image.limits[0].max == 1
    assert image.limits[1].min == 0 and image.limits[1].max == 3


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".jpeg", ".bmp"])
def test_save_image_format(fixture_image, tmp_path, suffix):
    """Tests saving an image to an image file."""
    save_path = tmp_path / f"test{suffix}"
    fixture_image.save_png(save_path)
    assert Path(save_path).exists()


@pytest.mark.parametrize("transform", [np.square, np.log, np.exp, np.sqrt])
def test_match_histograms(fixture_image, transform):
    """Tests matching histograms of an image by transforming it with a monotonic
    function and then matching back to the original image."""
    image_transformed = transform(fixture_image)
    data_matched = image_transformed.match_histogram(fixture_image)
    assert np.allclose(data_matched.array, fixture_image.array)


@pytest.mark.parametrize("key, value", [("name", "test"), ("number", 3)])
def test_add_metadata(fixture_image, key, value):
    """Tests appending metadata to an image."""
    image = fixture_image.add_metadata(key=key, value=value)
    assert key in image.metadata
    assert image.metadata[key] == value


def test_metadata_list_of_arrays_roundtrip(fixture_image, tmp_path):
    """Tests that a list of numpy arrays saved as metadata round-trips correctly."""
    arrays = [np.random.rand(5, 3) for _ in range(3)]
    image = fixture_image.add_metadata("frames", arrays)
    image.save(tmp_path / "test.hdf5")
    loaded = Image.load(tmp_path / "test.hdf5")
    assert isinstance(loaded.metadata["frames"], list)
    assert len(loaded.metadata["frames"]) == 3
    for original, recovered in zip(arrays, loaded.metadata["frames"]):
        assert np.allclose(original, recovered)


def test_append_metadata(fixture_image_with_metadata):
    """Tests appending metadata to an image."""
    fixture_image_with_metadata.append_metadata("new_key", "new_value")
    assert "new_key" in fixture_image_with_metadata.metadata
    assert isinstance(fixture_image_with_metadata.metadata["new_key"], list)
    assert fixture_image_with_metadata.metadata["new_key"][0] == "new_value"

    fixture_image_with_metadata.append_metadata("new_key", "second_value")
    assert "new_key" in fixture_image_with_metadata.metadata
    assert len(fixture_image_with_metadata.metadata["new_key"]) == 2


def test_grid(fixture_image):
    """Tests the grid method."""
    grid = fixture_image.grid
    assert grid.shape == (fixture_image.shape[0], fixture_image.shape[1], 2)
    assert np.min(grid[:, :, 0]) == fixture_image.limits[0].min
    assert np.max(grid[:, :, 0]) == fixture_image.limits[0].max
    assert np.min(grid[:, :, 1]) == fixture_image.limits[1].min
    assert np.max(grid[:, :, 1]) == fixture_image.limits[1].max


def test_flatgrid(fixture_image):
    """Tests the flatgrid method."""
    flatgrid = fixture_image.flatgrid
    assert flatgrid.shape == (fixture_image.size, 2)
    assert np.min(flatgrid[:, 0]) == fixture_image.limits[0].min
    assert np.max(flatgrid[:, 0]) == fixture_image.limits[0].max
    assert np.min(flatgrid[:, 1]) == fixture_image.limits[1].min
    assert np.max(flatgrid[:, 1]) == fixture_image.limits[1].max


def test_equal(fixture_image):
    """Tests the __eq__ method."""
    assert fixture_image == fixture_image
    assert fixture_image == fixture_image.copy()
    assert fixture_image != fixture_image + 1


def test_transpose(fixture_image):
    """Tests the transpose method."""
    image = fixture_image.transpose()
    assert image.shape == (fixture_image.shape[1], fixture_image.shape[0])
    assert fixture_image.transpose().transpose() == fixture_image


@pytest.mark.parametrize("dim", [0, 1])
def test_flip(fixture_image, dim):
    """Tests the flip method along each dimension."""
    image = fixture_image.flip(dim)
    expected = np.flip(fixture_image.array, axis=dim)
    assert np.allclose(image.array, expected)
    # Flipping should not change the limits
    assert image.limits == fixture_image.limits
    # Flipping twice should return the original image
    assert fixture_image.flip(dim).flip(dim) == fixture_image


def test_default_labels_and_units():
    """Default labels follow the zyx convention; units are one empty string per axis."""
    image = Image(np.zeros((3, 4, 5)))
    assert image.labels == ("dim_0", "y", "x")
    assert image.units == ("", "", "")


def test_labels_units_length_validated():
    """Labels and units must have one entry per dimension."""
    with pytest.raises(AssertionError):
        Image(np.zeros((3, 4)), labels=("only_one",))
    with pytest.raises(AssertionError):
        Image(np.zeros((3, 4)), units=("s", "s", "s"))


def test_labels_units_survive_ufunc_and_arithmetic():
    """Shape-preserving operations carry labels and units through unchanged."""
    image = Image(np.random.rand(3, 4), labels=("a", "b"), units=("s", "m"))
    for derived in (np.sin(image), image * 2, image + 1, image.abs(), image.copy()):
        assert derived.labels == ("a", "b")
        assert derived.units == ("s", "m")


def test_labels_units_after_integer_indexing():
    """Integer indexing drops the corresponding label and unit."""
    image = Image(
        np.random.rand(3, 4, 5), labels=("z", "y", "x"), units=("s", "m", "m")
    )
    sub = image[1]
    assert sub.labels == ("y", "x")
    assert sub.units == ("m", "m")


def test_labels_units_after_newaxis():
    """np.newaxis inserts an empty label and unit."""
    image = Image(np.random.rand(3, 4), labels=("y", "x"), units=("m", "m"))
    expanded = image[:, None, :]
    assert expanded.labels == ("y", "", "x")
    assert expanded.units == ("m", "", "m")


def test_labels_units_after_transpose():
    """Transpose reorders labels and units to match the permuted axes."""
    image = Image(
        np.random.rand(3, 4, 5), labels=("z", "y", "x"), units=("s", "m", "m")
    )
    transposed = image.transpose((2, 0, 1))
    assert transposed.labels == ("x", "z", "y")
    assert transposed.units == ("m", "s", "m")


def test_labels_units_hdf5_roundtrip(tmp_path):
    """Labels and units are persisted to and restored from HDF5."""
    image = Image(np.random.rand(3, 4), labels=("y", "x"), units=("m", "m"))
    image.save(tmp_path / "test.hdf5")
    loaded = Image.load(tmp_path / "test.hdf5")
    assert loaded.labels == ("y", "x")
    assert loaded.units == ("m", "m")


def test_labels_units_hdf5_sliced_load(tmp_path):
    """A sliced load restructures labels and units like the array."""
    image = Image(
        np.random.rand(3, 4, 5), labels=("z", "y", "x"), units=("s", "m", "m")
    )
    image.save(tmp_path / "test.hdf5")
    loaded = Image.load(tmp_path / "test.hdf5", indices=(1, slice(None), slice(None)))
    assert loaded.shape == (4, 5)
    assert loaded.labels == ("y", "x")
    assert loaded.units == ("m", "m")


def test_to_pixels(fixture_image):
    """Tests the to_pixels method."""
    image = fixture_image.to_pixels()
    assert image.max() == 1
    assert image.min() == 0


def test_log_compress(fixture_image):
    """Tests the log_compress method."""
    image = fixture_image.normalize().log_compress()
    assert image.max() == 0
    assert fixture_image.log_compress().log_expand() == fixture_image


def test_window(fixture_image):
    """Tests the get_window method with the (min, max)-per-dimension format."""
    window = fixture_image.get_window([(0, 1), (0, 1)])
    assert isinstance(window, Image)


def test_window_partial_dim0(fixture_image):
    """Slicing only dim 0 leaves dim 1 untouched."""
    window = fixture_image.get_window([(0, 1), None])
    assert window.shape[1] == fixture_image.shape[1]


def test_window_partial_dim1(fixture_image):
    """Slicing only dim 1 leaves dim 0 untouched."""
    window = fixture_image.get_window([None, (0, 1)])
    assert window.shape[0] == fixture_image.shape[0]


def test_window_zero_dim_size():
    """No division by zero when a dimension has zero spatial size."""
    img = Image(np.zeros((10, 1)), limits=[(-1, 1), (0.5, 0.5)])
    result = img.get_window([(-1, 0), (0.5, 0.5)])
    assert result is not None


def test_coordinates_to_indices(fixture_image):
    """Tests the coordinates_to_indices method."""
    x_coords = np.array([0.0, 0.5, fixture_image.limits[0].max])
    y_coords = np.array([1.0, 1.5, fixture_image.limits[1].max])
    coordinates = np.stack((x_coords, y_coords), axis=1)
    indices = fixture_image.coordinates_to_indices(coordinates)
    print(indices)
    assert indices.shape == (3, 2)
    assert np.allclose(
        indices[-1, :],
        np.array([fixture_image.shape[0] - 1, fixture_image.shape[1] - 1]),
    )
