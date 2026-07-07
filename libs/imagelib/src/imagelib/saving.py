from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np

from .extent import (
    Extent,
    LimitsND,
    LimitsNDInput,
    compute_limits_after_slicing,
    select_axis_values_after_slicing,
)


def save_hdf5_image(
    path,
    array,
    limits: LimitsNDInput,
    metadata=None,
    labels=None,
    units=None,
    group="/image",
    append=False,
):
    """
    Saves an image to an hdf5 file.

    Parameters
    ----------
    path : str
        The path to the hdf5 file.
    array : np.ndarray
        The image to save.
    limits : LimitsNDInput
        The spatial limits of the image, one (min, max) pair per dimension.
    metadata : dict
        Additional metadata to save.
    labels : sequence of str
        Per-dimension axis labels.
    units : sequence of str
        Per-dimension axis units.
    group : str
        The group in the hdf5 file where the image will be saved.
    """

    limits = LimitsND(limits)

    path = Path(path)

    if path.exists() and not append:
        path.unlink()
    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    file_mode = "a" if append else "w"
    with h5py.File(path, file_mode) as hdf5_file:
        hdf5_file.require_group(group)
        dataset = hdf5_file[group].create_dataset("image", data=array)
        dataset.attrs["limits"] = _limits_to_array(limits)
        if labels is not None:
            dataset.attrs["labels"] = list(labels)
        if units is not None:
            dataset.attrs["units"] = list(units)
        if metadata is not None:
            save_dict_to_hdf5(hdf5_file, metadata, parent_group=group)


def _limits_to_array(limits: LimitsND) -> np.ndarray:
    """Flatten LimitsND to (dim0_min, dim0_max, dim1_min, dim1_max, ...)."""
    flat = []
    for dim_limits in limits:
        flat.append(dim_limits.min)
        flat.append(dim_limits.max)
    return np.array(flat)


def load_hdf5_image(
    path,
    indices=slice(None),
    group="/",
):
    """
    Loads an image from an hdf5 file.

    Parameters
    ----------
    path : str
        The path to the hdf5 file.
    indices : slice
        The indices to load from the image.
    group : str
        The group in the hdf5 file where the image is saved.

    Returns
    -------
    image : NDImage
        The loaded image.
    """

    with h5py.File(path, "r") as hdf5_file:
        dataset = hdf5_file[group]["image"]
        attrs = dataset.attrs
        original_shape = dataset.shape
        array = dataset[indices]
        limits = compute_limits_after_slicing(
            current_shape=original_shape, limits=_read_limits(attrs), key=indices
        )
        labels = _read_axis_metadata(attrs, "labels", indices)
        units = _read_axis_metadata(attrs, "units", indices)
        metadata = load_hdf5_to_dict(hdf5_file, parent_group=group)
        metadata.pop("image", None)
    from .ndimage import NDImage

    return NDImage(
        array=array, limits=limits, metadata=metadata, labels=labels, units=units
    )


def _read_limits(attrs) -> LimitsND:
    """Reads limits from HDF5 attrs, falling back to the legacy 'extent' format."""
    if "limits" in attrs:
        return LimitsND(np.asarray(attrs["limits"]))
    return LimitsND.from_extent(Extent(attrs["extent"]))


def _read_axis_metadata(attrs, name, indices):
    """Reads a per-axis string attribute and restructures it for the slice."""
    if name not in attrs:
        return None
    values = [str(value) for value in attrs[name]]
    return select_axis_values_after_slicing(values, indices, "")


def save_dict_to_hdf5(hdf5_file, data_dict, parent_group="/"):
    """
    Recursively saves a nested dictionary to an HDF5 file.

    Parameters
    ----------
    hdf5_file : h5py.File
        Opened h5py.File object.
    data_dict : dict
        (Nested) dictionary to save.
    parent_group : h5py.Group
        Current group path in HDF5 file (default is root "/").
    """
    data_dict = deepcopy(data_dict)
    data_dict = _lists_to_numbered_dict(data_dict)
    for key, value in data_dict.items():
        group_path = f"{parent_group}/{key}"
        if isinstance(value, dict):
            # Create a new group for nested dictionary
            hdf5_file.require_group(group_path)
            save_dict_to_hdf5(hdf5_file, value, parent_group=group_path)
        else:
            if value is None:
                continue
            # Convert leaf items into datasets
            hdf5_file[group_path] = value


def _lists_to_numbered_dict(data_dict):
    """Transforms all lists in a dictionary to dictionaries with numbered keys."""
    for key, value in data_dict.items():
        if isinstance(value, list):
            data_dict[key] = {str(i).zfill(3): v for i, v in enumerate(value)}
        elif isinstance(value, dict):
            data_dict[key] = _lists_to_numbered_dict(value)
    return data_dict


def _is_numbered_dict(data_dict):
    keys = data_dict.keys()
    try:
        keys = [int(k) for k in keys]
    except ValueError:
        return False
    return set(keys) == set(range(len(keys)))


def _numbered_dicts_to_list(data_dict):
    """Transforms all dictionaries with numbered keys to lists."""
    for key, value in data_dict.items():
        if isinstance(value, dict):
            if _is_numbered_dict(value):
                data_dict[key] = [value[k] for k in sorted(value.keys(), key=int)]
            else:
                data_dict[key] = _numbered_dicts_to_list(value)
    return data_dict


def load_hdf5_to_dict(hdf5_file, parent_group="/"):
    """
    Recursively reads an HDF5 file into a nested dictionary.

    Parameters
    ----------
    hdf5_file : h5py.File
        Opened h5py.File object.
    parent_group : str
        Current group path in HDF5 file (default is root "/").

    Returns
    -------
        Nested dictionary representing the HDF5 file structure.
    """
    data_dict = {}
    for key in hdf5_file[parent_group]:
        item_path = f"{parent_group}/{key}"
        if isinstance(hdf5_file[item_path], h5py.Group):
            data_dict[key] = load_hdf5_to_dict(hdf5_file, parent_group=item_path)
        else:
            item = hdf5_file[item_path][()]
            if isinstance(item, bytes):
                item = item.decode("utf-8")
            # Convert scalar numpy arrays to Python scalars
            elif np.isscalar(item):
                item = item.item()

            data_dict[key] = item

    return _numbered_dicts_to_list(data_dict)


def check_hdf5_image_hash(path, hashable):
    """
    Checks the hash of an image in an hdf5 file.

    Parameters
    ----------
    path : str
        The path to the hdf5 file.
    hashable : any
        The data to check the hash against.

    Returns
    -------
    bool
        True if the hash matches, False otherwise.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")

    with h5py.File(path, "r") as dataset:
        stored_hash = dataset["image"].attrs.get("hash", None)

    return stored_hash == hash(hashable)
