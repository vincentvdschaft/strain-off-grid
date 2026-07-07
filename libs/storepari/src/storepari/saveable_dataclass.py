from __future__ import annotations

import dataclasses
import importlib
from dataclasses import dataclass
from typing import Self

import h5py
import numpy as np


def load_dataclass(path: str, group: str = "/"):
    """Loads a dataclass instance from an HDF5 file."""
    with h5py.File(path, "r") as f:
        grp = f[group]
        cls = _resolve_class(grp)
        return cls._read_from_group(grp)


def _resolve_class(grp: h5py.Group):
    """Imports the dataclass referenced by the group's metadata attributes."""
    module = importlib.import_module(grp.attrs["__module__"])
    cls = module
    for name in grp.attrs["__class__"].split("."):
        cls = getattr(cls, name)
    return cls


@dataclass
class SaveableDataclass:
    """Base class providing generic HDF5 serialization for dataclasses."""

    def to_hdf5(self, path: str, group: str = "/") -> Self:
        with h5py.File(path, "a") as f:
            grp = f.require_group(group)
            self._write_to_group(grp)
        return self

    def _write_to_group(self, grp: h5py.Group) -> None:
        print(type(self), [f.name for f in dataclasses.fields(self)])
        grp.attrs["__class__"] = type(self).__qualname__
        grp.attrs["__module__"] = type(self).__module__
        for fld in dataclasses.fields(self):
            value = getattr(self, fld.name)
            self._write_value(grp, fld.name, value)

    @staticmethod
    def _write_value(grp: h5py.Group, key: str, value) -> None:
        if key in grp:
            del grp[key]  # overwrite

        if isinstance(value, np.ndarray):
            grp.create_dataset(key, data=value)
        elif isinstance(value, (int, float, str, bool)):
            grp.attrs[key] = value
        elif isinstance(value, (list, tuple)) and any(
            dataclasses.is_dataclass(item) for item in value
        ):
            sub = grp.require_group(key)
            sub.attrs["__list__"] = True
            sub.attrs["__len__"] = len(value)
            for i, item in enumerate(value):
                item_grp = sub.require_group(str(i))
                item._write_to_group(item_grp)
        elif isinstance(value, (list, tuple)):
            arr = np.array(value)
            if arr.dtype.kind in ("U", "S"):
                grp.create_dataset(
                    key, data=np.array(value, dtype=object), dtype=h5py.string_dtype()
                )
            else:
                grp.create_dataset(key, data=arr)
        elif dataclasses.is_dataclass(value):
            sub = grp.require_group(key)
            value._write_to_group(sub)
        elif value is None:
            grp.attrs[key] = "__none__"
        else:
            raise TypeError(f"Unsupported field type for '{key}': {type(value)}")

    @classmethod
    def from_hdf5(cls, path: str, group: str = "/"):
        with h5py.File(path, "r") as f:
            grp = f[group]
            return cls._read_from_group(grp)

    @classmethod
    def _read_from_group(cls, grp: h5py.Group):
        kwargs = {}

        for fld in dataclasses.fields(cls):
            name = fld.name

            if name in grp:
                item = grp[name]
                if isinstance(item, h5py.Group):
                    if item.attrs.get("__list__", False):
                        # List of dataclasses — recurse into each indexed subgroup
                        length = item.attrs["__len__"]
                        values = []
                        for i in range(length):
                            item_grp = item[str(i)]
                            item_cls = _resolve_class(item_grp)
                            values.append(item_cls._read_from_group(item_grp))
                        kwargs[name] = values
                    else:
                        # Nested dataclass — recurse
                        item_cls = _resolve_class(item)
                        kwargs[name] = item_cls._read_from_group(item)
                else:
                    kwargs[name] = item[()]  # load dataset
            elif name in grp.attrs:
                val = grp.attrs[name]
                if isinstance(val, np.generic):
                    val = (
                        val.item()
                    )  # h5py returns numpy scalars, not native Python types
                kwargs[name] = None if val == "__none__" else val
            # else: field has a default, leave it out of kwargs

        return cls(**kwargs)
