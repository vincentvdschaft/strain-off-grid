import dataclasses
import importlib
from dataclasses import dataclass
from typing import Self, get_type_hints

import h5py
import numpy as np


def load_dataclass(path: str, group: str = "/"):
    """Loads a dataclass instance from an HDF5 file."""
    with h5py.File(path, "r") as f:
        grp = f[group]
        try:
            cls = _resolve_class(grp)
        # TODO: Remove this fallback
        except Exception:
            from strain_off_grid.phantoms.ring import ShortAxisPhantom

            cls = ShortAxisPhantom
        return cls._read_from_group(grp)


def _resolve_class(grp: h5py.Group):
    """Imports the dataclass referenced by the group's metadata attributes."""
    module = importlib.import_module(grp.attrs["__module__"])
    cls = module
    for name in grp.attrs["__class__"].split("."):
        cls = getattr(cls, name)
    return cls


@dataclass
class HDF5Mixin:
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
            print(f"Writing field '{fld.name}' of type {fld.type}")
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
        elif isinstance(value, (list, tuple)):
            arr = np.array(value)
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
        hints = get_type_hints(cls)

        for fld in dataclasses.fields(cls):
            name = fld.name
            ftype = hints.get(name, None)

            if name in grp:
                item = grp[name]
                if isinstance(item, h5py.Group):
                    # Nested dataclass — recurse
                    kwargs[name] = ftype._read_from_group(item)
                else:
                    kwargs[name] = item[()]  # load dataset
            elif name in grp.attrs:
                val = grp.attrs[name]
                kwargs[name] = None if val == "__none__" else val
            # else: field has a default, leave it out of kwargs

        return cls(**kwargs)
