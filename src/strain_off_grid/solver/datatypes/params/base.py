from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp

from strain_off_grid.solver.datatypes.params.components import Param


# Marker classes (no logic, just for the type checker)
class Scaled:
    pass


class Physical:
    pass


T = TypeVar("T", Scaled, Physical)


@jax.tree_util.register_dataclass
@dataclass
class ParamsBase(Generic[T]):
    def to_physical(self) -> ParamsBase[Physical]:
        return _self_iterate(self, lambda x: x.to_physical())

    def to_scaled(self) -> "ParamsBase[Scaled]":
        return _self_iterate(self, lambda x: x.to_scaled())

    def index_scatterers(self, indices) -> ParamsBase:
        """Recursively indexes the scatterer dimension of all internal parameters."""
        return _self_iterate(self, lambda x: x.index_scatterers(indices))

    def index_frames(self, indices) -> ParamsBase:
        """Recursively indexes the frame dimension of all internal parameters."""
        return _self_iterate(self, lambda x: x.index_frames(indices))

    def copy(self, **kwargs) -> ParamsBase:
        """Creates a copy of the Params object, optionally updating some fields."""
        return self.__class__(**{**self.__dict__, **kwargs})

    def save_to_hdf5(self, h5_group):
        """Iterates through fields and tells each one to save itself."""
        for field_name in self.__dataclass_fields__:
            param_obj = getattr(self, field_name)
            if isinstance(param_obj, Param):
                param_obj.save_to_hdf5(h5_group)

    @classmethod
    def from_arrays(cls, **kwargs) -> ParamsBase:
        initializer = {}
        for field_name in cls.__dataclass_fields__:
            if field_name not in kwargs:
                raise ValueError(f"Missing required parameter: {field_name}")
            param_cls = Param._registry[field_name]
            initializer[field_name] = param_cls(data=kwargs[field_name])
        return cls(**initializer)

    @classmethod
    def load_from_hdf5(cls, h5_group) -> ParamsBase:
        """Uses the Registry to reconstruct the class from HDF5 keys."""
        kwargs = {}
        for key in h5_group.keys():
            param_cls = Param._registry.get(key)
            if not param_cls:
                continue
            # Find which field in ParamsPhysical corresponds to this Param class
            for f_name, f_def in cls.__dataclass_fields__.items():
                if f_def.type == param_cls:
                    kwargs[f_name] = param_cls(data=jnp.array(h5_group[key][()]))
        return cls(**kwargs)

    def add_frame(self, n_tx, extrapolate_alpha=1.0) -> ParamsBase:
        """Recursively adds a frame to all internal parameters that have a frame dimension."""
        return _self_iterate(
            self,
            lambda x: x.add_frame(n_tx, extrapolate_alpha)
            if isinstance(x, Param)
            else x,
        )


def _self_iterate(self, func):
    """Helper function to apply a function to all Param fields."""

    return jax.tree_util.tree_map(
        lambda x: func(x) if isinstance(x, Param) else x,
        self,
        is_leaf=lambda x: isinstance(x, Param),
    )
