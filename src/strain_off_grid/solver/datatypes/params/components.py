from __future__ import annotations

from abc import ABC
from dataclasses import Field, dataclass, field
from functools import partial

import jax
import jax.numpy as jnp

WAVELENGTH_2D = 1540.0 / 3.9e6


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "data",
    ],
    meta_fields=[],
)
@dataclass
class Param(ABC):
    data: jnp.ndarray
    name: str = field(init=False, default="base_param")
    scat_dim: int = field(init=False, default=-1)
    frame_dim: int = field(init=False, default=-1)
    ndim: int = field(init=False, default=-1)
    scaling: float = field(init=False, default=1.0)
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name_obj = cls.__dict__.get("name")
        if isinstance(name_obj, Field):
            name_default = name_obj.default
        elif isinstance(name_obj, str):
            name_default = name_obj
        else:
            name_default = cls.__dataclass_fields__["name"].default
        Param._registry[name_default] = cls

        # Register the class in JAX's pytree system
        jax.tree_util.register_dataclass(cls, data_fields=["data"], meta_fields=[])

    def to_scaled(self):
        return self.__class__(data=self.data / self.scaling)

    def to_physical(self):
        return self.__class__(data=self.data * self.scaling)

    def __getitem__(self, key):
        return self.__class__(data=self.data[key])

    def index_scatterers(self, indices):
        """Indexes the scatterer dimension with the given indices."""
        if self.scat_dim == -1:
            return self
        return self.__class__(data=_slice_in_dim(self.data, self.scat_dim, indices))

    def index_frames(self, indices):
        """Indexes the frame dimension with the given indices."""
        if self.frame_dim == -1:
            return self
        return self.__class__(data=_slice_in_dim(self.data, self.frame_dim, indices))

    def add_frame(self, n_tx, extrapolate_alpha=1.0):
        if self.frame_dim == -1:
            return self.__class__(data=self.data)
        return self.__class__(data=_extend_frames(self.data, self.frame_dim, n_tx))

    def save_to_hdf5(self, hdf5_group):
        hdf5_group.create_dataset(self.name, data=self.data)

    def load_from_hdf5(self, hdf5_group):
        return self.__class__(data=hdf5_group[self.name][()])

    @property
    def shape(self):
        return self.data.shape


def _slice_in_dim(data, dim, slice_obj):
    slices = [slice(None)] * data.ndim
    slices[dim] = slice_obj
    return data[tuple(slices)]


def _frames_to_add(n_active, n_tx):
    """Sides to grow so the window stays centered like the model's (n_tx - n_active) // 2 offset."""
    remaining = n_tx - n_active
    if remaining >= 2:
        return True, True
    if remaining <= 0:
        return False, False
    extends_at_front = remaining % 2 == 0
    return extends_at_front, not extends_at_front


def _extend_frames(data, frame_dim, n_tx, frame_offset=0.0):
    n_active = data.shape[frame_dim]
    add_first, add_last = _frames_to_add(n_active, n_tx)
    parts = [data]
    if add_first:
        parts.insert(
            0, _slice_in_dim(data, frame_dim, slice(0, 1, None)) + frame_offset
        )
    if add_last:
        parts.append(
            _slice_in_dim(data, frame_dim, slice(-1, None, None)) + frame_offset
        )
    return jnp.concatenate(parts, axis=frame_dim)


@dataclass
class ScatPos(Param):
    data: jnp.ndarray
    name: str = field(init=False, default="scat_pos")
    scat_dim: int = field(init=False, default=0)
    frame_dim: int = field(init=False, default=1)
    ndim: int = field(init=False, default=3)
    scaling: float = field(init=False, default=WAVELENGTH_2D)


@dataclass
class ScatPosParis(Param):
    data: jnp.ndarray
    name: str = field(init=False, default="scat_pos")
    scat_dim: int = field(init=False, default=0)
    frame_dim: int = field(init=False, default=1)
    ndim: int = field(init=False, default=3)
    scaling: float = field(init=False, default=1480 / 1e6)


@dataclass
class ScatAmpPerFrame(Param):
    data: jnp.ndarray
    name: str = field(init=False, default="scat_amp")
    scat_dim: int = field(init=False, default=0)
    frame_dim: int = field(init=False, default=1)
    ndim: int = field(init=False, default=2)
    scaling: float = field(init=False, default=1.0)

    def to_scaled(self):
        return ScatAmpPerFrame(data=jnp.log(jnp.abs(self.data) + 1e-8) / self.scaling)

    def to_physical(self):
        return ScatAmpPerFrame(data=jnp.exp(self.data * self.scaling))

    def add_frame(self, n_tx, extrapolate_alpha=1.0):
        if self.frame_dim == -1:
            return self.__class__(data=self.data)
        return self.__class__(
            data=_extend_frames(self.data, self.frame_dim, n_tx, frame_offset=1.0)
        )


@dataclass
class ScatAmp(Param):
    data: jnp.ndarray
    name: str = field(init=False, default="scat_amp")
    scat_dim: int = field(init=False, default=0)
    ndim: int = field(init=False, default=1)
    scaling: float = field(init=False, default=20.0)

    # def to_scaled(self):
    #     return ScatAmp(data=jnp.log(jnp.abs(self.data) + 1e-8) / self.scaling)

    # def to_physical(self):
    #     return ScatAmp(data=jnp.exp(self.data * self.scaling))


@dataclass
class WaveformRFFTOffsetAngles(Param):
    data: jnp.ndarray
    name: str = field(init=False, default="waveform_rfft_offset_angles")
    scat_dim: int = field(init=False, default=-1)
    frame_dim: int = field(init=False, default=-1)
    ndim: int = field(init=False, default=3)


@dataclass
class WaveformRFFTOffset(Param):
    data: jnp.ndarray
    name: str = field(init=False, default="waveform_rfft_offset")
    ndim: int = field(init=False, default=2)
    scaling: float = field(init=False, default=20.0)

    def to_scaled(self):
        return WaveformRFFTOffset(
            data=jnp.log(jnp.abs(self.data) + 1e-8) / self.scaling
        )

    def to_physical(self):
        return WaveformRFFTOffset(data=jnp.exp(self.data * self.scaling))

    def compute_waveform(self, waveform_rfft):
        return waveform_rfft * (self.data[:, 0] * jnp.exp(1j * self.data[:, 1]))


@dataclass
class ElementGains(Param):
    data: jnp.ndarray
    name: str = field(init=False, default="element_gains")
    scat_dim: int = field(init=False, default=-1)
    frame_dim: int = field(init=False, default=-1)
    ndim: int = field(init=False, default=1)


@dataclass
class DeltaPos(Param):
    data: jnp.ndarray
    name: str = field(init=False, default="delta_pos")
    scat_dim: int = field(init=False, default=0)
    frame_dim: int = field(init=False, default=1)
    ndim: int = field(init=False, default=2)
    scaling: float = field(init=False, default=WAVELENGTH_2D)


@dataclass
class DeltaPosParis(Param):
    data: jnp.ndarray
    name: str = field(init=False, default="delta_pos")
    scat_dim: int = field(init=False, default=0)
    frame_dim: int = field(init=False, default=1)
    ndim: int = field(init=False, default=2)
    scaling: float = field(init=False, default=20 * 1480 / 1e6)


@dataclass
class Width(Param):
    data: jnp.ndarray
    name: str = field(init=False, default="width")
    scat_dim: int = field(init=False, default=-1)
    frame_dim: int = field(init=False, default=-1)


@dataclass
class SirenParams(Param):
    data: dict
    name: str = field(init=False, default="siren_params")
    scat_dim: int = field(init=False, default=-1)
    frame_dim: int = field(init=False, default=-1)

    def save_to_hdf5(self, hdf5_group):
        pass

    def load_from_hdf5(self, hdf5_group):
        return {}


@dataclass
class ElementDelays(Param):
    data: jnp.ndarray
    name: str = field(init=False, default="element_delays")
    scat_dim: int = field(init=False, default=-1)
    frame_dim: int = field(init=False, default=-1)

    def to_scaled(self):
        return ElementDelays(data=self.data / 1e-6)

    def to_physical(self):
        return ElementDelays(data=self.data * 1e-6)


@dataclass
class TXPhases(Param):
    data: jnp.ndarray
    name: str = field(init=False, default="tx_phases")
    scat_dim: int = field(init=False, default=0)
    frame_dim: int = field(init=False, default=-1)


@dataclass
class TPeak(Param):
    data: jnp.ndarray
    name: str = field(init=False, default="t_peak_fit")

    def to_scaled(self):
        return TPeak(data=self.data / 1e-6)

    def to_physical(self):
        return TPeak(data=self.data * 1e-6)


@dataclass
class SoundSpeed(Param):
    data: jnp.ndarray
    name: str = field(init=False, default="sound_speed")

    def to_scaled(self):
        return SoundSpeed(data=self.data / 50.0)

    def to_physical(self):
        return SoundSpeed(data=self.data * 50.0)


@dataclass
class DirectivityScaling(Param):
    data: jnp.ndarray
    name: str = field(init=False, default="directivity_scaling")


@dataclass
class AttenuationCoefficient(Param):
    data: jnp.ndarray
    name: str = field(init=False, default="attenuation_coefficient")

    def to_scaled(self):
        return AttenuationCoefficient(data=self.data * 1e1)

    def to_physical(self):
        return AttenuationCoefficient(data=jnp.abs(self.data / 1e1))


@dataclass
class ScatSpectrum(Param):
    data: jnp.ndarray
    name: str = field(init=False, default="scat_spectrum")
    scat_dim: int = field(init=False, default=0)
    ndim: int = field(init=False, default=2)


@dataclass
class ScatWaveformParams(Param):
    data: jnp.ndarray
    name: str = field(init=False, default="scat_waveform_params")
    scat_dim: int = field(init=False, default=0)
    ndim: int = field(init=False, default=2)


@dataclass
class Phases(Param):
    data: jnp.ndarray
    name: str = field(init=False, default="phases")
    scat_dim: int = field(init=False, default=0)
    frame_dim: int = field(init=False, default=1)
    transmit_dim: int = field(init=False, default=2)
    ndim: int = field(init=False, default=3)


@dataclass
class WaveformParams(Param):
    data: jnp.ndarray
    name: str = field(init=False, default="waveform_params")
    scat_dim: int = field(init=False, default=0)
    transmit_dim: int = field(init=False, default=1)
    ndim: int = field(init=False, default=3)


@dataclass
class DirecitivityFalloff(Param):
    data: jnp.ndarray
    name: str = field(init=False, default="directivity_falloff")
    ndim: int = field(init=False, default=0)
