from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax

from strain_off_grid.solver.datatypes.params.base import ParamsBase, Physical, Scaled, T
from strain_off_grid.solver.datatypes.params.components import (
    AttenuationCoefficient,
    DeltaPos,
    DeltaPosParis,
    DirecitivityFalloff,
    DirectivityScaling,
    ElementGains,
    ScatAmpPerFrame,
    ScatPos,
    ScatPosParis,
    ScatSpectrum,
    ScatWaveformParams,
    SoundSpeed,
    TPeak,
    WaveformParams,
    WaveformRFFTOffset,
)


@jax.tree_util.register_dataclass
@dataclass
class ParamsRegular(ParamsBase[T]):
    scat_pos: ScatPos
    scat_amp: ScatAmpPerFrame
    waveform_rfft_offset: WaveformRFFTOffset
    waveform_params: WaveformParams
    delta_pos: DeltaPos
    attenuation_coefficient: AttenuationCoefficient
    sound_speed: SoundSpeed
    directivity_falloff: DirecitivityFalloff

    def to_physical(self) -> ParamsRegular[Physical]:
        return cast(ParamsRegular[Physical], super().to_physical())

    def to_scaled(self) -> ParamsRegular[Scaled]:
        return cast(ParamsRegular[Scaled], super().to_scaled())


@jax.tree_util.register_dataclass
@dataclass
class ParamsParis(ParamsBase[T]):
    scat_pos: ScatPosParis
    scat_amp: ScatAmpPerFrame
    waveform_rfft_offset: WaveformRFFTOffset
    delta_pos: DeltaPosParis
    sound_speed: SoundSpeed
    directivity_scaling: DirectivityScaling
    attenuation_coefficient: AttenuationCoefficient
    scat_spectrum: ScatSpectrum
    element_gains: ElementGains
    scat_waveform_params: ScatWaveformParams
    directivity_falloff: DirecitivityFalloff
    t_peak: TPeak

    def to_physical(self) -> ParamsParis[Physical]:
        return cast(ParamsParis[Physical], super().to_physical())

    def to_scaled(self) -> ParamsParis[Scaled]:
        return cast(ParamsParis[Scaled], super().to_scaled())
