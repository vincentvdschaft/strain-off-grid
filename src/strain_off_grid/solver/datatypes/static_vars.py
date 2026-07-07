from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass
class StaticVars:
    probe_geometry: jnp.ndarray
    waveform_rfft: jnp.ndarray
    initial_times: jnp.ndarray
    freqs: jnp.ndarray
    sound_speed: float
    center_frequency: float
    sampling_frequency: float
    element_width: float
    attenuation_coef: float
    extent: tuple
    tgc_gain: float
    t_peak: float
    l1_regularization: float
    expected_velocity_range: tuple
    planewave_time_offsets: jnp.ndarray
    planewave_angles: jnp.ndarray
    t0_delays: jnp.ndarray
    tx_apodizations: jnp.ndarray
    polar_angles: jnp.ndarray
    focus_distances: jnp.ndarray

    @property
    def n_el(self):
        return self.probe_geometry.shape[0]

    @property
    def n_tx(self):
        return self.initial_times.shape[0]

    @property
    def n_fbins(self):
        return self.freqs.size

    @property
    def wavelength(self):
        return self.sound_speed / self.center_frequency
