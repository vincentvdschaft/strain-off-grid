"""Custom RFFT Functions that assume the RFFT spectrum is all zero except for the
n_fbins frequency bins in the middle.
"""

import jax.numpy as jnp


def get_custom_rfft_fns(
    sampling_frequency, demodulation_frequency, bandwidth, source_signal_size
):
    """Returns custom RFFT functions that operate on a reduced frequency range of size n_fbins.

    Returns
    -------
    custom_rfft : callable
        A function that computes the RFFT of an input array.
    custom_irfft : callable
        A function that computes the inverse RFFT of an input array.
    custom_rfftfreq : callable
        A function that returns the frequency bins for the RFFT.
    """

    def all_source_freqs():
        freqs_all = (
            jnp.fft.fftfreq(source_signal_size, 1 / sampling_frequency)
            + demodulation_frequency
        )
        return freqs_all

    def freqs_mask():
        freqs_all = all_source_freqs()
        mask = jnp.abs(freqs_all - demodulation_frequency) < bandwidth / 2
        return mask

    def custom_irfft(x_rfft, axis=-1):
        spectrum = _full_spectrum_along_axis(x_rfft, axis)
        return jnp.fft.ifft(spectrum, axis=axis)

    def _full_spectrum_along_axis(x_rfft, axis):
        spectrum_shape = list(x_rfft.shape)
        spectrum_shape[axis] = source_signal_size
        spectrum = jnp.zeros(spectrum_shape, dtype=x_rfft.dtype)
        slices = [slice(None)] * x_rfft.ndim
        slices[axis] = freqs_mask()
        return spectrum.at[tuple(slices)].set(x_rfft)

    def custom_rfftfreq():
        return all_source_freqs()[freqs_mask()]

    def custom_rfft(x, axis=-1):
        return _index_in_dim(jnp.fft.fft(x, axis=axis), axis, freqs_mask())

    return custom_rfft, custom_irfft, custom_rfftfreq


def _index_in_dim(arr, dim, indices):
    """Index an array along a specific dimension.

    Parameters
    ----------
    arr : jnp.ndarray
        The input array to index.
    dim : int
        The dimension along which to index.
    indices : jnp.ndarray
        The indices to select along the specified dimension.

    Returns
    -------
    jnp.ndarray
        The indexed array.
    """
    # Create a tuple of slice(None) for all dimensions
    slices = [slice(None)] * arr.ndim
    # Replace the slice for the specified dimension with the indices
    slices[dim] = indices
    return arr[tuple(slices)]


class RFFT:
    def __init__(
        self,
        sampling_frequency,
        demodulation_frequency,
        bandwidth,
        source_signal_size,
    ):
        self.sampling_frequency = sampling_frequency
        self.demodulation_frequency = demodulation_frequency
        self.bandwidth = bandwidth
        self.source_signal_size = source_signal_size
        self.rfft, self.irfft, self.rfftfreq = get_custom_rfft_fns(
            sampling_frequency,
            demodulation_frequency,
            bandwidth,
            source_signal_size,
        )
        self.n_fbins = self.rfftfreq().shape[0]

    def rfft(self, x: jnp.ndarray, axis=-1) -> jnp.ndarray:
        return self.rfft(x, axis=axis)

    def irfft(self, x_rfft: jnp.ndarray, axis=-1) -> jnp.ndarray:
        return self.irfft(x_rfft, axis=axis)

    def rfftfreq(self) -> jnp.ndarray:
        return self.rfftfreq()
