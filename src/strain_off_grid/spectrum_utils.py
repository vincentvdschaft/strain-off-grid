"""
Helper functions for spectrum and ultrasound operations.

NOTE:
n_fft is the number of FFT points, which is a power of 2.
n_rfft is the number of positive frequency bins, which is n_fft // 2 + 1.

"""

import jax.numpy as jnp
import numpy as np


def get_pulse_spectrum_fn(fc, n_period, fs):
    """Computes the spectrum of a sine that is windowed with a Hann window.

    Parameters
    ----------
    fc : float
        The center frequency of the pulse.
    n_period : float
        The number of periods to include in the pulse.

    Returns
    -------
    spectrum_fn : callable
        A function that computes the spectrum of the pulse for the input frequencies
        in Hz.
    """
    std = n_period / 2 / fc

    def spectrum_fn(f):
        return 1 / 1j * (gaussian_fd(f - fc, std) - gaussian_fd(f + fc, std)) * fs / 2

    return spectrum_fn


def gaussian_fd(f, std):
    """The fourier transform of a gaussian window in the time domain with given std in the time domain in seconds.

    Parameters
    ----------
    f : array-like
        The input frequencies in Hz.
    std : float
        The standard deviation of the gaussian window in seconds.

    Returns
    -------
    gaussian_fd_vals : array-like
        The values of the Gaussian window function in the frequency domain.
    """
    return std * jnp.sqrt(2 * jnp.pi) * jnp.exp(-2 * (jnp.pi * f * std) ** 2)


# ==============================================================================
# Windows
# ==============================================================================
def hann(x, width):
    """Hann window function.

    Parameters
    ----------
    x : array-like
        The input values.
    width : float
        The width of the window. This is the total width from -x to x. The window will
        be nonzero in the range [-width/2, width/2].

    Returns
    -------
    hann_vals : array-like
        The values of the Hann window function.
    """
    return jnp.where(
        jnp.abs(x) < width / 2, 1 / width * jnp.cos(np.pi * x / width) ** 2, 0
    )


def hann_unnormalized(x, width):
    """Hann window function that is 1 at the peak. This means that the integral of the
    window function is not necessarily 1.

    Parameters
    ----------
    x : array-like
        The input values.
    width : float
        The width of the window. This is the total width from -x to x. The window will
        be nonzero in the range [-width/2, width/2].

    Returns
    -------
    hann_vals : array-like
        The values of the Hann window function.
    """
    return jnp.where(jnp.abs(x) < width / 2, jnp.cos(np.pi * x / width) ** 2, 0)


def hann_fd(f, width):
    """The fourier transform of a hann window in the time domain with given width in the time domain in seconds.

    Parameters
    ----------
    f : array-like
        The input frequencies in Hz.
    width : float
        The width of the window in seconds. This is the total width from -width/2 to width/2.

    Returns
    -------
    hann_fd_vals : array-like
        The values of the Hann window function in the frequency domain.
    """
    denom = 1.0 - (f * width) ** 2
    num = 0.5 * jnp.sinc(f * width)
    result = num / denom
    result = jnp.where(jnp.abs(result) > 1.1, 0.25, result)
    return jnp.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.25)


def get_transducer_bandwidth_fn(fc, bandwidth):
    """Computes the spectrum of a probe with a center frequency and bandwidth.

    Parameters
    ----------
    fc : float
        The center frequency of the probe.
    bandwidth : float
        The bandwidth of the probe.

    Returns
    -------
    spectrum_fn : callable
        A function that computes the spectrum of the pulse for the input frequencies
        in Hz.
    """

    def bandwidth_fn(f):
        return hann_unnormalized(jnp.abs(f) - fc, bandwidth)

    return bandwidth_fn


# ==============================================================================
# Ultrasound operations
# ==============================================================================
def delay(f, tau):
    """Applies a delay in the frequency domain.

    Parameters
    ----------
    f : array-like
        The input frequencies.
    tau : float
        The delay to apply.

    Returns
    -------
    spect : array-like
        The spectrum of the delay.
    """
    return jnp.exp(-1j * 2 * np.pi * tau * f)


def delay2(f, tau, n_fft, fs):
    """Applies a delay in the frequency domain without phase wrapping.

    Parameters
    ----------
    f : array-like
        The input frequencies.
    tau : float
        The delay to apply.
    n_fft : int
        The number of samples in the FFT.
    fs : float
        The sampling frequency.

    Returns
    -------
    spect : array-like
        The spectrum of the delay.
    """
    return jnp.where(tau < n_fft / fs, jnp.exp(-1j * 2 * np.pi * tau * f), 0)


def attenuate(f, attenuation_coef, dist):
    """Applies attenuation to the signal in the frequency domain.

    Parameters
    ----------
    f : array-like
        The input frequencies.
    attenuation_coef : float
        The attenuation coefficient in dB/cm/MHz.
    dist : float
        The distance the signal has traveled.

    Returns
    -------
    spect : array-like
        The spectrum of the attenuation.
    """
    return jnp.exp(-np.log(10) * attenuation_coef / 20 * dist * 100 * jnp.abs(f) * 1e-6)


def spread(dist, mindist=1e-4):
    dist = jnp.clip(dist, mindist, None)
    return mindist / dist


def directivity(f, theta, element_width, sound_speed, rigid_baffle=False):
    wavelength = sound_speed / f

    response = jnp.sinc(element_width / wavelength * jnp.sin(theta))
    if not rigid_baffle:
        response *= jnp.cos(theta)
    return response


def to_time_domain(spect, n_fft=-1, n_ax=-1, axis=-1):
    if n_fft == -1:
        n_fft = spect.shape[axis]

    td_signal = jnp.real(jnp.fft.irfft(spect, axis=axis, n=n_fft))

    td_signal = pad_or_cut(td_signal, n_ax, axis)

    if spect.shape[axis] <= n_fft / 2:
        td_signal *= 2
    return td_signal


def pad_or_cut(arr, size, axis):
    if arr.shape[axis] < size:
        padding = [(0, 0) for _ in range(arr.ndim)]
        padding[axis] = (0, size - arr.shape[axis])
        return jnp.pad(arr, padding, mode="constant")
    else:
        return jnp.take(arr, jnp.arange(size), axis=axis)


def waveform_samples_to_fbins(
    waveform_samples,
    target_fs,
    n_fft=1024,
    fs_waveform=250e6,
):
    """Takes a sampled waveform and computes the positive frequency bins of the waveform"""

    # Compute the FFT of the waveform
    waveform_samples_fft = jnp.fft.rfft(waveform_samples, n=n_fft)

    # Interpolate the FFT to the target frequency resolution
    fbins = interp_rfft(waveform_samples_fft, fs_waveform, target_fs, n_fft)

    return fbins


def waveform_fbins_to_samples(
    positive_fbins,
):
    """Converts the positive frequency bins of a waveform to the time domain."""

    waveform_samples = jnp.fft.irfft(positive_fbins)

    return waveform_samples


def waveform_resample_to_td(
    waveform_rfft, fs, target_fs, target_n_fft, trim=False, normalize=False
):
    """Convenience function to resample a waveform to a new sampling frequency, bring
    it to the time domain, and then trim it.

    Parameters
    ----------
    waveform_rfft : jnp.ndarray
        The positive frequency bins of the waveform of shape [n_fft // 2 + 1].
    fs : float
        The sampling frequency of the input waveform.
    target_fs : float
        The target sampling frequency.
    target_n_fft : int
        The number of FFT points to interpolate to before converting to the time domain.
    trim : bool
        Whether to trim the waveform to remove zeros at the end.
    normalize : bool
        Whether to normalize the waveform to have unit maximum amplitude.

    Returns
    -------
    waveform_td : jnp.ndarray
        The resampled waveform.
    t : jnp.ndarray
        The time vector.
    """

    waveform_rfft_interp = interp_rfft(
        waveform_rfft, fs, target_fs, target_n_fft=target_n_fft
    )
    waveform_td = waveform_fbins_to_samples(waveform_rfft_interp)
    if trim:
        waveform_td = trim_waveform_td(waveform_td)
    if normalize:
        waveform_td = waveform_td / jnp.max(jnp.abs(waveform_td))
    t = jnp.arange(waveform_td.size) / target_fs
    return waveform_td, t


def interp_rfft(input_rfft_bins, input_fs, target_fs, target_n_fft):
    """Interpolates the positive frequency bins of a waveform to a new frequency resolution.

    Parameters
    ----------
    input_rfft_bins : jnp.ndarray
        The positive frequency bins of the waveform of shape [n_fft // 2 + 1].
    input_fs : float
        The sampling frequency of the input waveform.
    target_fs : float
        The target sampling frequency.
    target_n_fft : int
        The number of FFT points to interpolate to.

    Returns
    -------
    waveform_rfft_interp : jnp.ndarray
        The interpolated waveform.
    """
    n_fft = 2 * (input_rfft_bins.size - 1)

    assert np.log2(target_n_fft) % 1 == 0, "target_n_fft must be a power of 2"

    input_freqs = np.fft.rfftfreq(n_fft, 1 / input_fs)
    target_freqs = np.fft.rfftfreq(target_n_fft, 1 / target_fs)

    if target_fs > input_fs:
        delta_freq = target_freqs[1] - target_freqs[0]
        waveform_spectrum_offset_freqs = np.fft.rfftfreq(target_n_fft, 1 / target_fs)
        waveform_spectrum_offset_freqs = waveform_spectrum_offset_freqs[
            waveform_spectrum_offset_freqs > input_freqs[-1]
        ]
        waveform_spectrum_offset_fbins = np.zeros(waveform_spectrum_offset_freqs.size)
        input_freqs = np.concatenate([input_freqs, waveform_spectrum_offset_freqs])
        input_rfft_bins = np.concatenate(
            [input_rfft_bins, waveform_spectrum_offset_fbins]
        )

    input_rfft_bins_mag = jnp.abs(input_rfft_bins)
    input_rfft_bins_phase = jnp.angle(input_rfft_bins)

    waveform_rfft_interp_mag = jnp.interp(
        target_freqs, input_freqs, input_rfft_bins_mag
    )
    waveform_rfft_interp_phase = jnp.interp(
        target_freqs, input_freqs, input_rfft_bins_phase
    )

    waveform_rfft_interp = waveform_rfft_interp_mag * jnp.exp(
        1j * waveform_rfft_interp_phase
    )

    return waveform_rfft_interp * target_fs / input_fs


def trim_waveform_td(pulse_td, threshold_value=1e-3, threshold_n_samples=5, t=None):
    """Trims the waveform to the first zero crossing."""
    normval = jnp.max(jnp.abs(pulse_td))
    pulse_td_normalized = pulse_td / normval

    # Detect the first point where the signal is below the threshold for threshold_n_samples
    threshold_filter = jnp.ones(threshold_n_samples)
    threshold_filter_sum = jnp.convolve(
        jnp.abs(pulse_td_normalized) < threshold_value, threshold_filter, mode="same"
    )

    # Return the pulse as is if the threshold is never reached
    if np.all(threshold_filter_sum == 0):
        return pulse_td

    # Find the first index where the threshold is reached for threshold_n_samples
    first_nonzero = jnp.argmax(threshold_filter_sum >= threshold_n_samples)

    # Clip the first nonzero to be at least 2
    first_nonzero = jnp.clip(first_nonzero, 2, None)

    if t is not None:
        return t[:first_nonzero], pulse_td[:first_nonzero]

    return pulse_td[:first_nonzero]


def fft_energy(signal, axis=-1, n_fft_bins=None):
    """Computes the energy of the signal, normalized by the number of FFT bins."""
    if n_fft_bins is None:
        n_fft_bins = signal.shape[axis]
    return jnp.sqrt(jnp.mean(jnp.abs(signal) ** 2, axis=axis))  # / n_fft_bins


def normalize_to_unit_energy(signal, n_fft_bins=None, axis=-1):
    """Normalizes the signal to have unit energy."""
    normval = 1 / fft_energy(signal, axis=axis, n_fft_bins=n_fft_bins)
    return signal * normval


def compute_waveform_rfft(fc, n_period, fs, n_fft, band, tgc_alpha):
    """Computes an anlytical waveform in the frequency domain.

    Parameters
    ----------
    fc : float
        Center frequency [Hz].
    n_period : float
        Number of periods.
    fs : float
        Sampling frequency [Hz].
    n_fft : int
        Number of FFT points.
    band : float
        Bandwidth [Hz].
    tgc_alpha : float
        TGC alpha. The tgc curve is computed as exp(tgc_alpha * t / fs).

    Returns
    -------
    waveform_rfft : jnp.ndarray
        The positive frequency bins of the waveform of shape [n_fft // 2 + 1].
    """
    # Compute the waveform in the frequency domain
    freqs = np.fft.rfftfreq(n_fft, 1 / fs)
    waveform_rfft = get_pulse_spectrum_fn(fc, n_period=n_period, fs=fs)(freqs)

    # Shift to the start of the signal
    pulse_width = n_period / fc
    waveform_rfft *= delay2(freqs, pulse_width / 2, n_fft=n_fft, fs=fs)

    # Apply the transducer bandwidth
    # waveform_rfft = waveform_rfft * get_transducer_bandwidth_fn(fc, band)(freqs)

    # Apply the TGC curve
    waveform_td = jnp.fft.irfft(waveform_rfft)
    tgc_gain_curve = compute_tgc_curve(tgc_alpha, waveform_td.size, fs)
    waveform_td *= tgc_gain_curve
    waveform_rfft = jnp.fft.rfft(waveform_td)

    waveform_rfft = normalize_to_unit_energy(waveform_rfft)
    return waveform_rfft


def next_power_of_2(x):
    """Computes the next power of 2 of a number (e.g. 5.6 -> 8)."""
    return np.power(2, np.ceil(np.log2(x))).astype(int)


def compute_tgc_curve(tgc_alpha, n_ax, fs):
    """Compute the TGC curve given by exp(tgc_alpha * t / fs)."""
    tgc_gain_curve = jnp.exp(tgc_alpha * jnp.arange(n_ax) / fs)
    return tgc_gain_curve
