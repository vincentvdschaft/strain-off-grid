from __future__ import annotations

import copy
import os
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import zea
from imagelib import Image
from ulmtools import detect_peaks
from zea import Pipeline
from zea.ops import (
    ApplyWindow,
    BandPassFilter,
    Beamform,
    Demodulate,
    Downsample,
)

from strain_off_grid import console
from strain_off_grid.solver import (
    RFFT,
    ProgramState,
    StaticVars,
)
from strain_off_grid.solver.datatypes.config import SolverConfig
from strain_off_grid.solver.datatypes.dimensions import Dimensions
from strain_off_grid.solver.datatypes.indices import Indices
from strain_off_grid.solver.datatypes.params import (
    AttenuationCoefficient,
    DeltaPos,
    DirecitivityFalloff,
    ParamsRegular,
    Physical,
    ScatAmpPerFrame,
    ScatPos,
    SoundSpeed,
    WaveformParams,
    WaveformRFFTOffset,
)
from strain_off_grid.solver.model import forward_model
from strain_off_grid.spectrum_utils import (
    get_pulse_spectrum_fn,
    hann_unnormalized,
)
from strain_off_grid.utils import (
    suppress_logs,
)


def pl(x, y, ax=None):
    sort_idx = np.argsort(x)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x[sort_idx], y[sort_idx])
    plt.savefig("out/debug.png")


def _smooth_with_edge_padding(values, filter_size):
    padded = np.pad(values, filter_size, mode="edge")
    kernel = np.ones(filter_size) / filter_size
    return np.convolve(padded, kernel, mode="same")[filter_size:-filter_size]


def _equalize_depth(images: Image):
    mean_intensity = np.clip(
        np.mean(np.abs(images.array), axis=(0, 2)), a_min=1e-4, a_max=None
    )

    mean_intensity = _smooth_with_edge_padding(mean_intensity, filter_size=32)

    strength_decrease = np.max(mean_intensity) / mean_intensity[-1]

    scaling = np.exp(np.linspace(0, np.log(strength_decrease), images.shape[1]))
    return images.with_array(scaling[None, :, None] * images.array)


def _get_rf_data(config):
    frames = slice(config.first_frame, config.first_frame + config.n_frames)
    transmits = config.resolved_transmits

    console.log(f"Loading RF data from [yellow]{config.input_file}[/yellow]...")

    with zea.File(config.input_file, mode="r") as file:
        rf_data = file.data.raw_data[frames, transmits, ...]
        parameters = file.load_parameters()
        parameters.selected_transmits = transmits
        timestamps = file.timestamps
        print(timestamps)
        if timestamps is not None:
            timestamps = timestamps.reshape(file.data.raw_data.shape[:2])
            timestamps = timestamps[frames, transmits]

    console.log(f"rf data shape: {rf_data.shape}")

    rf_data, parameters = preprocess_rf_data(
        rf_data, harmonic=config.harmonic, parameters=parameters
    )
    shape = rf_data.shape
    rf_data = rf_data.reshape((shape[0] * shape[1], *shape[2:]))
    parameters = _repeat_parameters_per_frame(parameters, n_frames=config.n_frames)
    return rf_data, parameters, timestamps


def _repeat_parameters_per_frame(parameters, n_frames):
    n_tx = parameters.n_tx
    for key in (
        "initial_times",
        "t0_delays",
        "tx_apodizations",
        "focus_distances",
        "transmit_origins",
        "polar_angles",
        "azimuth_angles",
        "waveforms_one_way",
        "waveforms_two_way",
    ):
        parameters[key] = np.repeat(parameters[key], n_frames, axis=0)
    parameters.n_tx = n_tx * n_frames
    parameters.selected_transmits = np.arange(n_tx * n_frames)

    return parameters


def save_h5(arr):
    with h5py.File("out/h5.hdf5", "w") as f:
        f.create_dataset("arr", data=arr)
        console.log("Saved arr to out/h5.hdf5.")


def initialize(
    config: SolverConfig,
    overwrite_path: Path = None,
    rf_scan_probe: tuple | None = None,
    total_scaling_factor: float = 1.0,
) -> ProgramState:
    assert isinstance(config, SolverConfig)

    key = jax.random.PRNGKey(config.seed)

    # ==========================================================================
    # Load the data
    # ==========================================================================
    rf_data, parameters, timestamps = _get_rf_data(config)

    # ==========================================================================
    # Beamform the RF data
    # ==========================================================================
    console.log("Beamforming the RF data...")
    images_das = beamform_all_frames(
        rf_data=rf_data,
        parameters=copy.deepcopy(parameters),
        target_region=config.target_region,
        apply_lens_correction=False,
        harmonic=config.harmonic,
        f_number=config.f_number,
        enable_pfield=False,
    )
    images_das = images_das.save("out.hdf5")
    sampling_frequency = float(parameters.sampling_frequency)

    rf_data = jnp.array(rf_data, dtype=jnp.float32)

    rfft = RFFT(
        sampling_frequency=sampling_frequency,
        demodulation_frequency=parameters.demodulation_frequency,
        bandwidth=parameters.bandwidth,
        source_signal_size=rf_data.shape[-3],
    )
    # rf_data = _prepare_rf(
    #     rf_data=rf_data,
    #     tgc_gain_curve=parameters.tgc_gain_curve,
    #     sampling_frequency=sampling_frequency,
    #     rfft=rfft,
    #     tgc_per_256_samples=config.tgc_per_256_samples,
    #     demodulation_frequency=parameters.demodulation_frequency,
    # )
    save_h5(rfft.rfft(rf_data, axis=-3))
    # save_h5(rf_data)
    # exit()

    y_rfft = rfft.rfft(zea.func.channels_to_complex(rf_data), axis=-2)
    y_rfft, new_scaling_factor = _normalize_y_rfft(y_rfft)
    y_rfft_flat = y_rfft.ravel()

    n_frames = config.n_frames
    n_tx = parameters.n_tx
    n_el = parameters.n_el
    n_fbins = y_rfft.shape[-2]

    # ==========================================================================
    # Initialize waveform
    # ==========================================================================
    waveform_rfft, parameters, t_peak = initialize_waveform(
        config, parameters, images_das, rfft
    )
    with h5py.File("out/y_rfft.hdf5", "w") as f:
        f.create_dataset("y_rfft", data=y_rfft)
        console.log("Saved y_rfft to out/y_rfft.hdf5.")
        f.create_dataset("waveform_rfft", data=waveform_rfft)
        f.create_dataset("freqs", data=rfft.rfftfreq())
    # ==========================================================================
    # Detect peaks
    # ==========================================================================
    console.log("Detecting peaks...", end="")
    wavelength = parameters.sound_speed / parameters.demodulation_frequency
    # peaks = find_peaks(image, threshold=config.peak_detection_threshold, max_peaks=1024)
    if not config.initialize_randomly:
        peaks, intensities = detect_peaks_from_multiple_frames(
            images_das.normalize(),
            config.max_n_peaks,
            config.peak_detection_threshold,
            config.min_distance_between_peaks_wl * wavelength,
        )
        peaks, intensities = (
            peaks[: config.max_n_peaks],
            intensities[: config.max_n_peaks],
        )
    else:
        image = images_das[n_tx // 2]
        peaks = (
            jax.random.uniform(key, shape=(config.max_n_peaks, 2), dtype=jnp.float32)
            * jnp.array(
                [
                    image.extent[1] - image.extent[0],
                    image.extent[3] - image.extent[2],
                ]
            )[None]
            + jnp.array([image.extent[0], image.extent[2]])[None]
        )

    intensities = jax.random.normal(key, shape=(config.max_n_peaks,), dtype=jnp.float32)

    if False:
        console.log("[red]WARNING:[/red] Shifting peaks for testing purposes.")
        peaks = peaks + jnp.array([0.0, wavelength])[None]

    n_scat = peaks.shape[0]
    console.log(f" -> Detected {n_scat} peaks.")

    # ==========================================================================
    # Initialize the opt vars
    # ==========================================================================
    console.log("Initializing variables...")
    planewave_angles = jnp.array(parameters.polar_angles).astype(jnp.float32)
    probe_geometry = parameters.probe_geometry
    if config.is_2d:
        probe_geometry = probe_geometry[:, [0, 2]]
    sound_speed = float(parameters.sound_speed)
    sampling_frequency = float(parameters.sampling_frequency)
    demodulation_frequency = float(parameters.demodulation_frequency)
    element_width = float(parameters.element_width)

    attenuation_coef_np_m_hz = config.attenuation_coef * 1e-2 * 1e-6

    static_vars = StaticVars(
        probe_geometry=probe_geometry,
        waveform_rfft=waveform_rfft,
        planewave_angles=planewave_angles,
        planewave_time_offsets=_get_planewave_time_offsets_2d(
            planewave_angles, probe_geometry, sound_speed
        ),
        freqs=rfft.rfftfreq(),
        sound_speed=sound_speed,
        center_frequency=demodulation_frequency,
        sampling_frequency=sampling_frequency,
        element_width=element_width,
        attenuation_coef=attenuation_coef_np_m_hz,
        extent=config.target_region,
        tgc_gain=_compute_tgc_gain(
            tgc_per_256_samples=config.tgc_per_256_samples,
            sampling_frequency=sampling_frequency,
        ),
        initial_times=jnp.array(parameters.initial_times).astype(jnp.float32),
        t_peak=jnp.array(t_peak).astype(jnp.float32),
        l1_regularization=config.l1_regularization,
        expected_velocity_range=config.expected_velocity_range_wl_per_frame,
        t0_delays=jnp.array(parameters.t0_delays).astype(jnp.float32),
        tx_apodizations=jnp.array(parameters.tx_apodizations).astype(jnp.float32),
        polar_angles=jnp.array(parameters.polar_angles).astype(jnp.float32),
        focus_distances=jnp.array(parameters.focus_distances).astype(jnp.float32),
    )

    key, subkey = jax.random.split(key)
    key, subkey2 = jax.random.split(key)
    key, subkey3 = jax.random.split(key)

    scat_pos = _scat_pos_from_peaks(peaks, n_frames=1, is_2d=config.is_2d)

    scat_amp = (
        jnp.ones((1, 1)) * 1e-4
        + jax.random.normal(key, shape=(n_scat, 1), dtype=jnp.float32) * 1e-5
    )

    waveform_rfft_offset = jnp.tile(
        jnp.array([1.0, 0.0], dtype=jnp.float32), (n_fbins, 1)
    )

    opt_vars_opt = ParamsRegular[Physical](
        scat_pos=ScatPos(scat_pos),
        scat_amp=ScatAmpPerFrame(scat_amp),
        waveform_rfft_offset=WaveformRFFTOffset(waveform_rfft_offset),
        delta_pos=DeltaPos(
            jax.random.normal(subkey, shape=(n_scat, 1), dtype=jnp.float32) * 1e-5
        ),
        attenuation_coefficient=AttenuationCoefficient(
            jnp.array(attenuation_coef_np_m_hz, dtype=jnp.float32)
        ),
        waveform_params=WaveformParams(
            jax.random.normal(subkey2, shape=(n_scat, n_tx, 6), dtype=jnp.float32)
            * 1e-3
        ),
        sound_speed=SoundSpeed(jnp.array(sound_speed, dtype=jnp.float32)),
        directivity_falloff=DirecitivityFalloff(jnp.ones(6, dtype=jnp.float32)),
    ).to_scaled()

    with h5py.File("out/freqs.hdf5", "w") as f:
        f.create_dataset("freqs", data=static_vars.freqs)
        console.log("Saved freqs to out/freqs.hdf5.")

    #
    # ==========================================================================
    # Initialize the optimizer
    # ==========================================================================
    console.log("Initializing optimizer...")
    scheduler = optax.schedules.exponential_decay(
        init_value=config.learning_rate,
        transition_steps=config.n_iterations_per_frame,
        decay_rate=1e-1,
        transition_begin=(n_tx // 2 + 1) * config.n_iterations_per_frame,
        end_value=config.learning_rate / 2,
    )

    optimizer = optax.chain(
        optax.zero_nans(),
        optax.scale_by_adam(),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1),
    )

    if os.environ.get("RFULM_DEVELOPMENT", "0") == "1":
        steps = jnp.arange(config.n_iterations)
        lrs = jax.vmap(scheduler)(steps)
        fig, ax = plt.subplots()
        ax.plot(steps, lrs)
        ax.set_aspect("auto")

        plt.savefig("out/learning_rate_schedule.png")
        plt.close()
    opt_state = optimizer.init(opt_vars_opt)

    # ==========================================================================
    # Store it all in program_state
    # ==========================================================================
    program_state = ProgramState(
        key=jax.random.PRNGKey(config.seed),
        opt_vars=opt_vars_opt.to_physical(),
        static_vars=static_vars,
        opt_state=opt_state,
        optimizer=optimizer,
        indices_all=Indices.get_full(
            n_tx=n_tx,
            n_fbins=n_fbins,
            n_el=n_el,
        ),
        y_rfft_flat=y_rfft_flat,
        y_rfft_shape=y_rfft.shape,
        iteration=0,
        dimensions=Dimensions(
            n_frames=n_frames,
            n_scat=n_scat,
            n_tx=n_tx,
            n_el=n_el,
            n_fbins=n_fbins,
        ),
        config=config,
        rfft=rfft,
        beamformed_images=images_das.save("out/beamformed_images.hdf5"),
        forward_model=forward_model,
        parameters=parameters,
        total_scaling_factor=1.0,
        timestamps=timestamps,
    )

    return program_state


def initialize_waveform(config, parameters, images_das: Image, rfft: RFFT):
    if config.use_analytical_waveform:
        rfftfreq = jnp.fft.rfftfreq(4096, 1 / 250e6)

        t_peak = images_das.metadata["t_peak"]

        waveform_rfft_250 = get_pulse_spectrum_fn(
            fc=parameters.demodulation_frequency, n_period=3.0, fs=250e6
        )(rfftfreq)
        waveform_samples_250 = jnp.fft.irfft(waveform_rfft_250, n=4096)
        parameters.waveforms_one_way = np.tile(
            waveform_samples_250, (parameters.n_tx, 1)
        )
        parameters.waveforms_two_way = np.tile(
            waveform_samples_250, (parameters.n_tx, 1)
        )
        if False:
            fig, axes = plt.subplots(2, 1)
            ax_time, ax_freq = axes
            time = jnp.arange(waveform_samples_250.size) / 250e6
            ax_time.plot(time, waveform_samples_250)
            ax_time.set_xlabel("Time [s]")
            ax_time.set_ylabel("Amplitude")
            ax_freq.plot(rfftfreq, jnp.abs(waveform_rfft_250))
            ax_freq.set_xlabel("Frequency [Hz]")
            ax_freq.set_ylabel("Magnitude")
            plt.show()

    else:
        waveform_samples_250 = parameters.waveforms_two_way[0]
        t_peak = 0.0
    waveform_rfft = _prepare_waveform_rfft(
        waveform_samples=waveform_samples_250,
        sampling_frequency=parameters.sampling_frequency,
        freqs=rfft.rfftfreq(),
        tgc_per_256_samples=config.tgc_per_256_samples,
    )
    if False:
        fig, axes = plt.subplots(2, 1)
        ax_time, ax_freq = axes
        waveform_samples = rfft.irfft(waveform_rfft)
        time = jnp.arange(waveform_samples.size) / parameters.sampling_frequency
        waveform_samples = waveform_samples * jnp.exp(
            1j * 2 * jnp.pi * parameters.demodulation_frequency * time
        )
        pl(time, waveform_samples, ax=ax_time)
        ax_time.set_xlabel("Time [s]")
        pl(rfft.rfftfreq(), jnp.abs(waveform_rfft), ax=ax_freq)
        ax_freq.set_xlabel("Frequency [Hz]")
        plt.show()
    return waveform_rfft, parameters, t_peak


def get_pipeline_harmonic(baseband, demodulation_frequency):
    operations = []
    # Add the demodulate operation

    operations.append(BandPassFilter())
    operations.append(ApplyWindow())
    operations.append(Demodulate())
    operations.append(Downsample(factor=2))

    # Add beamforming ops
    operations.append(
        Beamform(
            beamformer="delay_multiply_and_sum",
            # num_patches=32,
            enable_pfield=False,
        ),
    )
    operations.append(zea.ops.ChannelsToComplex())

    return zea.ops.Pipeline(operations)


def preprocess_rf_data(rf_data, parameters, harmonic=False):
    parameters.bandwidth = 2.0e6
    baseband = rf_data.shape[-1] == 2
    operations = []
    if harmonic:
        operations.append(BandPassFilter())
        # operations.append(Downsample(factor=4))
    # Add the demodulate operation
    operations.append(ApplyWindow())
    if not baseband:
        operations.append(Demodulate())

    pipeline = Pipeline(operations)
    with suppress_logs():
        filtered_parameters = pipeline.prepare_parameters(
            parameters,
        )
    inputs = {pipeline.key: rf_data, **filtered_parameters}
    outputs = pipeline(**inputs)

    for key in outputs:
        if key == "data":
            continue
        if key in ("n_ax", "n_ch"):
            parameters[key] = int(outputs[key])
        else:
            parameters[key] = outputs[key]

    # parameters["tgc_gain_curve"] = parameters["tgc_gain_curve"][::4]
    parameters["center_frequency"] = parameters["demodulation_frequency"]

    return outputs[pipeline.output_key], parameters


def beamform_all_frames(
    rf_data,
    parameters,
    target_region,
    apply_lens_correction=False,
    lens_thickness=1e-3,
    t_peak=None,
    harmonic=False,
    f_number=1.0,
    beamformer="delay_multiply_and_sum",
    num_patches=32,
    enable_pfield=True,
):
    operations = []

    # Add beamforming ops
    operations.append(
        Beamform(
            beamformer=beamformer,
            num_patches=num_patches,
            enable_pfield=enable_pfield,
        ),
    )
    operations.append(zea.ops.ChannelsToComplex())

    if t_peak is None:
        parameters.t_peak = (
            np.array(
                zea.func.compute_time_to_peak_stack(
                    parameters.waveforms_two_way, parameters.center_frequency
                )
            )
            * 0.0
        )
    else:
        parameters.t_peak = np.array(t_peak)

    parameters.f_number = f_number
    parameters.lens_thickness = lens_thickness
    parameters.apply_lens_correction = apply_lens_correction
    parameters.lens_sound_speed = 1000.0

    parameters.xlims = target_region[:2]
    parameters.zlims = target_region[2:]
    parameters.pixels_per_wavelength = 2
    # scan.t_peak = np.array(2 / scan.demodulation_frequency)
    # scan.bandwidth = scan.center_frequency * 2
    # console.log(f"scan.t_peak: {scan.t_peak}")
    console.log(
        f"lens correction: {apply_lens_correction}, lens thickness: {lens_thickness}"
    )
    pipeline = Pipeline(operations)
    parameters_dict = {}

    frames = []
    for tx in range(rf_data.shape[0]):
        inputs = {pipeline.key: rf_data[tx : tx + 1][None]}
        parameters.selected_transmits = np.array([tx])
        with suppress_logs():
            parameters_dict = pipeline.prepare_parameters(
                parameters,
            )
        # parameters = _remove_redundant_parameters(parameters)
        parameters_dict["apply_lens_correction"] = apply_lens_correction
        outputs = pipeline(**inputs, **parameters_dict)

        frames.append(outputs[pipeline.output_key][0])
    im_das = np.stack(frames, axis=0)

    xlims = parameters.xlims
    zlims = parameters.zlims

    im_das = im_das * (
        1.0 + np.exp(-np.square(np.linspace(-3, 3, im_das.shape[-1]))[None, None])
    )

    image_sequence = Image(
        im_das,
        limits=(0, im_das.shape[0] - 1, *zlims, *xlims),
        labels=("frame", "z", "x"),
    ).add_metadata(key="t_peak", value=parameters_dict.get("t_peak")[0])

    console.log(f"t_peak: {parameters_dict.get('t_peak')[0]}")

    return image_sequence


def _remove_redundant_parameters(parameters):
    for key in (
        "zlims",
        "n_el",
        "center_frequency",
        "xlims",
    ):
        del parameters[key]
    return parameters


# ==============================================================================
# Helper functions
# ==============================================================================
def _scat_pos_from_peaks(peaks, n_frames, is_2d=True):
    initializer = [peaks[:, 0], peaks[:, 1]]
    if not is_2d:
        initializer.insert(1, jnp.zeros_like(peaks[:, 0]))
    scat_pos = jnp.stack(initializer, axis=-1)[:, None]
    return jnp.repeat(scat_pos, repeats=n_frames, axis=1)


def _get_planewave_time_offsets_2d(planewave_angles, probe_geometry, sound_speed):
    v = jnp.stack(
        [
            jnp.sin(planewave_angles),
            jnp.cos(planewave_angles),
        ],
        axis=-1,
    )

    # (n_tx, n_el, 2) @ (1, 2, n_el)
    delays = probe_geometry[None] @ v[:, :, None] / sound_speed  # (n_tx, n_el, 1)
    planewave_time_offsets = jnp.abs(jnp.max(delays, axis=(1, 2)))
    return planewave_time_offsets


def _prepare_rf(
    rf_data,
    tgc_gain_curve,
    sampling_frequency,
    rfft: RFFT,
    tgc_per_256_samples: float,
    demodulation_frequency: float,
) -> jnp.ndarray:
    assert isinstance(rf_data, (jnp.ndarray, np.ndarray))
    assert rf_data.ndim == 5  # (n_frames, n_tx, n_ax, n_el, 1)
    assert isinstance(tgc_gain_curve, (jnp.ndarray, np.ndarray))
    assert tgc_gain_curve.ndim == 1  # (n_ax,)
    assert isinstance(rfft, RFFT)
    assert isinstance(sampling_frequency, (float, int))
    assert isinstance(tgc_per_256_samples, (float, int))

    rf_data = rf_data[:, :, : tgc_gain_curve.size, :, :]

    if rf_data.shape[-1] == 1:
        rf_data = zea.func.ultrasound.demodulate(
            rf_data, demodulation_frequency, sampling_frequency
        )

    with h5py.File("out/rf_data.hdf5", "w") as f:
        f.create_dataset("rf_data", data=rf_data)
        console.log("Saved rf_data to out/rf_data.hdf5.")

    # --------------------------------------------------------------------------
    # Cut off the start with the internal reflection
    # --------------------------------------------------------------------------
    ax0, ax1 = 180 // 4, 200 // 4
    n_ax = rf_data.shape[2]

    window = np.ones(n_ax)
    window[:ax1] = hann_unnormalized(x=np.linspace(-ax1, 0, ax1), width=2 * (ax1 - ax0))
    rf_data = rf_data * window[None, None, :, None, None]

    # --------------------------------------------------------------------------
    # Axial RF normalization
    # --------------------------------------------------------------------------

    # First divide out the TGC gain curve to get the actual values
    rf_data /= tgc_gain_curve[None, None, :, None, None]

    # Compute the desired TGC gain curve
    tgc_gain = _compute_tgc_gain(tgc_per_256_samples, sampling_frequency)
    scaling = _compute_tgc_curve(tgc_gain, n_ax, sampling_frequency)

    # Apply the new TGC gain curve
    rf_data = rf_data * scaling[None, None, :, None, None]

    # rf_data = rf_data * jnp.exp(
    #     jnp.linspace(jnp.log(0), jnp.log(10), rf_data.shape[2])[
    #         None, None, :, None, None
    #     ]
    # )

    return rf_data


def _normalize_y_rfft(y_rfft):
    normalization_factor = jnp.percentile(jnp.abs(y_rfft), 99.9)
    y_rfft = y_rfft / normalization_factor
    return y_rfft, float(normalization_factor)


def _compute_tgc_curve(tgc_gain, n_ax, sampling_frequency):
    """Compute the TGC curve given by exp(tgc_alpha * t / sampling_frequency)."""

    t = jnp.arange(n_ax) / sampling_frequency
    tgc_gain_curve = jnp.exp(tgc_gain * t)
    return tgc_gain_curve


def _compute_tgc_gain(tgc_per_256_samples, sampling_frequency):
    """Computes the exponential scaling factor tgc_gain given the gain per 256
    samples. This tgc_gain is then used to compute the TGC curve as

    exp(tgc_gain * t).

    Parameters
    ----------
    tgc_per_256_samples : float
        Gain per 256 samples.
    sampling_frequency : float
        Sampling frequency [Hz].

    Returns
    -------
    tgc_gain : float
        The exponential scaling factor.
    """

    t_max = 256 / sampling_frequency
    tgc_gain = np.log(tgc_per_256_samples) / t_max
    return tgc_gain


def _probe_geometry_2d(probe_geometry):
    assert isinstance(probe_geometry, (np.ndarray, jnp.ndarray))
    assert probe_geometry.ndim == 2
    assert probe_geometry.shape[1] == 3
    return probe_geometry[:, np.array([0, 2])]


def _load_tgc_gain_curve(path):
    with h5py.File(path, "r") as f:
        try:
            tgc_gain_curve = f["scan"]["tgc_gain_curve"][:]
        except KeyError:
            n_ax = f["data"]["raw_data"].shape[2]
            tgc_gain_curve = np.ones(n_ax)
    return tgc_gain_curve.astype(np.float32)


def _load_waveform_samples(path):
    with h5py.File(path, "r") as f:
        waveform_samples = f["scan"]["waveforms_two_way"]["waveform_000"][:]

    return waveform_samples.astype(np.float32)


def _prepare_waveform_rfft(
    waveform_samples, sampling_frequency, freqs, tgc_per_256_samples: float
):
    waveform_samples *= _compute_tgc_curve(
        tgc_gain=_compute_tgc_gain(tgc_per_256_samples, sampling_frequency),
        n_ax=waveform_samples.size,
        sampling_frequency=250e6,
    )
    waveform_samples = jnp.pad(
        waveform_samples, (0, 4096 - waveform_samples.size), mode="constant"
    )
    fft_size_250MHz = _next_power_of_2(waveform_samples.size)

    waveform_rfft_250MHz = jnp.fft.rfft(waveform_samples, fft_size_250MHz, axis=0)
    freqs_250MHz = jnp.fft.rfftfreq(fft_size_250MHz, 1 / 250e6)

    waveform_rfft = _complex_interp(
        x_source=freqs_250MHz,
        y_source=waveform_rfft_250MHz,
        x_target=freqs,
    )

    waveform_rfft = waveform_rfft / jnp.max(jnp.abs(waveform_rfft))
    return waveform_rfft


def _complex_interp(x_source, y_source, x_target):
    """Interpolates complex values y_source defined at x_source to x_target.

    Parameters
    ----------
    x_source : jnp.ndarray
        The x-coordinates of the source points.
    y_source : jnp.ndarray
        The complex values at the source points.
    x_target : jnp.ndarray
        The x-coordinates of the target points.

    Returns
    -------
    y_target : jnp.ndarray
        The interpolated complex values at the target points.
    """
    assert jnp.iscomplexobj(y_source)

    y_real = jnp.interp(x_target, x_source, jnp.real(y_source))
    y_imag = jnp.interp(x_target, x_source, jnp.imag(y_source))
    y_target = y_real + 1j * y_imag
    return jnp.array(y_target).astype(jnp.complex64)


def _next_power_of_2(x):
    """Computes the next power of 2 of a number (e.g. 5.6 -> 8)."""
    return np.power(2, np.ceil(np.log2(x))).astype(int)


def detect_peaks_from_multiple_frames(
    images_das: Image,
    max_n_peaks: int,
    threshold: float,
    min_distance_between_peaks: float = 0.0,
):
    n_frames = images_das.shape[0]
    all_peaks = []
    all_intensities = []

    images_das = images_das.abs().normalize()

    for i in [n_frames // 2]:
        image = images_das[i]
        peaks, intensities = detect_peaks(
            np.abs(image.array).T, image.extent, threshold, min_distance_between_peaks
        )
        all_peaks.append(peaks)
        all_intensities.append(intensities)

    all_peaks = jnp.concatenate(all_peaks, axis=0)
    all_intensities = jnp.concatenate(all_intensities, axis=0)

    all_peaks, all_intensities = _sort_peaks(all_peaks, all_intensities)
    return all_peaks[:max_n_peaks], all_intensities[:max_n_peaks]


def _sort_peaks(peaks, intensities):
    sort_indices = jnp.argsort(intensities)[::-1]
    peaks = peaks[sort_indices]
    intensities = intensities[sort_indices]
    return peaks, intensities
