"""Alternative cardiac simulation using zea's frequency-domain simulator.

Instead of running Field II in MATLAB (``run_simulate_cardiac.m``), this script
simulates the RF data entirely in Python with :func:`zea.simulator.simulate_rf`
and writes the result directly in the zea (USBMD) format.

It consumes the **same** scatterer file produced by
``generate_cardiac_scatterers.py`` (``scat_pos`` of shape
``(n_frames, n_scat, 3)`` and ``scat_amp`` of shape ``(n_frames, n_scat)``) and
uses the same acquisition parameters as ``run_simulate_cardiac.m`` so the two
back-ends are comparable.

``simulate_rf`` builds a dense ``(n_scat, n_el, n_el, n_freq)`` tensor, so the
scatterers are processed in batches (RF data is a linear superposition of the
per-scatterer responses) to keep memory bounded.
"""

import argparse
from pathlib import Path

import jax
import numpy as np
import zea
from rich.progress import Progress
from scipy.signal.windows import kaiser
from zea import init_device
from zea.beamform.delays import compute_t0_delays_focused, compute_t0_delays_planewave
from zea.data.file import CustomElement
from zea.data.spec import ProbeSpec, ScanSpec
from zea.simulator import get_pulse_spectrum_fn, simulate_rf

from strain_off_grid import (
    RectanglePhantom,
    ShortAxisPhantom,
    StaticPhantom,
    distances_to_edge,
)
from strain_off_grid.phantoms.dataclass_saving import load_dataclass


def main():
    init_device()

    def build_probe_geometry(n_el: int, pitch: float) -> np.ndarray:
        """Return the (n_el, 3) element positions of the linear array in meters."""
        x = (np.arange(n_el) - n_el / 2 + 0.5) * pitch
        return np.stack([x, np.zeros(n_el), np.zeros(n_el)], axis=1).astype(np.float32)

    def build_phantom_elements(positions: np.ndarray, amplitudes: np.ndarray) -> list:
        """Wrap the scatterer positions and amplitudes as custom elements."""
        return [
            CustomElement(
                name="scatterer_positions",
                data=positions,
                description="The positions of the scatterers in the phantom",
                unit="m",
                group_name="phantom",
            ),
            CustomElement(
                name="scatterer_amplitudes",
                data=amplitudes,
                description="The amplitudes of the scatterers in the phantom",
                unit="–",
                group_name="phantom",
            ),
        ]

    def construct_waveform_samples(
        center_frequency: float,
        sampling_frequency: float = 250e6,
        n_samples: int = 1024,
    ) -> np.ndarray:
        pulse_spectrum_fn = get_pulse_spectrum_fn(center_frequency, n_period=3)
        freqs = np.fft.rfftfreq(n_samples, 1 / sampling_frequency)
        pulse_spectrum = pulse_spectrum_fn(freqs)
        pulse_samples = np.fft.irfft(pulse_spectrum, n_samples)
        return pulse_samples.astype(np.float32)

    def _get_t0_delays(
        focal_type: str,
        probe_geometry: np.ndarray,
        focus_distances: np.ndarray,
        angles: np.ndarray,
    ) -> np.ndarray:
        n_tx = focus_distances.shape[0]
        transmit_origins = np.zeros((n_tx, 3), dtype=np.float32)
        if focal_type == "focused":
            return compute_t0_delays_focused(
                transmit_origins=transmit_origins,
                focus_distances=focus_distances,
                probe_geometry=probe_geometry,
                polar_angles=angles,
            )
        elif focal_type == "diverging":
            return compute_t0_delays_focused(
                transmit_origins=transmit_origins,
                focus_distances=focus_distances,
                probe_geometry=probe_geometry,
                polar_angles=angles,
            )
        elif focal_type == "planewave":
            return compute_t0_delays_planewave(
                probe_geometry=probe_geometry, polar_angles=angles
            )
        else:
            raise ValueError(f"Unknown focal type: {focal_type}")

    def get_phantom(phantom_type: str, args):
        if phantom_type == "cardiac":
            return CardiacPhantom(
                A=np.diag([1.0, 1.0, -1.0]),
                b=np.array([0.0, 0.0, 50e-3]),
                focal_length=30e-3,
            )
        elif phantom_type == "ring":
            return ShortAxisPhantom(
                A=np.eye(3),
                b=np.array([0.0, 0.0, 40e-3]),
                inner_diameter_min=30e-3,
                inner_diameter_max=40e-3,
                outer_diameter_min=40.5e-3,
                outer_diameter_max=50e-3,
                thickness_variation=0.0e-3,
                torsion_amplitude=np.deg2rad(12.0),
            )
        elif phantom_type == "polygon":
            # return PolygonPhantom.from_csv("phantom_framerate.csv")
            return load_dataclass("out/miccai_cardiac_phantom.hdf5")
        elif phantom_type == "static":
            return StaticPhantom(
                A=np.diag([1.0, 1.0, 1.0]),
                b=np.array([0.0, 0.0, 50e-3]),
            )
        elif phantom_type == "rectangle":
            return RectanglePhantom(
                b=np.array([0.0, 0.0, 50e-3]),
                width=5e-3,
                height=20e-3,
                max_vertical_strain=2.0,
                max_horizontal_strain=2.0,
                frequency=60.0 / 60.0,
            )

        raise ValueError(f"Unknown phantom type: {phantom_type}")

    def get_focus_distances(
        focal_type: str,
        n_tx: int,
        focal_distance_focused: float = 73.92e-3,
        focal_distance_diverging: float = -10e-3,
    ) -> np.ndarray:
        if focal_type in ["focused", "diverging"]:
            return np.full(
                n_tx,
                focal_distance_focused
                if focal_type == "focused"
                else focal_distance_diverging,
                dtype=np.float32,
            )
        elif focal_type == "planewave":
            return np.zeros(n_tx, dtype=np.float32)
        else:
            raise ValueError(f"Unknown focal type: {focal_type}")

    # ======================================================================================
    #
    # ======================================================================================
    def parse_arguments() -> argparse.Namespace:
        """Parse the command line arguments."""
        parser = argparse.ArgumentParser(
            description="Simulate cardiac RF data with zea"
        )

        parser.add_argument(
            "--out",
            type=str,
            default="source/simulated_phantom.hdf5",
            help="Path to the output zea HDF5 file",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=1024,
            help="Number of scatterers per simulate_rf call",
        )
        parser.add_argument(
            "--frames", type=int, default=16, help="Number of frames to simulate"
        )
        parser.add_argument(
            "--scatterers", type=int, default=200, help="Number of scatterers"
        )
        parser.add_argument(
            "--slice-angle",
            type=float,
            default=np.pi / 2,
            help="Half-width (rad) of the elevation slab around the imaging plane",
        )
        parser.add_argument(
            "--prf",
            type=float,
            default=1000.0,
            help="Pulse repetition frequency in Hz (transmit spacing within a frame)",
        )
        parser.add_argument(
            "--angle-delta-deg",
            type=float,
            default=1.0,
            help="Maximum transmit angle in degrees",
        )
        parser.add_argument(
            "--focal-type",
            type=str,
            default="diverging",
            choices=["diverging", "focused", "planewave"],
            help="Use diverging-wave transmits (virtual source behind the array) "
            "instead of focused transmits",
        )
        parser.add_argument("--seed", type=int, default=0, help="Random seed")
        parser.add_argument(
            "--plot", action="store_true", help="Plot the scatterers and exit"
        )
        parser.add_argument(
            "--n-tx", type=int, default=3, help="Number of transmits per frame"
        )
        parser.add_argument(
            "--n-frames", type=int, default=1, help="Number of frames to simulate"
        )
        parser.add_argument(
            "--frame-rate", type=float, default=None, help="Frame rate in Hz"
        )
        parser.add_argument(
            "--phantom",
            type=str,
            default="cardiac",
            help="Type of phantom to simulate",
        )
        return parser.parse_args()

    # Acquisition parameters (matched to run_simulate_cardiac.m).
    N_EL = 80
    CENTER_FREQUENCY = 3.9e6
    SAMPLING_FREQUENCY = CENTER_FREQUENCY * 4
    APERTURE_WIDTH = 20e-3
    PITCH = APERTURE_WIDTH / (N_EL - 1)
    ELEMENT_WIDTH = PITCH * 0.9
    ELEMENT_HEIGHT = 10e-3
    SOUND_SPEED = 1540.0
    N_AX = 4096
    ATTENUATION_COEF = 0.062  # dB/cm/MHz

    """Run the zea simulation pipeline and save the result in zea format."""
    args = parse_arguments()

    n_tx = args.n_tx
    n_frames = args.n_frames
    n_tx_frames = n_tx * n_frames

    np.random.seed(args.seed)
    phantom = get_phantom(args.phantom, args)
    parameters_dict = {
        "probe_geometry": build_probe_geometry(n_el=N_EL, pitch=PITCH),
        "polar_angles": (np.arange(n_tx).astype(np.float32) - (n_tx - 1) / 2)
        * np.deg2rad(args.angle_delta_deg),
        "focus_distances": get_focus_distances(
            focal_type=args.focal_type,
            n_tx=n_tx,
        ),
        "focal_type": args.focal_type,
        "initial_times": np.zeros(n_tx, dtype=np.float32),
        "sampling_frequency": np.float32(SAMPLING_FREQUENCY),
        "center_frequency": np.float32(CENTER_FREQUENCY),
        "demodulation_frequency": np.float32(CENTER_FREQUENCY),
        "sound_speed": np.float32(SOUND_SPEED),
        "tx_apodizations": np.repeat(
            np.array(kaiser(N_EL, beta=2.0), dtype=np.float32)[None], n_tx, axis=0
        ),
    }
    parameters_dict["t0_delays"] = _get_t0_delays(
        focal_type=args.focal_type,
        probe_geometry=parameters_dict["probe_geometry"],
        focus_distances=parameters_dict["focus_distances"],
        angles=parameters_dict["polar_angles"],
    )

    positions = phantom.sample_points(n_points=args.scatterers)
    intensities = np.ones(positions.shape[0], dtype=np.float32)
    intensities = np.random.rayleigh(scale=2.0, size=positions.shape[0]).astype(
        np.float32
    )

    try:
        selected_indices = np.random.choice(
            np.arange(positions.shape[0]), size=args.scatterers // 5, replace=False
        )

        intensities[selected_indices] = (intensities[selected_indices] + 1.0) * 5.0
    except ValueError:
        # If args.scatterers is too small, skip the bright scatterer selection
        pass
    if args.phantom == "polygon":
        try:
            distances = distances_to_edge(
                positions[:, np.array([0, 2])], phantom.points[0]
            )
            intensities = intensities * (
                1.0 + 2.0 * np.exp(-np.square(distances / 2e-3)).astype(np.float32)
            )
        except Exception as e:
            print(f"Warning: Could not compute distances to polygon edge: {e}")
    elif args.phantom == "ring":
        distances = phantom.distances_to_edge(positions, 0.0)
        intensities = intensities * (
            1.0 + 2.0 * np.exp(-np.square(distances / 0.5e-3)).astype(np.float32)
        )

    if args.frame_rate is None:
        args.frame_rate = args.n_frames / phantom.period

    dt = 1 / args.prf
    time_between_last_tx_and_next_frame = 1 / args.frame_rate - (n_tx - 1) * dt
    t = 0.0
    rf_data = []
    batch_size = args.batch_size
    n_batches = (len(positions) + batch_size - 1) // batch_size
    with Progress() as progress:
        frame_task = progress.add_task("Frames", total=n_frames)
        tx_task = progress.add_task("Transmissions", total=n_tx)
        batch_task = progress.add_task("Batches", total=n_batches)
        for frame in range(n_frames):
            progress.reset(tx_task)
            for tx in range(n_tx):
                progress.reset(batch_task)
                print(
                    f"Translating scatterers to time {t:.5f} s for frame {frame}, tx {tx}"
                )
                positions_current = phantom.translate_to_time(positions, t)

                positions_current_distance_to_xz_plane = np.abs(positions_current[:, 1])

                rf_data_tx = None
                for start in range(0, len(positions), batch_size):
                    chunk = slice(start, start + batch_size)
                    part = jax.jit(simulate_rf, static_argnums=(3, 7))(
                        scatterer_positions=positions_current[chunk],
                        scatterer_magnitudes=intensities[chunk],
                        t0_delays=parameters_dict["t0_delays"][tx : tx + 1],
                        initial_times=parameters_dict["initial_times"][tx : tx + 1],
                        probe_geometry=parameters_dict["probe_geometry"],
                        sampling_frequency=parameters_dict["sampling_frequency"],
                        center_frequency=parameters_dict["center_frequency"],
                        sound_speed=parameters_dict["sound_speed"],
                        apply_lens_correction=False,
                        lens_thickness=1e-3,
                        lens_sound_speed=1000.0,
                        n_ax=N_AX,
                        attenuation_coef=ATTENUATION_COEF,
                        element_width=ELEMENT_WIDTH,
                        tx_apodizations=parameters_dict["tx_apodizations"][tx : tx + 1],
                    )
                    part = np.asarray(part)
                    rf_data_tx = part if rf_data_tx is None else rf_data_tx + part
                    progress.advance(batch_task)
                rf_data.append(rf_data_tx)
                progress.advance(tx_task)
                if tx < n_tx - 1:
                    t += dt

            t += time_between_last_tx_and_next_frame
            progress.advance(frame_task)

    raw_data = np.stack(rf_data, axis=0).astype(np.float32)
    raw_data = raw_data.reshape(n_frames, n_tx, *raw_data.shape[2:])

    waveforms = np.tile(
        construct_waveform_samples(
            center_frequency=CENTER_FREQUENCY,
            sampling_frequency=SAMPLING_FREQUENCY,
            n_samples=1024,
        ),
        (n_tx, 1),
    )

    time_to_next_transmit = np.full((n_frames, n_tx), dt, dtype=np.float32)
    time_to_next_transmit[:, -1] = time_between_last_tx_and_next_frame

    scan_spec = ScanSpec(
        sampling_frequency=SAMPLING_FREQUENCY,
        center_frequency=CENTER_FREQUENCY,
        demodulation_frequency=CENTER_FREQUENCY,
        initial_times=parameters_dict["initial_times"],
        t0_delays=parameters_dict["t0_delays"],
        tx_apodizations=parameters_dict["tx_apodizations"],
        focus_distances=parameters_dict["focus_distances"],
        transmit_origins=np.zeros((n_tx, 3), dtype=np.float32),
        polar_angles=parameters_dict["polar_angles"],
        azimuth_angles=np.zeros(n_tx, dtype=np.float32),
        sound_speed=SOUND_SPEED,
        waveforms_two_way=waveforms,
        waveforms_one_way=waveforms,
        time_to_next_transmit=time_to_next_transmit,
    )
    probe_spec = ProbeSpec(
        name="S5-1",
        probe_geometry=parameters_dict["probe_geometry"],
        probe_center_frequency=CENTER_FREQUENCY,
        element_width=ELEMENT_WIDTH,
        element_height=ELEMENT_HEIGHT * 5,
        probe_bandwidth_percent=100,
    )
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    zea.File.create(
        output_path,
        data={"raw_data": raw_data},
        scan=scan_spec,
        probe=probe_spec,
        overwrite=True,
    )

    phantom.to_hdf5(output_path, group="/custom/phantom")
