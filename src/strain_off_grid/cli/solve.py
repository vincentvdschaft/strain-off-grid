import os
from pathlib import Path
from time import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppresses TF/XLA C++ logs
os.environ["GRPC_VERBOSITY"] = "ERROR"  # optional, grpc noise
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import zea
from plotlib import STYLE_DARK, use_style

zea.init_device(verbose=True)
import argparse

from strain_off_grid import console, load_config
from strain_off_grid.solver import (
    initialize,
)
from strain_off_grid.solver.datatypes.config import SolverConfig
from strain_off_grid.solver.save_solutions import save_solution_from_structs
from strain_off_grid.solver.solver import compute_and_save_rf, solve_program


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        type=str,
        default=None,
        help="Path to the config file to use. If not provided, the default config/regular.toml will be used.",
    )
    parser.add_argument(
        "--first-frame",
        nargs="?",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--subtract-path",
        type=str,
        default="out/subtracted.hdf5",
        help="Path to the config file to use for subtraction.",
    )
    parser.add_argument(
        "--transmits",
        type=str,
        default=None,
        help="Transmits to use, e.g. '0-3' or '0 1 2 3' or 'all'.",
    )
    parser.add_argument(
        "--solution-path",
        type=str,
        default="out/solution.hdf5",
        help="Path to save the solution.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/cardiac.toml",
        help="Path to the config file to use.",
    )
    args = parser.parse_args()

    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )

    use_style(STYLE_DARK)

    console.rule("[bold yellow]RF-ULM Paper Example")

    if args.path is not None:
        overwrite_dict = {"input_file": args.path}
    else:
        overwrite_dict = {}

    if args.transmits is not None:
        overwrite_dict["transmits"] = args.transmits
    config = load_config(Path(args.config), SolverConfig, overwrite_dict=overwrite_dict)

    if args.first_frame is not None:
        config = config.update(first_frame=args.first_frame)

    out_dir = Path("out/")
    out_dir.mkdir(exist_ok=True)

    # ==============================================================================
    # Solving
    # ==============================================================================

    for path in Path(out_dir, "out-points/").glob("*.png"):
        path.unlink()

    with console.status("Initializing...") as status:
        if config.paris_model:
            console.log("[bold green]Using Paris model")
            program_state = initialize_paris(config)
        else:
            console.log("[bold green]Using regular model")
            program_state = initialize(config)

    console.log("Running optimization...")
    start_time = time()
    if True:
        program_state.opt_vars, program_state.opt_state = solve_program(
            program_state,
        )
    end_time = time()

    console.log(
        f"Time for {program_state.config.n_iterations} iterations with batch size {program_state.config.batch_size}: {end_time - start_time:.2f} s"
    )
    # ==============================================================================
    #
    # ==============================================================================
    console.log("Computing and saving RF...")

    # ======================================================================================
    # Check accuracy
    # ======================================================================================

    def _rotate_to_plane(positions):
        depth = jnp.linalg.norm(positions[..., 1:], axis=-1)
        return jnp.stack([positions[..., 0], jnp.zeros_like(depth), depth], axis=-1)

    try:
        with h5py.File(config.input_file, "r") as f:
            first_frame = program_state.config.first_frame
            n_frames = program_state.config.n_frames
            true_positions = []
            for n in range(first_frame, first_frame + n_frames):
                true_positions.append(
                    _rotate_to_plane(
                        f[f"/non_standard_elements/scatterer_positions/frame_{n:03d}"][
                            :
                        ]
                    )
                )

        wavelength = (
            program_state.static_vars.sound_speed
            / program_state.static_vars.center_frequency
        )
        all_matched_distances = []
        for n in range(program_state.dimensions.n_frames):
            estimated_positions = program_state.opt_vars.scat_pos.data[:, n]
            true_pos = true_positions[n]
            diffs = estimated_positions[None,] - true_pos[:, None]
            distances = jnp.linalg.norm(diffs, axis=-1)
            min_idx = jnp.argmin(distances, axis=1)
            matched_diffs = diffs[np.arange(diffs.shape[0]), min_idx]
            matched_distances = jnp.linalg.norm(matched_diffs, axis=-1)
            console.log(
                (jnp.round(matched_diffs / wavelength * 1000)).astype(jnp.int32)
            )
            console.log(
                f"Frame {n}: mean distance to true positions: {matched_distances.mean():.2e} m, median: {jnp.median(matched_distances):.2e} m"
            )
            all_matched_distances.append(matched_distances)
        all_matched_distances = jnp.concatenate(all_matched_distances)
        median = jnp.median(all_matched_distances)
        no_more_than_5_times_median = all_matched_distances <= 5 * median
        all_matched_distances = all_matched_distances[no_more_than_5_times_median]

        console.log(
            f"Overall mean distance to true positions: {all_matched_distances.mean():.2e} m, median: {jnp.median(all_matched_distances):.2e} m"
        )
    except Exception as e:
        console.log(f"[bold red]Could not compute accuracy: {e}")

    # ======================================================================================
    #
    # ======================================================================================
    opt_vars = program_state.opt_vars
    intensities = jnp.abs(jnp.mean(opt_vars.scat_amp.data, axis=1))
    sort_indices = jnp.argsort(intensities)[::-1]
    program_state.opt_vars = program_state.opt_vars.index_scatterers(sort_indices)

    save_solution_from_structs(
        args.solution_path, program_state.opt_vars, program_state, extensive=True
    )

    y_rfft = compute_and_save_rf(
        opt_vars=program_state.opt_vars,
        static_vars=program_state.static_vars,
        rfft=program_state.rfft,
        path=out_dir / "simulated.hdf5",
        forward_model=program_state.forward_model,
        parameters=program_state.parameters,
    )

    parameters = program_state.parameters
    raw_data_original = jnp.fft.ifft(
        program_state.y_rfft_flat.reshape(y_rfft.shape), axis=2
    )
    raw_data_fit = jnp.fft.ifft(y_rfft.reshape(y_rfft.shape), axis=2)
    raw_data = raw_data_original - raw_data_fit
    raw_data = jnp.stack([jnp.real(raw_data), jnp.imag(raw_data)], axis=-1)
    from strain_off_grid.solver.initialize import (
        _compute_tgc_curve,
        _compute_tgc_gain,
    )

    tgc_gain = _compute_tgc_gain(
        config.tgc_per_256_samples, parameters.sampling_frequency
    )
    tgc_gain_curve = _compute_tgc_curve(
        tgc_gain, raw_data.shape[2], parameters.sampling_frequency
    )
    parameters._params["tgc_gain_curve"] = tgc_gain_curve
    parameters.n_ax = raw_data.shape[2]
    print(parameters.selected_transmits)

    y_concat = jnp.concatenate(
        [
            program_state.y_rfft_flat.reshape(y_rfft.shape),
            y_rfft,
        ],
        axis=3,
    )
    with h5py.File(out_dir / "comparison.hdf5", "w") as f:
        arr = program_state.rfft.irfft(y_concat, axis=2)
        f.create_dataset(
            "y_rfft_concat", data=jnp.stack([jnp.real(arr), jnp.imag(arr)], axis=-1)
        )

    with h5py.File(out_dir / "comparison_yrfft.hdf5", "w") as f:
        arr = y_concat
        f.create_dataset(
            "y_rfft_concat", data=jnp.stack([jnp.real(arr), jnp.imag(arr)], axis=-1)
        )

    y_error = program_state.y_rfft_flat.reshape(y_rfft.shape) - y_rfft
    with h5py.File(out_dir / "error.hdf5", "w") as f:
        arr = program_state.rfft.irfft(y_error, axis=2)
        f.create_dataset("y_rfft_error", data=arr)
