import os
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import zea
from jax import jit
from zea.data.spec import ProbeSpec, ScanSpec

from strain_off_grid import console
from strain_off_grid.solver import ProgramState
from strain_off_grid.solver.custom_rfft import RFFT
from strain_off_grid.solver.datatypes import (
    Indices,
    ParamsType,
    StaticVars,
    StaticVarsType,
)
from strain_off_grid.solver.datatypes.indices import Indices
from strain_off_grid.solver.datatypes.params import (
    ParamsBase,
)
from strain_off_grid.solver.save_solutions import save_solution_from_structs


def solve_program(program_state):
    _input_warnings(program_state)
    return solve(
        opt_vars=program_state.opt_vars,
        static_vars=program_state.static_vars,
        indices=program_state.indices_all,
        y_rfft_flat=program_state.y_rfft_flat,
        optimizer=program_state.optimizer,
        n_iterations=program_state.config.n_iterations,
        n_iterations_per_frame=program_state.config.n_iterations_per_frame,
        batch_size=program_state.config.batch_size,
        opt_state=program_state.opt_state,
        freeze_waveform=program_state.config.freeze_waveform,
        remove_weak_scatterers_threshold=program_state.config.remove_weak_scatters_threshold,
        program_state=program_state,
        forward_model=program_state.forward_model,
    )


def _soft_threshold(x, threshold):
    """Applies soft thresholding to the input array."""
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0.0)


def _sample_indices(static_vars, indices_all, key, n_frames, batch_size):
    n_el = static_vars.n_el
    if False:
        idx = indices_all.sample_all_elements_and_frames(
            key,
            n_samples=batch_size // n_el,
            n_el=n_el,
            n_frames=n_frames,
        )
    else:
        probability = jnp.exp(
            -0.5
            * jnp.square(
                (static_vars.freqs[indices_all.fbins] - static_vars.center_frequency)
                / static_vars.center_frequency
            )
        )
        idx = jax.random.choice(key, indices_all.size, (batch_size,), p=probability)
    return idx


def get_update(prior_scaling, program_state: ProgramState) -> Callable:
    loss_fn = get_loss_fn(forward_model=program_state.forward_model)
    accumulation_steps = program_state.config.gradient_accumulation_steps

    def compute_grad_for_batch(key, opt_vars_opt, static_vars, y_rfft, iteration):
        n_tx = opt_vars_opt.scat_pos.shape[1]
        n_fbins = static_vars.n_fbins
        key, key2, key3, key4 = jax.random.split(key, 4)
        tx_idx = jax.random.randint(key3, (), 0, n_tx)

        p = jnp.exp(-(jnp.fft.fftshift(jnp.linspace(-2, 2, n_fbins)) ** 2))
        fbin_idx = jax.random.choice(key4, n_fbins, p=p)

        return jax.value_and_grad(loss_fn)(
            opt_vars_opt,
            static_vars,
            tx_idx,
            fbin_idx,
            y_rfft[tx_idx, fbin_idx],
            iteration,
            program_state.config.n_iterations,
            program_state.config.n_iterations_per_frame,
            prior_scaling,
        )

    def update(carry, inputs):
        (
            key,
            opt_vars_opt,
            static_vars,
            opt_state,
            indices_all,
            y_rfft_flat,
            iteration,
        ) = carry

        assert isinstance(opt_vars_opt, ParamsBase)
        assert isinstance(static_vars, StaticVars)
        assert isinstance(indices_all, Indices)

        split_keys = jax.random.split(key, accumulation_steps + 1)
        key, accumulation_keys = split_keys[0], split_keys[1:]

        n_frames = opt_vars_opt.scat_pos.shape[1]
        n_tx = opt_vars_opt.scat_pos.shape[1]
        n_fbins = static_vars.n_fbins
        n_el = static_vars.n_el
        y_rfft = jnp.reshape(y_rfft_flat, (n_tx, n_fbins, n_el))

        batched_grad_fn = jax.vmap(
            compute_grad_for_batch,
            in_axes=(0, None, None, None, None),
        )
        losses, grads = batched_grad_fn(
            accumulation_keys, opt_vars_opt, static_vars, y_rfft, iteration
        )
        total_loss = jnp.sum(losses)
        total_grad = jax.tree_util.tree_map(lambda g: jnp.sum(g, axis=0), grads)

        mean_loss = total_loss / accumulation_steps
        mean_grad = scale_grads(total_grad, 1.0 / accumulation_steps)

        updates, opt_state = program_state.optimizer.update(mean_grad, opt_state)
        opt_vars_opt = optax.apply_updates(opt_vars_opt, updates)

        # opt_vars = opt_vars_opt.to_physical()
        # opt_vars.scat_amp.data = _soft_threshold(
        #     opt_vars.scat_amp.data, program_state.config.l1_regularization
        # )
        # opt_vars_opt = opt_vars.to_scaled()

        return (
            key,
            opt_vars_opt,
            static_vars,
            opt_state,
            indices_all,
            y_rfft_flat,
            iteration + 1,
        ), mean_loss

    return update


def solve(
    opt_vars: ParamsType,
    static_vars: StaticVars,
    indices: Indices,
    y_rfft_flat: jnp.ndarray,
    optimizer: optax.GradientTransformation,
    n_iterations: int,
    n_iterations_per_frame: int,
    batch_size: int,
    forward_model,
    opt_state=None,
    freeze_waveform: bool = False,
    program_state=None,
    remove_weak_scatterers_threshold: float = 1e-3,
):
    """Solves the optimization problem."""
    assert isinstance(opt_vars, ParamsBase)
    assert isinstance(static_vars, StaticVars)
    assert isinstance(indices, Indices)
    assert isinstance(y_rfft_flat, jnp.ndarray)
    assert indices.transmits.shape == y_rfft_flat.shape
    assert indices.fbins.shape == y_rfft_flat.shape
    assert indices.elements.shape == y_rfft_flat.shape
    assert y_rfft_flat.ndim == 1

    development_mode = os.environ.get("RFULM_DEVELOPMENT") == "1"
    if development_mode:
        console.log("[yellow]Development mode enabled[/yellow]")

    key = jax.random.PRNGKey(0)
    n_tx = static_vars.n_tx

    opt_vars_opt = opt_vars.to_scaled()

    if opt_state is None:
        opt_state = optimizer.init(opt_vars_opt)

    y_rfft_flat_middle_frames, indices_middle_frames = _get_middle_yrfft_and_indices(
        y_rfft_flat,
        indices,
        n_tx_total=static_vars.n_tx,
        n_tx_current=opt_vars_opt.scat_pos.shape[1],
    )

    carry = _construct_carry(
        key=key,
        opt_vars_opt=opt_vars_opt,
        static_vars=static_vars,
        opt_state=opt_state,
        indices_all=indices_middle_frames,
        y_rfft_flat=y_rfft_flat_middle_frames,
        iteration=0,
    )

    iteration = 0

    prior_scaling = optax.exponential_decay(
        1e-4,
        n_iterations_per_frame,
        1e3,
        transition_begin=(n_tx // 2 + 1) * n_iterations_per_frame,
        end_value=1e0,
    )

    if development_mode:
        save_solution_from_structs(
            "out/latest_solution.hdf5",
            opt_vars_opt.to_physical(),
            program_state,
            extensive=True,
        )
        save_solution_from_structs(
            f"out/solutions/it_{program_state.solve_index:02d}_{iteration:06d}_solution.hdf5",
            opt_vars_opt.to_physical(),
            program_state,
            extensive=True,
        )
        # Plot prior scaling
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        x = np.arange(n_iterations)
        y = prior_scaling(x)
        ax.plot(x, y)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Prior Scaling")
        ax.set_title("Prior Scaling Schedule")
        plt.tight_layout()
        plt.savefig("out/prior_scaling.png", bbox_inches="tight")
        plt.close(fig)
    assert program_state is not None
    update = get_update(prior_scaling=prior_scaling, program_state=program_state)
    losses = []
    while iteration < n_iterations:
        length = min(n_iterations_per_frame, n_iterations - iteration)

        with console.status(
            f"Optimizing iteration {iteration}-{iteration + length}..."
        ):
            t0 = time.perf_counter()
            carry, new_losses = jax.lax.scan(update, carry, length=length)
            new_losses.block_until_ready()
            elapsed = time.perf_counter() - t0
            console.log(
                f"  {length} iterations in {elapsed:.1f}s "
                f"({elapsed / length * 1000:.1f} ms/it)"
            )

        if np.isnan(np.array(new_losses)).any():
            nan_iteration = iteration + np.where(np.isnan(np.array(new_losses)))[0][0]
            console.log(
                f"[bold red]NaN detected in losses at iteration {nan_iteration}.[/bold red]"
            )
        losses.extend(new_losses)

        (key, opt_vars_opt, static_vars, opt_state, _, _, iteration) = carry
        console.log(f"iteration: {iteration}")

        if development_mode:
            save_solution_from_structs(
                f"out/solutions/it_{program_state.solve_index:02d}_{iteration:06d}_solution.hdf5",
                opt_vars_opt.to_physical(),
                program_state,
                extensive=True,
            )
            save_solution_from_structs(
                "out/latest_solution.hdf5",
                opt_vars_opt.to_physical(),
                program_state,
                extensive=True,
            )
        is_not_last_solve = iteration < n_iterations
        if is_not_last_solve:
            opt_vars = opt_vars_opt.to_physical()
            mask_weak = mask_remove_weak_scatterers(
                opt_vars, threshold=remove_weak_scatterers_threshold
            )
            if False:
                mask_close = mask_merge_scatterers_that_are_always_close(
                    opt_vars, threshold=static_vars.wavelength * 1.5
                )
            else:
                mask_close = jnp.ones_like(
                    mask_weak, dtype=bool
                )  # Disable merging for now
            mask = jnp.logical_and(mask_weak, mask_close)
            console.log(
                f"Applying mask: {jnp.sum(~mask)}/{mask.size} scatterers removed."
            )
            opt_vars_opt, opt_state = apply_mask(mask, opt_vars_opt, opt_state)

            console.log("Adding another frame to optimization variables.")
            if opt_vars_opt.scat_pos.shape[1] < n_tx:
                opt_vars_opt = opt_vars_opt.add_frame(n_tx)

            # opt_state = _reinit_opt_state(opt_vars_opt, optimizer, opt_state)
            schedule_state = opt_state[-2]
            opt_state = list(optimizer.init(opt_vars_opt))
            opt_state[-2] = schedule_state
            opt_state = tuple(opt_state)

            y_rfft_flat_middle_frames, indices_middle_frames = (
                _get_middle_yrfft_and_indices(
                    y_rfft_flat,
                    indices,
                    n_tx_total=static_vars.n_tx,
                    n_tx_current=opt_vars_opt.scat_pos.shape[1],
                )
            )

            carry = _construct_carry(
                key=key,
                opt_vars_opt=opt_vars_opt,
                static_vars=static_vars,
                opt_state=opt_state,
                indices_all=indices_middle_frames,
                y_rfft_flat=y_rfft_flat_middle_frames,
                iteration=iteration,
            )

            # print(f"lam: {carry[-1]}")

        if development_mode:
            with console.status("Saving loss plot..."):
                fig, ax = plt.subplots()
                step = 25
                losses_subsampled = losses[::step]
                iters = np.arange(len(losses_subsampled)) * step
                ax.plot(iters, 20 * np.log10(losses_subsampled))
                ax.set_xlabel("Iteration [-]")
                ax.set_ylabel("Loss [dB]")
                plt.tight_layout()
                plt.savefig("out/loss.png", bbox_inches="tight")
                plt.close(fig)
            console.log("Loss plot saved.")

    _, opt_vars_opt, _, _, _, _, _ = carry

    if development_mode:
        save_solution_from_structs(
            "out/latest_solution.hdf5",
            opt_vars_opt.to_physical(),
            program_state,
            extensive=True,
        )

    return opt_vars_opt.to_physical(), opt_state


def smooth_tracks(opt_vars: ParamsType):
    """Applies a simple moving average filter to the scatterer tracks."""

    scat_pos = opt_vars.scat_pos.data
    scat_pos_padded = jnp.concatenate(
        [scat_pos[:, :1, :], scat_pos, scat_pos[:, -1:, :]], axis=1
    )
    kernel = jnp.array([0.25, 0.5, 0.25])
    scat_pos_smoothed = jnp.apply_along_axis(
        lambda m: jnp.convolve(m, kernel, mode="valid"), axis=1, arr=scat_pos_padded
    )
    opt_vars.scat_pos.data = scat_pos_smoothed

    scat_amp = opt_vars.scat_amp.data
    scat_amp_padded = jnp.concatenate(
        [scat_amp[:, :1], scat_amp, scat_amp[:, -1:]], axis=1
    )
    scat_amp_smoothed = jnp.apply_along_axis(
        lambda m: jnp.convolve(m, kernel, mode="valid"), axis=1, arr=scat_amp_padded
    )
    opt_vars.scat_amp.data = scat_amp_smoothed
    return opt_vars


def scale_grads(grads, factor):
    """Scales the gradients by the given factor."""
    return jax.tree_util.tree_map(lambda g: g * factor, grads)


def _reinit_opt_state(opt_vars_opt: ParamsBase, optimizer, opt_state):
    new_opt_state = optimizer.init(opt_vars_opt)
    final_opt_state = []
    for new_state, old_state in zip(new_opt_state, opt_state):
        if isinstance(new_state, optax.ScaleByAdamState) and isinstance(
            old_state, optax.ScaleByAdamState
        ):
            final_state = optax.ScaleByAdamState(
                count=old_state.count,
                mu=new_state.mu.overwrite_center_frames(old_state.mu),
                nu=new_state.nu.overwrite_center_frames(old_state.nu),
            )
        else:
            final_state = new_state
        final_opt_state.append(final_state)
    return tuple(final_opt_state)


def _get_middle_yrfft_and_indices(
    y_rfft_flat_total, indices_all, n_tx_total, n_tx_current
):
    middle = n_tx_total // 2
    n_offset = n_tx_current // 2
    mask = jnp.abs(indices_all.transmits - middle) <= n_offset

    return y_rfft_flat_total[mask], Indices.get_full(
        n_tx=n_tx_current,
        n_fbins=indices_all.fbins.max() + 1,
        n_el=indices_all.elements.max() + 1,
    )


def _construct_carry(
    key: jnp.ndarray,
    opt_vars_opt: ParamsType,
    static_vars: StaticVars,
    opt_state: Any,
    indices_all: Indices,
    y_rfft_flat: jnp.ndarray,
    iteration: int,
) -> tuple[jnp.ndarray, ParamsType, StaticVars, Any, Indices, jnp.ndarray, int]:
    assert isinstance(opt_vars_opt, ParamsBase)
    assert isinstance(static_vars, StaticVars)
    assert isinstance(indices_all, Indices)
    assert isinstance(y_rfft_flat, jnp.ndarray)

    return (
        key,
        opt_vars_opt,
        static_vars,
        opt_state,
        indices_all,
        y_rfft_flat,
        iteration,
    )


def get_loss_fn(
    forward_model: Callable,
) -> Callable:
    def loss_fn(
        opt_vars_opt: ParamsType,
        static_vars: StaticVars,
        tx_idx: int,
        fbin_idx: int,
        y_rfft: jnp.ndarray,
        iteration: int,
        n_iterations: int,
        n_iterations_per_frame: int,
        prior_scaling,
    ):
        assert isinstance(opt_vars_opt, ParamsBase)
        assert isinstance(static_vars, StaticVars)

        opt_vars = opt_vars_opt.to_physical()

        y_rfft_hat = forward_model(opt_vars, static_vars, tx_idx, fbin_idx)

        mse_loss = jnp.mean(jnp.abs(y_rfft_hat - y_rfft) ** 2)

        prior_loss = compute_prior_loss(
            opt_vars,
            iteration,
            n_iterations,
            n_iterations_per_frame,
            static_vars,
            prior_scaling,
            l1_regularization=static_vars.l1_regularization,
            velocity_range=static_vars.expected_velocity_range,
        )

        total_loss = mse_loss + prior_loss

        return total_loss

    return loss_fn


def _norm(x, axis=-1):
    return jnp.sqrt(jnp.sum(x**2, axis=axis) + 1e-8)


def _prior_annealing_ramp(iteration, n_iterations_per_frame):
    """Ramps from 0→1 within each n_iterations_per_frame block, resetting every new frame."""
    position_in_block = iteration % n_iterations_per_frame
    return position_in_block / n_iterations_per_frame


# Relative weights of the prior loss terms.
PRIOR_WEIGHT_SMOOTHNESS = 1e-2 * 0.0
PRIOR_WEIGHT_VELOCITY_RANGE = 1e-2 * 0.0
PRIOR_WEIGHT_AMPLITUDE_SMOOTHNESS = 1e-3
PRIOR_WEIGHT_WAVEFORM_POWER = 1e-1
PRIOR_WEIGHT_DIRECTION_COHERENCE = 1e-2

# Neighbourhood radius (in wavelengths) over which scatterers are expected to
# share a common direction of motion.
DIRECTION_COHERENCE_RADIUS_WL = 5.0


def _velocities_in_wavelengths_per_frame(scat_pos_data, wavelength):
    """Frame-to-frame displacement of each scatterer, expressed in wavelengths."""
    return jnp.diff(scat_pos_data / wavelength, axis=1)


# def _path_smoothness_loss(velocities, min_speed_wl_per_frame=1e-1):
#     norms = jnp.linalg.norm(velocities, axis=-1, keepdims=True)
#     directions = velocities / jnp.where(norms > 1e-3, norms, 1e-3)

#     cosine_direction_change = jnp.sum(
#         directions[:, 1:, :] * directions[:, :-1, :], axis=-1
#     )
#     loss = jnp.clip(0.8 - cosine_direction_change, 0, None)
#     loss = jnp.where(
#         jnp.linalg.norm(velocities[:, 1:, :], axis=-1) < min_speed_wl_per_frame, 0, loss
#     )
#     return jnp.mean(loss)


def _path_smoothness_loss(velocities, min_speed_wl_per_frame=1e-1):
    accelerations = jnp.diff(velocities, axis=1)
    jerks = jnp.linalg.norm(jnp.diff(accelerations, axis=1), axis=-1)
    return jnp.sum(jnp.square(jerks))


def _velocity_range_loss(speeds, velocity_range):
    """Soft band penalty: zero inside the expected speed range, quadratic outside it."""
    min_speed, max_speed = velocity_range
    too_slow = jnp.square(jax.nn.relu(min_speed - speeds))
    too_fast = jnp.square(jax.nn.relu(speeds - max_speed))
    return jnp.mean(too_slow + too_fast)


def _path_prior_loss(opt_vars, n_frames, ramp, wavelength, velocity_range):
    """Keeps scatterer speeds within the expected range and trajectories smooth."""
    if n_frames < 2:
        return 0.0
    velocities = _velocities_in_wavelengths_per_frame(
        opt_vars.scat_pos.data, wavelength
    )
    speeds = _norm(velocities, axis=-1)
    loss = (
        PRIOR_WEIGHT_VELOCITY_RANGE
        * ramp
        * _velocity_range_loss(speeds, velocity_range)
    )
    if n_frames >= 3:
        loss += PRIOR_WEIGHT_SMOOTHNESS * ramp * _path_smoothness_loss(velocities)
    return loss


def _amplitude_smoothness_loss(opt_vars, n_frames, ramp):
    """Penalizes large frame-to-frame changes in scatterer amplitude."""
    if n_frames < 2:
        return 0.0
    amplitude_diff = jnp.diff(opt_vars.scat_amp.data, axis=1)
    return PRIOR_WEIGHT_AMPLITUDE_SMOOTHNESS * ramp * jnp.mean(jnp.abs(amplitude_diff))


def _unit_directions(vectors):
    """Normalizes each vector to unit length, leaving near-zero vectors small."""
    norms = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / (norms + 1e-8)


def _mean_positions(scat_pos_data):
    """Average position of each scatterer across frames."""
    return jnp.mean(scat_pos_data, axis=1)


def _average_directions(scat_pos_data, wavelength):
    """Unit-length mean direction of motion of each scatterer."""
    velocities = _velocities_in_wavelengths_per_frame(scat_pos_data, wavelength)
    return _unit_directions(jnp.mean(velocities, axis=1))


def _pairwise_distances(positions):
    """Euclidean distance between every pair of scatterer positions."""
    differences = positions[:, None, :] - positions[None, :, :]
    return _norm(differences, axis=-1)


def _pairwise_closeness(positions, radius):
    """Weight in [0, 1] that decays with distance, with the self-pairs removed."""
    distances = _pairwise_distances(positions)
    weights = jnp.exp(-jnp.square(distances / radius))
    n_scatterers = positions.shape[0]
    return weights * (1.0 - jnp.eye(n_scatterers))


def _pairwise_direction_disagreement(directions):
    """One minus the cosine similarity between every pair of directions."""
    return 1.0 - directions @ directions.T


def _direction_coherence_loss(opt_vars, wavelength, radius):
    """Penalizes nearby scatterers whose average directions of motion disagree."""
    if opt_vars.scat_pos.shape[1] < 2:
        return 0.0
    positions = _mean_positions(opt_vars.scat_pos.data)
    directions = _average_directions(opt_vars.scat_pos.data, wavelength)
    weights = _pairwise_closeness(positions, radius)
    disagreement = _pairwise_direction_disagreement(directions)
    return jnp.sum(weights * disagreement) / (jnp.sum(weights) + 1e-8)


def _waveform_power_loss(opt_vars, static_vars):
    """Keeps the fitted waveform's average power close to unity."""
    waveform_fft = opt_vars.waveform_rfft_offset.compute_waveform(
        static_vars.waveform_rfft
    )
    waveform_power = jnp.mean(jnp.square(jnp.abs(waveform_fft)))
    return jnp.square(waveform_power - 1.0)


def compute_prior_loss(
    opt_vars: ParamsType,
    iteration: int,
    n_iterations: int,
    n_iterations_per_frame: int,
    static_vars: StaticVars,
    prior_scaling,
    l1_regularization: float,
    velocity_range: tuple,
):
    """Computes the prior loss for the given optimization variables.

    ``velocity_range`` is the expected ``(min, max)`` scatterer speed in wavelengths
    per frame; speeds inside this band are unpenalized while smoothness and amplitude
    priors are annealed in over each frame block.
    """
    assert isinstance(opt_vars, ParamsBase)

    n_frames = opt_vars.scat_pos.shape[1]
    ramp = _prior_annealing_ramp(iteration, n_iterations_per_frame)
    wavelength = static_vars.wavelength
    loss = 0.0
    # loss += _path_prior_loss(opt_vars, n_frames, ramp, wavelength, velocity_range)
    # loss += _amplitude_smoothness_loss(opt_vars, n_frames, ramp)
    loss += jnp.mean(jnp.abs(opt_vars.scat_amp.data)) * l1_regularization
    loss += _waveform_power_loss(opt_vars, static_vars) * PRIOR_WEIGHT_WAVEFORM_POWER
    # if n_frames >= 4:
    #     radius = wavelength * DIRECTION_COHERENCE_RADIUS_WL
    #     loss += (
    #         _direction_coherence_loss(opt_vars, wavelength, radius)
    #         * ramp
    #         * PRIOR_WEIGHT_DIRECTION_COHERENCE
    #     )

    # loss += (
    #     jnp.linalg.norm(jnp.diff(jnp.diff(opt_vars.scat_pos.data, axis=1), axis=1))
    #     * 3e-2
    # )
    if n_frames >= 3:
        loss += jnp.where(
            iteration > n_iterations_per_frame * 3,
            jnp.sum(
                jnp.square(
                    jnp.diff(
                        jnp.diff(opt_vars.scat_pos.data / wavelength, axis=1),
                        axis=1,
                    ),
                )
            )
            * 1e-3,
            0,
        )

        # loss += jnp.where(
        #     iteration > n_iterations_per_frame * 2,
        #     jnp.sum(
        #         jnp.square(
        #             jnp.linalg.norm(
        #                 jnp.diff(opt_vars.scat_pos.data / wavelength, axis=1),
        #                 axis=-1,
        #             ),
        #         )
        #     )
        #     * 1e-6,
        #     0,
        # )
        pass
    return loss


def _input_warnings(program_state):
    n_frames = program_state.opt_vars.scat_pos.shape[1]
    if (
        program_state.config.n_iterations // program_state.config.n_iterations_per_frame
        < (n_frames // 2 - 1)
    ):
        console.log(
            f"[red]Warning:[/red] The number of iterations "
            f"({program_state.config.n_iterations}) is too small to fully solve all "
            f"frames ({n_frames}). Consider increasing n_iterations to at least "
            f"{(n_frames // 2 - 1) * program_state.config.n_iterations_per_frame}."
        )


def mask_remove_weak_scatterers(opt_vars: ParamsType, threshold: float) -> ParamsType:
    """Removes scatterers with amplitude below the given threshold."""
    assert isinstance(opt_vars, ParamsBase)

    mean_amplitudes = _mean_reduce_to_scat_dim(opt_vars.scat_amp)

    mean_amplitudes = mean_amplitudes / jnp.max(jnp.abs(mean_amplitudes) + 1e-4)

    mask = mean_amplitudes >= threshold

    console.log(f"Removing weak scatterers: {jnp.sum(~mask)}/{mask.size} removed.")

    return mask


def _mean_reduce_to_scat_dim(scat_amp) -> jnp.ndarray:
    all_dims = set(range(scat_amp.data.ndim)) - set((scat_amp.scat_dim,))
    scat_amp = jnp.abs(scat_amp.data)
    if len(all_dims) == 0:
        mean_amplitudes = scat_amp
    else:
        mean_amplitudes = jnp.mean(scat_amp, axis=tuple(all_dims))
    return mean_amplitudes


def mask_merge_scatterers_that_are_always_close(
    opt_vars: ParamsType, threshold: float
) -> jnp.ndarray:
    """Merges scatterers that are closer than the given threshold in all frames."""
    assert isinstance(opt_vars, ParamsBase)

    # Compute pairwise distances between scatterers in each frame
    distances = jnp.linalg.norm(
        opt_vars.scat_pos.data[:, None, :, :] - opt_vars.scat_pos.data[None, :, :, :],
        axis=-1,
    )  # shape (n_scat, n_scat, n_frames)

    # shape (n_scat, n_scat)
    max_distances = jnp.max(distances, axis=-1)
    mean_intensities = _mean_reduce_to_scat_dim(opt_vars.scat_amp)
    mask_smaller = mean_intensities[:, None] < mean_intensities[None, :]
    mask_close = max_distances < threshold
    mask_keep = jnp.sum(jnp.logical_and(mask_smaller, mask_close), axis=-1) == 0
    jax.debug.print(
        "Merging scatterers that are always close: {}/{} scatterers removed.",
        jnp.sum(~mask_keep),
        opt_vars.scat_amp.shape[0],
    )
    return mask_keep


def apply_mask(mask, opt_vars: ParamsType, opt_state) -> tuple[ParamsType, Any]:
    """Removes scatterers based on the given mask."""
    assert isinstance(opt_vars, ParamsBase)

    console.log(f"Removing scatterers: {jnp.sum(~mask)}/{mask.size} removed.")

    new_opt_vars = opt_vars.index_scatterers(mask)

    new_opt_state = []
    for state in opt_state:
        if isinstance(state, optax.ScaleByAdamState):
            new_state = optax.ScaleByAdamState(
                count=state.count,
                mu=state.mu.index_scatterers(mask),
                nu=state.nu.index_scatterers(mask),
            )
        else:
            new_state = state
        new_opt_state.append(new_state)
    new_opt_state = tuple(new_opt_state)

    return new_opt_vars, new_opt_state


def compute_full_rf(
    opt_vars: ParamsType,
    static_vars: StaticVarsType,
    forward_model: Callable,
):
    """Computes the full RF for the given optimization variables."""

    assert isinstance(opt_vars, ParamsBase)
    assert isinstance(static_vars, StaticVars)

    batch_size = 256
    transmits, fbins = jnp.meshgrid(
        jnp.arange(opt_vars.scat_pos.shape[1]),
        jnp.arange(static_vars.n_fbins),
        indexing="ij",
    )
    transmits, fbins = transmits.ravel(), fbins.ravel()
    indices = jnp.arange(transmits.size)

    n_batches = transmits.size // batch_size

    sample_indices_batches = indices[: n_batches * batch_size].reshape(
        (n_batches, batch_size)
    )
    sample_indices_remainder = indices[n_batches * batch_size :]

    @jit
    def compute_interval(carry, indices_batch):
        """Computes the RF for the given interval."""

        y_rfft_hat = jax.vmap(forward_model, in_axes=(None, None, 0, 0))(
            opt_vars,
            static_vars,
            transmits[indices_batch],
            fbins[indices_batch],
        )
        return carry, y_rfft_hat

    _, y_rfft_hat = jax.lax.scan(
        compute_interval,
        None,
        sample_indices_batches,
    )

    remainder = indices.size % batch_size
    if remainder > 0:
        y_rfft_hat_remainder = jax.vmap(forward_model, in_axes=(None, None, 0, 0))(
            opt_vars,
            static_vars,
            transmits[sample_indices_remainder],
            fbins[sample_indices_remainder],
        )
        y_rfft_hat = jnp.concatenate(
            (y_rfft_hat.ravel(), y_rfft_hat_remainder.ravel()), axis=0
        )

    return y_rfft_hat.ravel()


def compute_and_save_rf(
    opt_vars: ParamsType,
    static_vars: StaticVars,
    path: str,
    rfft: RFFT,
    forward_model: Callable,
    parameters,
):
    console.log("Computing RF with Paris model...")
    assert isinstance(opt_vars, ParamsBase)
    assert isinstance(static_vars, StaticVars)
    assert isinstance(rfft, RFFT)

    n_tx = static_vars.n_tx
    n_fbins = static_vars.n_fbins
    n_el = static_vars.n_el

    y_rfft_flat = compute_full_rf(opt_vars, static_vars, forward_model)

    y_rfft = y_rfft_flat.reshape(1, n_tx, n_fbins, n_el)
    raw_data = rfft.irfft(y_rfft, axis=-2)
    raw_data = zea.func.ultrasound.upmix(
        raw_data,
        sampling_frequency=parameters.sampling_frequency,
        demodulation_frequency=parameters.demodulation_frequency,
        upsampling_rate=1,
    )[..., None]

    probe = ProbeSpec(
        name="S5-1",
        probe_geometry=parameters.probe_geometry,
        element_width=parameters.element_width,
    )

    scan = ScanSpec(
        sampling_frequency=parameters.sampling_frequency,
        center_frequency=parameters.center_frequency,
        initial_times=parameters.initial_times,
        t0_delays=parameters.t0_delays,
        sound_speed=parameters.sound_speed,
        focus_distances=parameters.focus_distances,
        polar_angles=parameters.polar_angles,
        azimuth_angles=parameters.azimuth_angles,
        tx_apodizations=parameters.tx_apodizations,
        waveforms_two_way=parameters.waveforms_two_way,
        waveforms_one_way=parameters.waveforms_one_way,
        transmit_origins=np.zeros((n_tx, 3)),
        demodulation_frequency=parameters.demodulation_frequency,
    )
    # with zea.File(out_dir / "simulated.hdf5", "w") as f:
    raw_data = np.array(raw_data, dtype=np.float32)
    zea.File.create(
        path,
        data={"raw_data": raw_data},
        scan=scan,
        probe=probe,
        overwrite=True,
    )

    return y_rfft
