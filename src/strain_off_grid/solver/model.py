# Prevent ruff from removing
# [
import jax  # noqa: F401
import jax.numpy as jnp
import numpy as np
from jax import jit

from strain_off_grid.solver.datatypes import (
    ParamsRegular,
    Physical,
    StaticVars,
)


@jit
def forward_model(
    opt_vars: ParamsRegular[Physical],
    static_vars: StaticVars,
    tx_idx: int,
    fbin_idx: int,
):
    #
    # scat_pos - (n_scat, n_frames, 2)
    # scat_amp - (n_scat, n_frames)
    # waveform_rfft_offset - (n_fbins,)
    # phases - (n_scat, n_frames)
    # probe_geometry - (n_el, 2)

    assert isinstance(opt_vars, ParamsRegular)
    assert isinstance(static_vars, StaticVars)

    n_scat, n_frames = opt_vars.scat_amp.shape
    n_fbins = opt_vars.waveform_rfft_offset.shape[0]
    n_el = static_vars.probe_geometry.shape[0]

    # assert opt_vars.scat_pos.shape == (n_scat, n_frames, 3), (
    #     f"scat_pos shape: {opt_vars.scat_pos.shape}, expected: {(n_scat, n_frames, 3)}"
    # )
    assert opt_vars.delta_pos.shape == (n_scat, n_frames), (
        f"delta_pos shape: {opt_vars.delta_pos.shape}, expected: {(n_scat, n_frames)}"
    )
    # assert static_vars.probe_geometry.shape == (n_el, 3), (
    #     f"probe_geometry shape: {static_vars.probe_geometry.shape}, expected: {(n_el, 3)}"
    # )
    assert static_vars.waveform_rfft.shape == (n_fbins,), (
        f"waveform_rfft shape: {static_vars.waveform_rfft.shape}, expected: {(n_fbins,)}"
    )

    frequency = static_vars.freqs[fbin_idx]
    center_frequency = static_vars.center_frequency
    attenuation_coef = static_vars.attenuation_coef

    tx_idx_from_zero = tx_idx
    tx_idx_global_context = tx_idx + (
        (static_vars.n_tx - opt_vars.scat_amp.shape[1]) // 2
    )

    # ==================================================================================
    # Unpack variables
    # ==================================================================================
    scat_pos = opt_vars.scat_pos.data[:, tx_idx_from_zero] * jnp.array([1.0, 1.0])[None]
    scat_amp = opt_vars.scat_amp.data
    waveform_rfft_offset = opt_vars.waveform_rfft_offset.data
    delta_pos = opt_vars.delta_pos.data
    sound_speed = static_vars.sound_speed

    relative_positions = scat_pos[:, None] - static_vars.probe_geometry[None]

    cone_distance = distance_to_focus_cone(
        scat_pos,
        static_vars.focus_distances[tx_idx_global_context],
        static_vars.polar_angles[tx_idx_global_context],
        aperture_size=jnp.abs(
            static_vars.probe_geometry[-1, 0] - static_vars.probe_geometry[0, 0]
        ),
    )
    wavelength = sound_speed / frequency
    cone_mask = jnp.exp(-((cone_distance / wavelength) ** 2))

    # (n_scat, n_frames, n_el) -> n_frames x n_el smaller than batch size, keep like this
    angles = jnp.arctan2(
        relative_positions[..., 0],
        relative_positions[..., -1],
    )  # (n_scat, n_frames, n_el)

    # ==============================================================================
    #
    # ==============================================================================

    distances = jnp.linalg.norm(
        relative_positions,
        axis=-1,
    )

    tau_tx = (
        distances / sound_speed
        + static_vars.t0_delays[tx_idx_global_context]
        - static_vars.initial_times[tx_idx_global_context]
        + static_vars.t_peak
    )

    tau_rx = distances / sound_speed

    tau_phase_compensation = jnp.exp(
        1j
        * 2
        * jnp.pi
        * 2
        * jnp.linalg.norm(scat_pos, axis=-1)
        * center_frequency
        / sound_speed
    )

    # ==============================================================================
    #
    # ==============================================================================
    forward_field = (
        opt_vars.waveform_rfft_offset.compute_waveform(static_vars.waveform_rfft)[
            fbin_idx
        ]
    )[None]
    forward_field = static_vars.waveform_rfft[fbin_idx][None]

    # (n_scat, n_frames, n_fbins, n_el)
    directivity_scaling = directivity(
        angles,
        frequency,
        static_vars.element_width * opt_vars.directivity_falloff.data[0],
        sound_speed,
        # opt_vars.reduce_factor[:, indices.frames],
    )
    angles_factor = jnp.cos(angles)
    frequency_factor = frequency / center_frequency
    directivity_scaling = directivity_scaling * (
        opt_vars.directivity_falloff.data[0] * angles_factor**2
        + opt_vars.directivity_falloff.data[1] * angles_factor
        + opt_vars.directivity_falloff.data[2]
        + opt_vars.directivity_falloff.data[3] * frequency_factor**2
        + opt_vars.directivity_falloff.data[4] * frequency_factor
    )

    # jax.debug.print("sound_speed_adapted: {}", sound_speed_adapted)

    # (n_scat, n_frames, n_fbins, n_el) -> n_frames x n_fbins x n_el larger than batch size -> directly convert to samples
    delay_tx = jnp.exp(-1j * 2 * np.pi * frequency * tau_tx)
    delay_rx = jnp.exp(-1j * 2 * np.pi * frequency * tau_rx)
    attenuation_tx = jnp.exp(
        -static_vars.attenuation_coef * frequency * 1e-6 * distances * 1e2
    )
    attenuation_rx = jnp.exp(
        -static_vars.attenuation_coef * frequency * 1e-6 * distances * 1e2
    )

    tgc_compensation = jnp.exp(
        static_vars.tgc_gain * (tau_tx + distances / sound_speed)
    )

    # (n_scat, n_frames, n_el)
    geometric_spreading = 1e-2 / (distances + 3e-3)
    # jax.debug.print("geometric_spreading: {}", geometric_spreading)
    # jax.debug.print(
    #     "forward_field: {}",
    #     delay_rx * directivity_scaling * geometric_spreading,
    #     # * scat_amp[:, frame_idx][:, None],
    #     # * tgc_compensation
    #     # * attenuation_rx
    #     # * tau_phase_compensation[:, None],
    # )
    harmonic_amplitude = second_harmonic_scaling(
        z=scat_pos[..., -1],
        f0=center_frequency,
        c=sound_speed,
        rho=1000,
        beta=3.5,
        alpha_0=attenuation_coef,
        n=1.0,
    )
    # harmonic_amplitude = 1.0 - jnp.exp(jnp.square((scat_pos[..., -1] - 40e-3) / 40e-3))
    # jax.debug.print("harmonic_amplitude: {}", harmonic_amplitude)
    # (n_scat, samples)
    responses = (
        jnp.mean(
            forward_field[:, None]
            * delay_tx
            * directivity_scaling
            * geometric_spreading
            * static_vars.tx_apodizations[tx_idx_global_context][None]
            * attenuation_tx,
            axis=-1,
            keepdims=True,
        )
        # * harmonic_amplitude[..., None]
        # * jnp.exp(jnp.linalg.norm(scat_pos, axis=-1) / 80e-3)[..., None]
        * delay_rx
        * directivity_scaling
        * geometric_spreading
        * scat_amp[:, tx_idx_from_zero][:, None]
        * tgc_compensation
        * attenuation_rx
        # * tau_phase_compensation[:, None]
        # * jnp.exp(
        #     -1j
        #     * 2
        #     * jnp.pi
        #     * 2
        #     * delta_pos[:, tx_idx_from_zero]
        #     * center_frequency
        #     / sound_speed
        # )[:, None]
        # * cone_mask[:, None]
        # * jnp.exp(1j * opt_vars.phases.data[:, frame_idx, tx_idx][:, None])
        # * jnp.exp(1j * opt_vars.phase[:, None])
    )
    # jax.debug.print("responses: {}", jnp.max(jnp.abs(responses)))
    return jnp.sum(responses, axis=0)


def directivity(theta, freqs, element_width, sound_speed):
    wavelegth = sound_speed / freqs
    argument = element_width / wavelegth * jnp.sin(theta)

    mask = jnp.logical_and(freqs > 0.0, jnp.abs(theta) > 1e-4)

    reduce_factor = 0.0

    return jnp.where(
        mask,
        jnp.sinc(argument) * jnp.cos(theta) * jnp.exp(-((theta * reduce_factor) ** 2)),
        1.0,
    )


def distance_to_focus_cone(scat_pos, focus_distance, polar_angle, aperture_size):
    """
    Computes the distance of each scatterer to the focus cone (hourglass shape).

    Points inside the cone have distance zero. Points outside have a positive
    distance equal to the perpendicular distance to the nearest cone edge.

    The cone is defined by two infinite lines passing through the left/right
    aperture elements and the focus point. These lines form an hourglass shape,
    and a point is inside if it lies between both lines.

    Parameters
    ----------
    scat_pos : jnp.ndarray
        The positions of the scatterers, shape (n_scat, 3).
    focus_distance : float
        The distance from the probe to the focus point.
    polar_angle : float
        The polar angle of the focus point in radians, measured from z-axis.
    aperture_size : float
        The size of the aperture of the probe.

    Returns
    -------
    jnp.ndarray
        Distance to the cone boundary, shape (n_scat,). Zero inside, positive outside.
    """
    focus_point = polar_to_xz(focus_distance, polar_angle)
    left_element = jnp.array([-aperture_size / 2, 0.0])
    right_element = jnp.array([aperture_size / 2, 0.0])

    scat_xz = scat_pos[:, jnp.array([0, 2])]

    # Signed cross products: opposite signs => inside the hourglass.
    # Magnitude divided by edge length gives perpendicular distance to that edge.
    left_cross = cross_product_sign(left_element, focus_point, scat_xz)
    right_cross = cross_product_sign(right_element, focus_point, scat_xz)

    left_edge_len = jnp.linalg.norm(focus_point - left_element)
    right_edge_len = jnp.linalg.norm(focus_point - right_element)

    dist_to_left_edge = jnp.abs(left_cross) / left_edge_len
    dist_to_right_edge = jnp.abs(right_cross) / right_edge_len

    is_inside = left_cross * right_cross < 0
    dist_to_nearest_edge = jnp.minimum(dist_to_left_edge, dist_to_right_edge)

    return jnp.where(is_inside, 0.0, dist_to_nearest_edge)


def polar_to_xz(distance, polar_angle):
    """Converts polar coordinates (angle from z-axis) to an (x, z) point."""
    return jnp.array([distance * jnp.sin(polar_angle), distance * jnp.cos(polar_angle)])


def cross_product_sign(line_start, line_end, points):
    """Computes the signed cross product placing points left/right of a line."""
    line = line_end - line_start
    relative_points = points - line_start[None]
    return line[0] * relative_points[:, 1] - line[1] * relative_points[:, 0]


def second_harmonic_scaling(z, f0, c, rho, beta, alpha_0, n=1.0):
    omega = 2 * jnp.pi * f0
    alpha_1 = alpha_0
    alpha_2 = alpha_0 * (2**n)

    d_alpha = alpha_2 - alpha_1
    normal = (jnp.exp(-alpha_1 * z) - jnp.exp(-alpha_2 * z)) / jnp.where(
        d_alpha == 0, 1.0, d_alpha
    )
    degenerate = z * jnp.exp(-alpha_1 * z)

    depth_term = jnp.where(d_alpha == 0, degenerate, normal)
    return (beta * omega) / (2 * rho * c**3) * depth_term * 1e6
