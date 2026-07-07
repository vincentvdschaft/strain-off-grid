from pathlib import Path
from typing import Any, Mapping

import h5py
import jax.numpy as jnp
import numpy as np
from imagelib import Image

from strain_off_grid import console
from strain_off_grid.solver.datatypes import ParamsPhysical
from strain_off_grid.solver.datatypes.params.base import ParamsBase
from strain_off_grid.solver.datatypes.program_state import ProgramState


def save_solution_from_structs(
    path,
    opt_vars: ParamsPhysical,
    program_state: ProgramState | None = None,
    extensive=False,
):
    """Saves the optimized solution to an HDF5 file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    image = _get_base_image(program_state, opt_vars)
    if not extensive:
        axes = list(range(1, image.ndim))
        image = image.resample_scale(0.25, axes=axes)
    image = _add_metadata(image, program_state, opt_vars)
    image.save(path)
    save_params_to_hdf5(path, opt_vars)
    if program_state is not None:
        # save_normal_beamformed(path, program_state.beamformed_images_normal)
        save_total_scaling_factor(path, program_state.total_scaling_factor)
        copy_custom_group(program_state.config.input_file, path)

    console.log(f"Saved solution to {path}")


def copy_custom_group(source_path, destination_path):
    """Copies the "custom" group from the source HDF5 to the solution file."""
    with h5py.File(source_path, "r") as source:
        if "custom" not in source:
            return
        with h5py.File(destination_path, "r+") as destination:
            _overwrite_group(source, destination, "custom")


def _overwrite_group(source, destination, name):
    """Copies a group from source to destination, replacing any existing one."""
    if name in destination:
        del destination[name]
    source.copy(name, destination)


def save_normal_beamformed(path, beamformed_images_normal):
    """Saves the normal beamformed images to an HDF5 file."""
    with h5py.File(path, "r+") as f:
        f.create_dataset(
            "beamformed_images_normal",
            data=beamformed_images_normal,
        )


def save_total_scaling_factor(path, total_scaling_factor):
    """Saves the total scaling factor to an HDF5 file."""
    with h5py.File(path, "r+") as f:
        f.create_dataset(
            "total_scaling_factor",
            data=total_scaling_factor,
        )


def _get_base_image(program_state, opt_vars) -> Image:
    n_transmits_current = opt_vars.scat_pos.shape[1]
    n_transmits_total = program_state.static_vars.n_tx
    offset = (n_transmits_total - n_transmits_current) // 2
    return program_state.beamformed_images[offset : offset + n_transmits_current]


def _add_metadata(image, program_state, opt_vars):
    print(opt_vars.scat_pos.shape)

    positions, intensities = _positions_and_intensities_to_list(
        opt_vars.scat_pos.data, opt_vars.scat_amp.data
    )
    image = (
        image.add_metadata("positions", positions)
        .add_metadata("intensities", intensities)
        .add_metadata("waveform_rfft_original", program_state.static_vars.waveform_rfft)
        .add_metadata(
            "waveform_rfft_total",
            opt_vars.waveform_rfft_offset.compute_waveform(
                program_state.static_vars.waveform_rfft
            ),
        )
        .add_metadata("probe_geometry", program_state.static_vars.probe_geometry)
    )
    try:
        tx_apodizations = _construct_tx_apodizations_from_active_el(
            program_state.static_vars.active_element,
            program_state.static_vars.probe_geometry.shape[0],
        )
        image = image.add_metadata("tx_apodizations", tx_apodizations)
    except Exception:
        pass
    print(program_state.timestamps)
    image = image.add_metadata("timestamps", program_state.timestamps)

    return image


def _positions_and_intensities_to_list(positions, intensities):
    n_frames = positions.shape[1]
    if intensities.ndim == 1:
        intensities = intensities[:, None] * jnp.ones((1, n_frames))

    return [positions[:, i] for i in range(positions.shape[1])], [
        intensities[:, i] for i in range(intensities.shape[1])
    ]


def _construct_tx_apodizations_from_active_el(active_elements, n_elements):
    tx_apodizations = []
    for active in active_elements:
        apodization = np.zeros(n_elements)
        apodization[active] = 1.0
        tx_apodizations.append(apodization)
    return np.array(tx_apodizations)


def save_params_to_hdf5(
    file_path: str | Path,
    params: ParamsBase,
    additional_items: Mapping[str, Any] | None = None,
    internal_path: str = "/",
) -> None:
    """Stores a ParamsBase subclass instance and optional extra items in an HDF5 file."""
    with h5py.File(file_path, "a") as h5_file:
        params_group = _create_internal_path(h5_file, internal_path)
        params.save_to_hdf5(params_group)

        if additional_items:
            _write_hdf5_mapping(h5_file, additional_items)


def _create_internal_path(group: Any, path: str) -> Any:
    """Creates nested groups in the HDF5 file for a given internal path."""
    if not path.startswith("/"):
        path = "/" + path
    if path == "/":
        return group

    parts = path.split("/")
    current_group = group
    for part in parts:
        if part not in current_group:
            current_group.create_group(part)
        current_group = current_group[part]
    return current_group


def _write_hdf5_mapping(group: Any, mapping: Mapping[str, Any]) -> None:
    for key, value in mapping.items():
        _write_hdf5_item(group, key, value)


def _write_hdf5_item(group: Any, key: str, value: Any) -> None:
    if isinstance(value, Mapping):
        subgroup = group.create_group(key)
        _write_hdf5_mapping(subgroup, value)
        return

    if value is None:
        group.attrs[f"{key}__is_none"] = True
        return

    group.create_dataset(key, data=value)
