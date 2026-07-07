from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr

from strain_off_grid.utils import parse_indices_from_string


def load_config(
    config_path: Path, config_class, overwrite_dict: dict[str, Any] | None = None
) -> BaseModel:
    """Loads the config from a TOML file."""
    with open(config_path, "rb") as f:
        config_dict = tomllib.load(f)
    if overwrite_dict is not None:
        config_dict.update(overwrite_dict)
    return config_class(**config_dict)


class SolverConfig(BaseModel):
    input_file: Path
    sweep_file_glob: str = Field(default="")
    sweep_step_size: int = Field(default=4)
    output_dir: Path = Field(default=Path("out/"))
    learning_rate: float = Field(default=1e-3)
    batch_size: int = Field(default=32)
    n_iterations_per_frame: int = Field(default=500)
    seed: int = Field(default=42)
    target_region: list[float] = Field(default=[-7e-3, 7e-3, 1.5e-3, 12e-3])
    n_frames: int = Field(default=9)
    first_frame: int = Field(default=0)
    transmits: list[str | int] | str | int = Field(default="all")
    attenuation_coef: float = Field(default=0.5)
    das_dynamic_range: int = Field(default=70)
    tgc_per_256_samples: float = Field(default=6.0)
    tgc_per_256_samples_unmodelled: float = Field(default=1.0)
    peak_detection_threshold: float = Field(default=0.2)
    f_number: float = Field(default=1.0)
    min_distance_between_peaks_wl: float = Field(default=1.0)
    max_n_peaks: int = Field(default=1024)
    remove_weak_scatters_threshold: float = Field(default=1e-2)
    reference_solution_path: str | None = Field(default=None)
    freeze_waveform: bool = Field(default=False)
    l1_regularization: float = Field(default=0.0)
    expected_velocity_range_wl_per_frame: tuple[float, float] = Field(
        default=(0.0, 2.0)
    )
    n_fbins: int = Field(default=256)
    use_analytical_waveform: bool = Field(default=False)
    gaussian_correlation_threshold: float = Field(default=0.0)
    gradient_accumulation_steps: int = Field(default=1)
    initialize_randomly: bool = Field(default=False)
    subsolve_n_solves: int = Field(default=6)
    subsolve_top_k: int = Field(default=256)
    paris_model: bool = Field(default=False)
    t_peak: float | None = Field(default=None)
    harmonic: bool = False
    is_2d: bool = True

    _resolved_transmits: np.ndarray | slice | int = PrivateAttr()
    _resolved_sweep_paths: list[Path]

    def model_post_init(self, context: Any, /) -> None:
        self._resolved_transmits = parse_indices_from_string(self.transmits)
        self._n_iterations = (
            self.iteration_end_phase_start + self.n_iterations_per_frame * 2
        )
        print(f"n_iterations: {self.n_iterations}")

        if self.sweep_file_glob:
            self._resolved_sweep_paths = sorted(
                list(Path(".").glob(self.sweep_file_glob))
            )
        else:
            self._resolved_sweep_paths = []

    @property
    def resolved_transmits(self) -> np.ndarray | slice | int:
        return self._resolved_transmits

    @property
    def n_iterations(self) -> int:
        return self._n_iterations

    @property
    def resolved_sweep_paths(self):
        return self._resolved_sweep_paths

    @property
    def iteration_end_phase_start(self):
        n_tx = len(self.resolved_transmits)
        return self.n_iterations_per_frame * (n_tx // 2 + 1)

    def update(self, **nested_dict) -> SolverConfig:
        return self.model_copy(update=nested_dict)


def load_config_solver(
    config_path: Path, overwrite_dict: dict[str, Any] | None = None
) -> SolverConfig:
    """Loads the config from a TOML file."""
    with open(config_path, "rb") as f:
        config_dict = tomllib.load(f)
    if overwrite_dict is not None:
        config_dict.update(overwrite_dict)
    return SolverConfig(**config_dict)


def _resolve_input_file_path(path: str | Path) -> list[Path]:
    path = Path(path)
    # Check if the path is absolute
    files = _get_glob_files(str(path))

    if len(files) > 0:
        return files

    # Otherwise, try to resolve using PALA_DATA_ROOT
    pala_data_root = os.getenv("PALA_DATA_ROOT")
    assert pala_data_root is not None, (
        "Environment variable PALA_DATA_ROOT is not set. Cannot resolve input file path."
    )
    assert not path.is_absolute(), "No input files found and path is absolute."
    abs_path = Path(pala_data_root) / path
    files = _get_glob_files(str(abs_path))
    assert len(files) > 0, f"No input files found for path: {path}"
    return files


def _get_glob_files(path_glob: str) -> list[Path]:
    if Path(path_glob).is_absolute():
        return [Path(path_glob)]
    files = list(Path().glob(path_glob))
    return sorted(files)
