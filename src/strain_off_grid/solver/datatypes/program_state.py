from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import optax
import zea

from strain_off_grid.solver import RFFT
from strain_off_grid.solver.datatypes.config import SolverConfig
from strain_off_grid.solver.datatypes.dimensions import Dimensions
from strain_off_grid.solver.datatypes.indices import Indices
from strain_off_grid.solver.datatypes.params import ParamsParis, ParamsRegular, Scaled
from strain_off_grid.solver.datatypes.static_vars import StaticVars


@dataclass
class ProgramState:
    """Holds the state of the program."""

    key: jnp.ndarray
    opt_vars: ParamsRegular[Scaled] | ParamsParis[Scaled]
    static_vars: StaticVars
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    indices_all: Indices
    y_rfft_flat: jnp.ndarray
    y_rfft_shape: tuple
    iteration: int
    dimensions: Dimensions
    config: SolverConfig
    rfft: RFFT
    beamformed_images: list
    forward_model: Callable
    parameters: zea.scan.Parameters
    total_scaling_factor: float
    solve_index: int = 0
    timestamps: jnp.ndarray | None = None
