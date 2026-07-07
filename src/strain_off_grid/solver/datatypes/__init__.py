from .dimensions import Dimensions
from .indices import Indices
from .params import ParamsBase, ParamsParis, ParamsRegular, Physical, Scaled
from .program_state import ProgramState
from .static_vars import StaticVars

ParamsType = ParamsRegular
StaticVarsType = StaticVars
ParamsScaled = ParamsRegular[Scaled]
ParamsPhysical = ParamsRegular[Physical]

__all__ = [
    "Indices",
    "Dimensions",
    "StaticVars",
    "ParamsBase",
    "Scaled",
    "Physical",
    "ParamsParis",
    "ParamsRegular",
    "StaticVarsParis",
    "StaticVars",
    "ProgramState",
    "ParamsType",
    "StaticVarsType",
    "ParamsScaled",
    "ParamsPhysical",
]
