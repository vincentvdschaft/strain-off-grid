from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

TConfig = TypeVar("TConfig", bound=BaseModel)


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base, returning a new dict."""
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_path: Path | str,
    config_class: type[TConfig],
    overwrite_dict: dict[str, Any] | None = None,
) -> TConfig:
    """Loads the config from a TOML file."""
    config_path = Path(config_path)
    with open(config_path, "rb") as f:
        config_dict = tomllib.load(f)
    if overwrite_dict is not None:
        config_dict = _deep_merge(config_dict, overwrite_dict)
    return config_class(**config_dict)


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
