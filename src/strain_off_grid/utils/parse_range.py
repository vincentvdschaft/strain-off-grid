import numpy as np


def parse_indices_from_string(input_str: str) -> np.ndarray | slice:
    input_str = input_str.strip().replace(",", " ").replace(";", " ")
    parts = input_str.split()
    indices = [_parse_single_element(part) for part in parts]
    return np.concatenate(indices)


def _parse_single_element(indices: str) -> np.ndarray | slice:
    """Parse indices from a string.

    The format can be:
    - 0-4 -> [0, 1, 2, 3, 4]
    - 0-4-2 -> [0, 2, 4]
    - 3 -> [3]

    Parameters
    ----------
    indices : str
        A string representing indices.
    """
    if indices == "all":
        return slice(None)
    is_range = "-" in indices

    if is_range:
        parts = indices.split("-")
        if len(parts) == 2:
            start = int(parts[0]) if parts[0] != "" else None
            end = int(parts[1]) if parts[1] != "" else None
            return np.arange(start, end + 1)
        elif len(parts) == 3:
            start = int(parts[0]) if parts[0] != "" else None
            end = int(parts[1]) if parts[1] != "" else None
            step = int(parts[2]) if parts[2] != "" else 1
            return np.arange(start, end + 1, step)
        else:
            raise ValueError(f"Invalid range format: {indices}, parts: {parts}")

    # Single index
    return np.array([int(indices)])


# def _contrain_
