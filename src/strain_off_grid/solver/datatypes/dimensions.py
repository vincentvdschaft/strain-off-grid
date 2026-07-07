from dataclasses import dataclass


@dataclass
class Dimensions:
    """Holds the dimensions of the problem."""

    n_frames: int
    n_scat: int
    n_tx: int
    n_el: int
    n_fbins: int
