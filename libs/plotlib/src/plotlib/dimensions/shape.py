from typing import Tuple


class IntShape(tuple):
    def __new__(cls, width: int, height: int):
        return super().__new__(cls, (int(width), int(height)))

    @property
    def n_rows(self):
        return self[1]

    @property
    def n_cols(self):
        return self[0]

    @property
    def width(self):
        return self[0]

    @property
    def height(self):
        return self[1]


class FloatShape(tuple):
    def __new__(cls, width: float, height: float):
        return super().__new__(cls, (float(width), float(height)))

    @property
    def width(self):
        return self[0]

    @property
    def height(self):
        return self[1]
