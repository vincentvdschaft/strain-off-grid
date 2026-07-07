# from dataclasses import dataclass


# @dataclass
# class Margins:
#     """
#     A class to represent margins for a plot in inches.

#     Attributes:
#         left (float): Left margin in inches.
#         right (float): Right margin in inches.
#         top (float): Top margin in inches.
#         bottom (float): Bottom margin in inches.
#     """

#     left: float = 0.5
#     right: float = 0.5
#     top: float = 0.5
#     bottom: float = 0.5

#     def __post_init__(self):
#         if not all(
#             isinstance(margin, (int, float))
#             for margin in (self.left, self.right, self.top, self.bottom)
#         ):
#             raise TypeError("All margins must be numeric values (int or float).")
#         if any(margin < 0 for margin in (self.left, self.right, self.top, self.bottom)):
#             raise ValueError("Margins cannot be negative.")

#     @property
#     def width(self) -> float:
#         """Calculate the total width of the margins."""
#         return self.left + self.right

#     @property
#     def height(self) -> float:
#         """Calculate the total height of the margins."""
#         return self.top + self.bottom

#     def __repr__(self):
#         return f"Margins(left={self.left}, right={self.right}, top={self.top}, bottom={self.bottom})"

#     def to_tuple(self) -> tuple:
#         """Return the margins as a tuple."""
#         return (self.left, self.right, self.top, self.bottom)

#     def to_dict(self) -> dict:
#         """Return the margins as a dictionary."""
#         return {
#             "left": self.left,
#             "right": self.right,
#             "top": self.top,
#             "bottom": self.bottom,
#         }
