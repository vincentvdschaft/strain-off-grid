class Spacing(tuple):
    def __new__(cls, horizontal, vertical):
        return super().__new__(cls, (float(horizontal), float(vertical)))

    @property
    def horizontal(self):
        return self[0]

    @property
    def vertical(self):
        return self[1]

    def __repr__(self):
        return f"Spacing(horizontal={self.horizontal}, vertical={self.vertical})"

    def to_dict(self) -> dict:
        """Return the spacing as a dictionary."""
        return {
            "horizontal": self.horizontal,
            "vertical": self.vertical,
        }
