def extent_to_aspect_if_needed(
    extent: float | tuple[float, float, float, float] | None,
) -> float | None:
    """Calculate the aspect ratio from the extent of an axis if needed."""
    if isinstance(extent, (float, int)):
        return float(extent)

    if extent is None:
        return None

    return extent_to_aspect(extent)


def extent_to_aspect(extent: tuple[float, float, float, float]) -> float | None:
    """Calculate the aspect ratio from the extent of an axis."""
    x0, x1, y0, y1 = extent
    width = x1 - x0
    height = y1 - y0
    return height / width
