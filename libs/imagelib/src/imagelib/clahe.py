import numpy as np


def clahe(arr, clip_limit: float = 0.01, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    CLAHE on a 2D float array.
    clip_limit: fraction of total pixels in a tile (e.g. 0.01 = 1%)
    """
    arr = arr.astype(np.float32)
    h, w = arr.shape
    ty, tx = tile_grid_size  # number of tiles in each direction
    n_bins = 256

    # Quantize to [0, n_bins-1] for histogram computation
    arr_min, arr_max = arr.min(), arr.max()
    norm = (arr - arr_min) / (arr_max - arr_min + 1e-8)
    quantized = (norm * (n_bins - 1)).astype(np.int32)

    # Build tile maps: for each tile, compute a clipped & equalized LUT
    luts = np.zeros((ty, tx, n_bins), dtype=np.float32)
    tile_h = h / ty
    tile_w = w / tx

    for j in range(ty):
        for i in range(tx):
            y0, y1 = int(j * tile_h), int((j + 1) * tile_h)
            x0, x1 = int(i * tile_w), int((i + 1) * tile_w)
            tile = quantized[y0:y1, x0:x1]

            hist, _ = np.histogram(tile, bins=n_bins, range=(0, n_bins - 1))

            # Clip and redistribute
            clip_val = max(1, int(clip_limit * tile.size))
            excess = np.sum(np.maximum(hist - clip_val, 0))
            hist = np.minimum(hist, clip_val)
            hist += excess // n_bins  # redistribute excess uniformly

            # CDF -> LUT
            cdf = np.cumsum(hist)
            cdf_min = cdf[cdf > 0][0]
            lut = (cdf - cdf_min) / (tile.size - cdf_min + 1e-8)
            luts[j, i] = lut.astype(np.float32)

    # Bilinear interpolation between tile LUTs
    # Compute tile-center coordinates
    cy = (np.arange(ty) + 0.5) * tile_h  # shape (ty,)
    cx = (np.arange(tx) + 0.5) * tile_w  # shape (tx,)

    # For each pixel, find surrounding tile indices and weights
    py, px = np.mgrid[0:h, 0:w].astype(np.float32)  # pixel coords

    # Clamp to tile center range for border pixels
    iy = np.clip((py - cy[0]) / tile_h, 0, ty - 2).astype(np.int32)
    ix = np.clip((px - cx[0]) / tile_w, 0, tx - 2).astype(np.int32)

    wy = np.clip((py - cy[iy]) / tile_h, 0, 1)
    wx = np.clip((px - cx[ix]) / tile_w, 0, 1)

    q = quantized  # (h, w)

    # Gather the four surrounding LUT values
    v00 = luts[iy, ix][np.arange(h)[:, None], np.arange(w), q]
    v10 = luts[iy + 1, ix][np.arange(h)[:, None], np.arange(w), q]
    v01 = luts[iy, ix + 1][np.arange(h)[:, None], np.arange(w), q]
    v11 = luts[iy + 1, ix + 1][np.arange(h)[:, None], np.arange(w), q]

    result = (
        v00 * (1 - wy) * (1 - wx)
        + v01 * (1 - wy) * wx
        + v10 * wy * (1 - wx)
        + v11 * wy * wx
    )

    return result
