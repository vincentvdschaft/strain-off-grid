import numpy as np


def match_histograms(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image.

    Parameters
    ----------
    source : np.ndarray
        Image to transform; the histogram is computed over the flattened array.
    template : np.ndarray
        Template image; can have different dimensions to source.
    Returns
    -------
    matched : np.ndarray
        The transformed output image.
    """
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # Get the set of unique pixel values and their corresponding indices and counts
    _, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # Calculate the empirical cumulative distribution functions (CDF) for the source and template images
    s_quantiles = np.cumsum(s_counts).astype(np.float64) / source.size
    t_quantiles = np.cumsum(t_counts).astype(np.float64) / template.size

    # Interpolate to find the pixel values in the template image that correspond to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)
