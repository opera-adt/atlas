from __future__ import annotations

import logging

import numpy as np
from numba import njit, stencil
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage
from scipy.interpolate import NearestNDInterpolator

TWOPI = 2 * np.pi

logger = logging.getLogger("dolphin")


@njit(nogil=True)
def compute_phase_diffs(phase):
    """Compute the total number phase jumps > pi between adjacent pixels.

    If part of `phase` is known to be bad phase (e.g. over water),
    the values should be set to zero or a masked array should be passed:

        unwrapping_error(np.ma.masked_where(bad_area_mask, phase))


    Parameters
    ----------
    phase : ArrayLike
        Unwrapped interferogram phase.

    Returns
    -------
    int
        Total number of jumps exceeding pi.

    """
    return _compute_phase_diffs(phase)


@stencil
def _compute_phase_diffs(phase):
    d1 = np.abs(phase[0, 0] - phase[0, 1]) / np.pi
    d2 = np.abs(phase[0, 0] - phase[1, 0]) / np.pi
    # Subtract 0.5 so that anything below 1 gets rounded to 0
    return round(d1 - 0.5) + round(d2 - 0.5)


def rewrap_to_twopi(arr: ArrayLike) -> np.ndarray:
    """Rewrap `arr` to be between [-pi and pi].

    Parameters
    ----------
    arr : ArrayLike
        Unwrapped floating point phase

    Returns
    -------
    np.ndarray
        Array of phases between -pi and pi

    """
    return np.mod(np.pi + arr, TWOPI) - np.pi


def get_2pi_ambiguities(
    unw: NDArray[np.floating], round_decimals: int = 4
) -> NDArray[np.floating]:
    """Find the number of 2pi offsets from [-pi, pi) at each pixel of `unw`."""
    mod_2pi_image = np.mod(np.pi + unw, TWOPI) - np.pi
    re_wrapped = np.round(mod_2pi_image, round_decimals)
    return np.round((unw - re_wrapped) / (TWOPI), round_decimals - 1)


def interpolate_masked_gaps(
    unw: NDArray[np.float64], ifg: NDArray[np.complex64]
) -> None:
    """Perform phase unwrapping using nearest neighbor interpolation of ambiguities.

    Overwrites `unw`'s masked pixels with the interpolated values.

    This function takes an input unwrapped phase array containing NaNs at masked pixel.
    It calculates the phase ambiguity, K, at the attempted unwrapped pixels, then
    interpolates the ambiguities to fill the gaps.
    The masked pixels get the value of the original wrapped phase + 2pi*K.

    Parameters
    ----------
    unw : NDArray[np.float]
        Input unwrapped phase array with NaN values for masked areas.
    ifg : NDArray[np.complex64]
        Corresponding wrapped interferogram phase

    Returns
    -------
    np.ndarray
        Fully unwrapped phase array with interpolated values for previously
        masked areas.

    """
    # Create masks for valid areas
    ifg_valid = ~np.isnan(ifg) & (ifg != 0)
    unw_valid = ~np.isnan(unw)

    # Identify areas to interpolate: where ifg is valid but unw is not
    interpolate_mask = ifg_valid & ~unw_valid

    # If there's nothing to interpolate, we're done
    if not np.any(interpolate_mask):
        return

    # Calculate ambiguities for valid unwrapped pixels
    valid_pixels = ifg_valid & unw_valid
    # Work with a subsampled version for the nearest interpolator to speed it up
    sub = 4
    ambiguities = get_2pi_ambiguities(unw[::sub, ::sub][valid_pixels[::sub, ::sub]])

    # Get coordinates for valid pixels and pixels to interpolate
    valid_coords = np.array(np.where(valid_pixels[::sub, ::sub])).T
    interp_coords = np.array(np.where(interpolate_mask[::sub, ::sub])).T

    # Create and apply the interpolator
    interpolator = NearestNDInterpolator(valid_coords, ambiguities)
    interpolated_ambiguities = interpolator(interp_coords)

    # Apply interpolated ambiguities to the wrapped phase
    # Fill in the subsampled version, then resize it to full
    interpolated_sub = np.zeros(ifg[::sub, ::sub].shape, dtype=int)
    interpolated_sub[interpolate_mask[::sub, ::sub]] = interpolated_ambiguities

    unw[interpolate_mask] = np.angle(ifg[interpolate_mask]) + (
        TWOPI
        * np.round(_resize(interpolated_sub, ifg.shape)).astype(int)[interpolate_mask]
    )


def _resize(image: ArrayLike, output_shape: tuple[int, int]):
    """Resize `image` to be the same shape as `output_shape`."""
    input_shape = image.shape[-2:]
    factors = np.divide(input_shape, output_shape)

    # Translate modes used by np.pad to those used by scipy.ndimage
    zoom_factors = [1 / f for f in factors]
    return ndimage.zoom(image, zoom_factors, grid_mode=True, mode="grid-constant")
