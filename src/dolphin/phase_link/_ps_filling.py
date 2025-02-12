from __future__ import annotations

import logging
import warnings

import numpy as np
from numba import njit
from numpy.typing import ArrayLike

from dolphin._types import Strides
from dolphin.utils import take_looks

logger = logging.getLogger("dolphin")


def fill_ps_pixels(
    cpx_phase: ArrayLike,
    temp_coh: ArrayLike,
    slc_stack: ArrayLike,
    ps_mask: ArrayLike,
    strides: Strides,
    avg_mag: ArrayLike | None,
    reference_idx: int = 0,
    use_max_ps: bool = True,
):
    """Fill in the PS locations in the MLE estimate with the original SLC data.

    Overwrites `cpx_phase` and `temp_coh` in place.

    Parameters
    ----------
    cpx_phase : ndarray, shape = (nslc, rows, cols)
        The complex valued-MLE estimate of the phase.
    temp_coh : ndarray, shape = (rows, cols)
        The temporal coherence of the estimate.
    slc_stack : ArrayLike
        The original SLC stack, with shape (n_images, n_rows, n_cols)
    ps_mask : ndarray, shape = (rows, cols)
        Boolean mask of pixels marking persistent scatterers (PS).
    strides : Strides
        The decimation (y, x) factors
    avg_mag : ArrayLike, optional
        The average magnitude of the SLC stack, used to to find the brightest
        PS pixels to fill within each look window.
    reference_idx : int, default = 0
        SLC to use as reference for PS pixels. All pixel values are multiplied
        by the conjugate of this index
    use_max_ps : bool, optional, default = True
        If True, use the brightest PS pixel in each look window to fill in the
        MLE estimate. If False, use the average of all PS pixels in each look window.

    """
    if avg_mag is None:
        # Get the average magnitude of the SLC stack
        # nanmean will ignore single NaNs, but not all NaNs, per pixel
        with warnings.catch_warnings():
            # ignore the warning about nansum/nanmean of empty slice
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_mag = np.nanmean(np.abs(slc_stack), axis=0)
    mag = avg_mag.copy()

    # null out all the non-PS pixels when finding the brightest PS pixels
    mag[~ps_mask] = np.nan
    # For ps_mask, we set to True if any pixels within the window were PS
    ps_mask_looked = take_looks(ps_mask, *strides, func_type="any", edge_strategy="pad")
    # make sure it's the same size as the MLE result/temp_coh after padding
    ps_mask_looked = ps_mask_looked[: cpx_phase.shape[1], : cpx_phase.shape[2]]

    if use_max_ps:
        logger.debug("Using max PS pixel to fill in MLE estimate")
        # Get the indices of the brightest pixels within each look window
        slc_r_idxs, slc_c_idxs = get_max_idxs(mag, *strides)

        # we're only filling where there are PS pixels
        ref = np.exp(-1j * np.angle(slc_stack[reference_idx][slc_r_idxs, slc_c_idxs]))
        for i in range(len(slc_stack)):
            slc_phase = np.angle(slc_stack[i][slc_r_idxs, slc_c_idxs])
            cur_amp = np.abs(cpx_phase[i][ps_mask_looked])
            new_value = np.exp(1j * slc_phase) * ref
            cpx_phase[i][ps_mask_looked] = cur_amp * new_value

    else:
        # Get the average of all PS pixels within each look window
        # The referencing to SLC 0 is done in _get_avg_ps
        avg_ps = _get_avg_ps(slc_stack, ps_mask, strides)[
            :, : cpx_phase.shape[1], : cpx_phase.shape[2]
        ]
        ps_phases = np.angle(avg_ps[:, ps_mask_looked])

        # Set the angle only, don't change magnitude
        cpx_phase[:, ps_mask_looked] = np.abs(cpx_phase[:, ps_mask_looked]) * np.exp(
            1j * ps_phases
        )

    # Force PS pixels to have high temporal coherence
    temp_coh[ps_mask_looked] = 1


def _get_avg_ps(
    slc_stack: np.ndarray, ps_mask: np.ndarray, strides: Strides
) -> np.ndarray:
    # First, set all non-PS pixels to NaN
    slc_stack_nanned = slc_stack.copy()
    slc_stack_nanned[:, ~ps_mask] = np.nan
    # Reference all ps pixels in the SLC stack to the first SLC
    slc_stack_nanned[:, ps_mask] *= np.exp(
        -1j * np.angle(slc_stack_nanned[0, ps_mask])
    )[None]
    # Then, take the average of all PS pixels within each look window
    return take_looks(
        slc_stack_nanned,
        *strides,
        func_type="nanmean",
        edge_strategy="pad",
    )


@njit
def get_max_idxs(arr: ArrayLike, row_looks: int, col_looks: int):
    window_height, window_width = row_looks, col_looks
    height, width = arr.shape

    # Calculate the number of windows
    new_height = height // window_height
    new_width = width // window_width

    row_indices = []
    col_indices = []

    for i in range(new_height):
        for j in range(new_width):
            window = arr[
                i * window_height : (i + 1) * window_height,
                j * window_width : (j + 1) * window_width,
            ]

            max_value = -np.inf
            max_row, max_col = -1, -1

            for row in range(window_height):
                for col in range(window_width):
                    value = window[row, col]
                    if not np.isnan(value) and value > max_value:
                        max_value = value
                        max_row = row
                        max_col = col

            # If max_row and max_col are still -1, it means the window was all NaNs
            if max_row != -1 and max_col != -1:
                row_indices.append(i * window_height + max_row)
                col_indices.append(j * window_width + max_col)

    return np.array(row_indices), np.array(col_indices)
