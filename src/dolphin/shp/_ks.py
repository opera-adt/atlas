from __future__ import annotations

import logging
from math import exp, sqrt
from typing import Optional

import numba
import numpy as np
from numba import cuda
from numpy.typing import ArrayLike

from dolphin._types import Strides
from dolphin.utils import _get_slices, compute_out_shape

from ._common import remove_unconnected

logger = logging.getLogger("dolphin")


_get_slices = numba.njit(_get_slices)


def estimate_neighbors(
    amp_stack: ArrayLike,
    halfwin_rowcol: tuple[int, int],
    alpha: float,
    strides: Optional[dict[str, int]] = None,
    is_sorted: bool = False,
    prune_disconnected: bool = False,
):
    """Estimate the  at all pixels of `slc_stack` on the GPU."""
    # estimate_neighbors_gpu(
    #     sorted_amp_stack,
    #     halfwin_rowcol,
    #     strides_rowcol,
    #     alpha,
    #     neighbor_arrays,
    # )

    if strides is None:
        strides = {"x": 1, "y": 1}
    sorted_amp_stack = amp_stack if is_sorted else np.sort(amp_stack, axis=0)

    num_slc, rows, cols = sorted_amp_stack.shape
    ecdf_dist_cutoff = _get_ecdf_critical_distance(num_slc, alpha)
    logger.debug(f"ecdf_dist_cutoff: {ecdf_dist_cutoff}")

    strides_rowcol = strides["y"], strides["x"]
    out_rows, out_cols = compute_out_shape((rows, cols), Strides(*strides_rowcol))
    half_row, half_col = halfwin_rowcol
    is_shp = np.zeros(
        (out_rows, out_cols, 2 * half_row + 1, 2 * half_col + 1), dtype=np.bool_
    )

    _loop_over_neighbors(
        sorted_amp_stack,
        halfwin_rowcol,
        strides_rowcol,
        ecdf_dist_cutoff,
        prune_disconnected,
        is_shp,
    )

    return is_shp


@numba.njit(parallel=True, nogil=True)
def _loop_over_neighbors(
    sorted_amp_stack,
    halfwin_rowcol: tuple[int, int],
    strides_rowcol: tuple[int, int],
    ecdf_dist_cutoff: float,
    prune_disconnected: bool,
    is_shp: ArrayLike,
):
    """Estimate the SHPs of each pixel of `amp_stack` on the CPU."""
    num_slc, in_rows, in_cols = sorted_amp_stack.shape
    out_rows, out_cols = is_shp.shape[:2]

    half_row, half_col = halfwin_rowcol
    row_strides, col_strides = strides_rowcol
    r_start = row_strides // 2
    c_start = col_strides // 2
    # location to start counting from in the larger input
    r0, c0 = row_strides // 2, col_strides // 2

    for out_r in numba.prange(out_rows):
        for out_c in range(out_cols):
            in_r = r0 + out_r * row_strides
            in_c = c0 + out_c * col_strides
            # Clamp the window to the image bounds
            (r_start, r_end), (c_start, c_end) = _get_slices(
                half_row, half_col, in_r, in_c, in_rows, in_cols
            )

            amp_block = sorted_amp_stack[:, r_start:r_end, c_start:c_end]
            neighbors = is_shp[out_r, out_c, :, :]
            _set_neighbors(amp_block, halfwin_rowcol, ecdf_dist_cutoff, neighbors)
            if prune_disconnected:
                remove_unconnected(is_shp[out_r, out_c], inplace=True)

    return is_shp


# GPU version of the SHP finding algorithm using KS test
@cuda.jit
def estimate_neighbors_gpu(
    sorted_amp_stack,
    halfwin_rowcol: tuple[int, int],
    strides_rowcol: tuple[int, int],
    alpha: float,
    is_shp: ArrayLike,
):
    """Estimate the SHPs of each pixel of `amp_stack` on the GPU."""
    # Get the global position within the 2D GPU grid
    out_x, out_y = cuda.grid(2)
    out_rows, out_cols = is_shp.shape[:2]
    # Check if we are within the bounds of the array
    if out_y >= out_rows or out_x >= out_cols:
        return

    num_slc, rows, cols = sorted_amp_stack.shape
    ecdf_dist_cutoff = _get_ecdf_critical_distance(num_slc, alpha)
    half_row, half_col = halfwin_rowcol

    row_strides, col_strides = strides_rowcol
    r_start = row_strides // 2
    c_start = col_strides // 2
    in_r = r_start + out_y * row_strides
    in_c = c_start + out_x * col_strides

    # Get the input slices, clamping the window to the image bounds
    (r_start, r_end), (c_start, c_end) = _get_slices(
        half_row, half_col, in_r, in_c, rows, cols
    )

    amp_block = sorted_amp_stack[:, r_start:r_end, c_start:c_end]
    neighbors_pixel = is_shp[out_y, out_x, :, :]
    _set_neighbors(amp_block, halfwin_rowcol, ecdf_dist_cutoff, neighbors_pixel)
    remove_unconnected(is_shp[out_y, out_x], inplace=True)


@numba.njit(nogil=True)
def _set_neighbors(amp_block, halfwin_rowcol, ecdf_dist_cutoff, neighbors):
    _, rows, cols = amp_block.shape

    if rows < 2 * halfwin_rowcol[0] + 1 or cols < 2 * halfwin_rowcol[1] + 1:
        # not enough neighbors to test, make all false
        return

    # get the center pixel
    r_c, c_c = rows // 2, cols // 2
    x1 = amp_block[:, r_c, c_c]
    # TODO: if this is a bottleneck, we can do something smarter
    # by computing only the bottom right corner, then mirroring
    for i in range(rows):
        for j in range(cols):
            if i == r_c and j == c_c:
                neighbors[i, j] = False
                continue
            x2 = amp_block[:, i, j]

            ecdf_max_dist = _get_max_cdf_dist(x1, x2)

            neighbors[i, j] = ecdf_max_dist < ecdf_dist_cutoff


@numba.njit(nogil=True)
def _get_max_cdf_dist(x1, x2):
    """Get the maximum CDF distance between two arrays.

    Parameters
    ----------
    x1 : np.ndarray
        First array, size n, sorted
    x2 : np.ndarray
        Second array, size n, sorted

    Returns
    -------
    float
        Maximum empirical CDF distance between the two arrays

    Examples
    --------
    >>> x1 = np.array([1, 2, 3, 4, 5])
    >>> x2 = np.array([1, 2, 3, 4, 5])
    >>> print(f"{_get_max_cdf_dist(x1, x2):.1f}")  # doctest: +NUMBER
    0.0
    >>> x2 = np.array([2, 3, 4, 5, 6])
    >>> round(_get_max_cdf_dist(x1, x2), 2)  # doctest: +NUMBER
    0.2
    >>> round(_get_max_cdf_dist(x2, x1), 2)  # doctest: +NUMBER
    0.2
    >>> x2 = np.array([6, 7, 8, 9, 10])
    >>> _get_max_cdf_dist(x1, x2)  # doctest: +NUMBER
    1.0

    """
    n = x1.shape[0]
    i1 = i2 = i_out = 0
    cdf1 = cdf2 = 0
    max_dist = 0
    while i_out < 2 * n:
        if i1 == n:
            cdf2 += 1 / n
            i2 += 1
        elif i2 == n or (x1[i1] < x2[i2]):
            cdf1 += 1 / n
            i1 += 1
        elif x1[i1] > x2[i2]:
            cdf2 += 1 / n
            i2 += 1
        else:  # a tie
            cdf1 += 1 / n
            cdf2 += 1 / n
            i1 += 1
            i2 += 1
            i_out += 1  # jumping 2 ahead for tie
        i_out += 1
        max_dist = max(max_dist, abs(cdf1 - cdf2))
    return max_dist


@numba.njit(nogil=True)
def _get_ecdf_critical_distance(nslc, alpha):
    N = nslc / 2.0
    cur_dist = 0.01
    critical_distance = 0.1
    sqrt_N = sqrt(N)
    while cur_dist <= 1.0:
        value = cur_dist * (sqrt_N + 0.12 + 0.11 / sqrt_N)
        pvalue = 0
        for t in range(1, 101):
            pvalue += ((-1) ** (t - 1)) * exp(-2 * (value**2) * (t**2))
        pvalue = max(0.0, min(1.0, 2 * pvalue))

        if pvalue <= alpha:
            critical_distance = cur_dist
            break

        cur_dist += 0.001
    return critical_distance
