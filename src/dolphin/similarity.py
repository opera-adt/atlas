"""Module for computing phase similarity between complex interferogram pixels.

Uses metric from [@Wang2022AccuratePersistentScatterer] for similarity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Literal, Sequence

import numba
import numpy as np
from numpy.typing import ArrayLike

from dolphin._types import PathOrStr

logger = logging.getLogger("dolphin")


@numba.njit(nogil=True)
def phase_similarity(x1: ArrayLike, x2: ArrayLike):
    """Compute the similarity between two complex 1D vectors."""
    n = len(x1)
    out = 0.0
    for i in range(n):
        out += np.real(x1[i] * np.conj(x2[i]))
    return out / n


def median_similarity(
    ifg_stack: ArrayLike, search_radius: int, mask: ArrayLike | None = None
):
    """Compute the median similarity of each pixel and its neighbors.

    Resulting similarity matches Equation (5) of [@Wang2022AccuratePersistentScatterer]

    Parameters
    ----------
    ifg_stack : ArrayLike
        3D stack of complex interferograms, or floating point phase.
        Shape is (n_ifg, rows, cols)
    search_radius: int
        maximum radius (in pixels) to search for neighbors when comparing each pixel.
    mask: ArrayLike (optional)
        Array of mask from True/False indicating whether to include the pixel (True)
        or ignore it (False).

    Returns
    -------
    np.ndarray
        2D array (shape (rows, cols)) of the median similarity at each pixel.

    """
    return _create_loop_and_run(
        ifg_stack=ifg_stack,
        search_radius=search_radius,
        mask=mask,
        func=np.nanmedian,
    )


def max_similarity(
    ifg_stack: ArrayLike, search_radius: int, mask: ArrayLike | None = None
):
    """Compute the maximum similarity of each pixel and its neighbors.

    Resulting similarity matches Equation (6) of [@Wang2022AccuratePersistentScatterer]

    Parameters
    ----------
    ifg_stack : ArrayLike
        3D stack of complex interferograms, or floating point phase.
        Shape is (n_ifg, rows, cols)
    search_radius: int
        maximum radius (in pixels) to search for neighbors when comparing each pixel.
    mask: ArrayLike (optional)
        Array of mask from True/False indicating whether to include the pixel (True)
        or ignore it (False).

    Returns
    -------
    np.ndarray
        2D array (shape (rows, cols)) of the maximum similarity for any neighbor
        at a pixel.

    """
    return _create_loop_and_run(
        ifg_stack=ifg_stack,
        search_radius=search_radius,
        mask=mask,
        func=np.nanmax,
    )


def _create_loop_and_run(
    ifg_stack: ArrayLike,
    search_radius: int,
    mask: ArrayLike | None,
    func: Callable[[ArrayLike], np.ndarray],
):
    n_ifg, rows, cols = ifg_stack.shape
    # Mark any nans/all zeros as invalid
    invalid_mask = np.nan_to_num(ifg_stack).sum(axis=0) == 0
    if not np.iscomplexobj(ifg_stack):
        unit_ifgs = np.exp(1j * ifg_stack)
    else:
        unit_ifgs = np.exp(1j * np.angle(ifg_stack))
    out_similarity = np.full((rows, cols), fill_value=np.nan, dtype="float32")
    if mask is None:
        mask = np.ones((rows, cols), dtype="bool")
    mask[invalid_mask] = False

    if mask.shape != (rows, cols):
        raise ValueError(f"{ifg_stack.shape = }, but {mask.shape = }")

    idxs = get_circle_idxs(search_radius)
    loop_func = _make_loop_function(func)
    return loop_func(unit_ifgs, idxs, mask, out_similarity)


def _make_loop_function(
    summary_func: Callable[[ArrayLike], np.ndarray],
):
    """Create a JIT-ed function for some summary of the neighbors's similarity.

    E.g.: for median similarity, call

        median_sim = _make_loop_function(np.median)
    """

    @numba.njit(nogil=True, parallel=True)
    def _masked_sim_loop(
        ifg_stack: np.ndarray,
        idxs: np.ndarray,
        mask: np.ndarray,
        out_similarity: np.ndarray,
    ) -> np.ndarray:
        """Loop over each pixel, make a masked phase similarity to its neighbors."""
        _, rows, cols = ifg_stack.shape

        num_compare_pixels = len(idxs)
        # Buffer to hold all comparison during the parallel loop
        cur_sim = np.zeros((rows, cols, num_compare_pixels))

        for r0 in numba.prange(rows):
            for c0 in range(cols):
                # Get the current pixel
                m0 = mask[r0, c0]
                if not m0:
                    continue
                x0 = ifg_stack[:, r0, c0]

                cur_sim_vec = cur_sim[r0, c0]
                count = 0

                # compare to all pixels in the circle around it
                for i_idx in range(num_compare_pixels):
                    ir, ic = idxs[i_idx]
                    # Clip to the image bounds
                    r = max(min(r0 + ir, rows - 1), 0)
                    c = max(min(c0 + ic, cols - 1), 0)
                    if r == r0 and c == c0:
                        continue

                    # Check for a pixel to ignore
                    if not mask[r, c]:
                        continue

                    x = ifg_stack[:, r, c]
                    # cur_sim_vec[count] = w * phase_similarity(x0, x)
                    cur_sim_vec[count] = phase_similarity(x0, x)
                    count += 1
                    # Assuming `summary_func` is nan-aware
                out_similarity[r0, c0] = summary_func(cur_sim_vec[:count])
        return out_similarity

    return _masked_sim_loop


def get_circle_idxs(
    max_radius: int, min_radius: int = 0, sort_output: bool = True
) -> np.ndarray:
    """Get the relative indices of neighboring pixels in a circle.

    Adapted from c++ version of `psps` package:
    https://github.com/UT-Radar-Interferometry-Group/psps/blob/a15d458817fe7d06a6edaa0b3208ea78bc4782e7/src/cpp/similarity.cpp#L16
    """
    # using the mid-point circle drawing algorithm to search for neighboring PS pixels
    # # code adapted from "https://www.geeksforgeeks.org/mid-point-circle-drawing-algorithm/"
    visited = np.zeros((max_radius, max_radius), dtype=bool)
    visited[0][0] = True

    indices = []
    for r in range(1, max_radius):
        x = r
        y = 0
        p = 1 - r
        if r > min_radius:
            indices.append([r, 0])
            indices.append([-r, 0])
            indices.append([0, r])
            indices.append([0, -r])

        visited[r][0] = True
        visited[0][r] = True
        # flag > 0 means there are holes between concentric circles
        flag = 0
        while x > y:
            # do not need to fill holes
            if flag == 0:
                y += 1
                if p <= 0:
                    # Mid-point is inside or on the perimeter
                    p += 2 * y + 1
                else:
                    # Mid-point is outside the perimeter
                    x -= 1
                    p += 2 * y - 2 * x + 1

            else:
                flag -= 1

            # All the perimeter points have already been visited
            if x < y:
                break

            while not visited[x - 1][y]:
                x -= 1
                flag += 1

            visited[x][y] = True
            visited[y][x] = True
            if r > min_radius:
                indices.append([x, y])
                indices.append([-x, -y])
                indices.append([x, -y])
                indices.append([-x, y])

                if x != y:
                    indices.append([y, x])
                    indices.append([-y, -x])
                    indices.append([y, -x])
                    indices.append([-y, x])

            if flag > 0:
                x += 1

    if sort_output:
        # Sorting makes it run faster, better data access patterns
        return np.array(sorted(indices))
    else:
        # Indices run from middle outward
        return np.array(indices)


def create_similarities(
    ifg_file_list: Sequence[PathOrStr],
    output_file: PathOrStr,
    search_radius: int = 7,
    sim_type: Literal["median", "max"] = "median",
    block_shape: tuple[int, int] = (512, 512),
    num_threads: int = 5,
    add_overviews: bool = True,
    nearest_n: int | None = None,
):
    """Create a similarity raster from as stack of ifg files.

    Parameters
    ----------
    ifg_file_list : Sequence[PathOrStr]
        Paths to input interferograms
    output_file : PathOrStr
        Output raster path
    search_radius : int, optional
        Maximum radius to search for pixels, by default 7
    sim_type : str, optional
        Type of similarity function to run, by default "median"
        Choices: "median", "max"
    block_shape : tuple[int, int], optional
        Size of blocks to process at one time from `ifg_file_list`
        by default (512, 512)
    num_threads : int, optional
        Number of parallel blocks to process, by default 5
    add_overviews : bool, optional
        Whether to create overviews in `output_file` by default True
    nearest_n : int, optional
        If provided, reform the nearest N interferograms before computing similarity.

    """
    from dolphin._overviews import Resampling, create_image_overviews
    from dolphin.io import BackgroundRasterWriter, VRTStack, process_blocks
    from dolphin.timeseries import get_incidence_matrix

    if Path(output_file).exists():
        logger.info(f"{output_file} exists, skipping")
        return

    if sim_type == "median":
        sim_function = median_similarity
    elif sim_type == "max":
        sim_function = max_similarity
    else:
        raise ValueError(f"Unrecognized {sim_type = }")

    nodata_block = np.full(block_shape, fill_value=np.nan, dtype="float32")

    if nearest_n is not None:
        incidence_matrix = get_incidence_matrix(
            _create_nearest_n_pairs(len(ifg_file_list) + 1, n=nearest_n)
        )
        assert incidence_matrix.shape[1] == len(ifg_file_list)
    else:
        incidence_matrix = None

    def calc_sim(readers, rows, cols):
        block = readers[0][:, rows, cols]
        if np.sum(block) == 0 or np.isnan(block).all():
            return nodata_block[rows, cols], rows, cols

        if incidence_matrix is not None:
            block = _calc_nearest_diffs(block, incidence_matrix)

        out_avg = sim_function(ifg_stack=block, search_radius=search_radius)
        logger.debug(f"{rows = }, {cols = }, {block.shape = }, {out_avg.shape = }")
        return out_avg, rows, cols

    out_dir = Path(output_file).parent
    reader = VRTStack(ifg_file_list, outfile=out_dir / "sim_inputs.vrt")

    writer = BackgroundRasterWriter(
        output_file,
        like_filename=ifg_file_list[0],
        dtype="float32",
        driver="GTiff",
        nodata=np.nan,
    )
    process_blocks(
        [reader],
        writer,
        func=calc_sim,
        block_shape=block_shape,
        overlaps=(search_radius, search_radius),
        num_threads=num_threads,
    )
    writer.notify_finished()

    if add_overviews:
        logger.info("Creating overviews for unwrapped images")
        create_image_overviews(Path(output_file), resampling=Resampling.AVERAGE)


def _calc_nearest_diffs(block, incidence_matrix) -> np.ndarray:
    # Multiply the single-ref data by tall and skinny A matrix
    # to give the nearest-n differences
    num_imgs, rows, cols = block.shape
    block_mask = np.nan_to_num(block).sum(axis=0) == 0
    m, num_imgs = incidence_matrix.shape
    phase = np.angle(block) if np.iscomplexobj(block) else block
    columns = np.dot(incidence_matrix, phase.reshape(num_imgs, -1))
    block = columns.reshape(m, rows, cols)
    block[:, block_mask] = np.nan
    return np.exp(1j * block)


def _create_nearest_n_pairs(num_files: int, n: int = 3) -> list[tuple[int, int]]:
    """Create nearest-n interferogram pair indices for a list of `num_files` inputs."""
    ijs = []
    for i in range(num_files):
        for j in range(i + 1, i + n + 1):
            if j >= num_files:
                continue
            ijs.append((i, j))
    return ijs
