from __future__ import annotations

import logging
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Optional, Protocol, Sequence, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, jit, lax, vmap
from numpy.typing import ArrayLike, NDArray
from opera_utils import get_dates
from scipy import ndimage

from dolphin import io
from dolphin._overviews import ImageType, create_overviews
from dolphin._types import PathOrStr, ReferencePoint
from dolphin.utils import flatten, format_dates, full_suffix, get_nearest_date_idx

T = TypeVar("T")
DateOrDatetime = datetime | date
logger = logging.getLogger("dolphin")

__all__ = ["run"]


class InversionMethod(str, Enum):
    """Method to use for timeseries inversion."""

    L1 = "L1"
    L2 = "L2"


class ReferencePointError(ValueError):
    pass


def run(
    unwrapped_paths: Sequence[PathOrStr],
    conncomp_paths: Sequence[PathOrStr] | None,
    quality_file: PathOrStr,
    output_dir: PathOrStr,
    method: InversionMethod = InversionMethod.L1,
    reference_candidate_threshold: float = 0.95,
    run_velocity: bool = False,
    corr_paths: Sequence[PathOrStr] | None = None,
    weight_velocity_by_corr: bool = False,
    velocity_file: Optional[PathOrStr] = None,
    correlation_threshold: float = 0.0,
    block_shape: tuple[int, int] = (256, 256),
    num_threads: int = 4,
    reference_point: tuple[int, int] = (-1, -1),
    wavelength: float | None = None,
    add_overviews: bool = True,
    extra_reference_date: datetime | None = None,
) -> tuple[list[Path], list[Path] | None, ReferencePoint]:
    """Invert the unwrapped interferograms, estimate timeseries and phase velocity.

    Parameters
    ----------
    unwrapped_paths : Sequence[Path]
        Sequence unwrapped interferograms to invert.
    conncomp_paths : Sequence[Path]
        Sequence connected component files, one per file in `unwrapped_paths`
    quality_file: PathOrStr
        A file with the same size as each raster, like amplitude dispersion or
        temporal coherence
    output_dir : Path
        Path to the output directory.
    method : str, choices = "L1", "L2"
        Inversion method to use when solving Ax = b.
        Default is L2, which uses least squares to solve Ax = b (faster).
        "L1" minimizes |Ax - b|_1 at each pixel.
    reference_candidate_threshold: float
        The threshold for the quality metric to be considered a candidate
        reference point pixel.
        Only pixels with values in `quality_file` greater than
        `reference_candidate_threshold` will be considered a candidate.
        Default is 0.95.
    run_velocity : bool
        Whether to run velocity estimation on the inverted phase series
    corr_paths : Sequence[Path], optional
        Sequence interferometric correlation files, one per file in `unwrapped_paths`.
        If not provided, does no weighting by correlation.
    weight_velocity_by_corr : bool
        Flag to indicate whether the velocity fitting should use correlation as weights.
        Default is False.
    velocity_file : Path, Optional
        The output velocity file
    correlation_threshold : float
        Pixels with correlation below this value will be masked out
    block_shape : tuple[int, int], optional
        The shape of the blocks to process in parallel.
        Default is (256, 256)
    num_threads : int
        The parallel blocks to process at once.
        Default is 4.
    reference_point : tuple[int, int], optional
        Reference point (row, col) used if performing a time series inversion.
        If not provided, a point will be selected from a consistent connected
        component with low amplitude dispersion or high temporal coherence.
    wavelength : float, optional
        The wavelength of the radar signal, in meters.
        If provided, the output rasters are in meters and meters / year for
        the displacement and velocity rasters.
        If not provided, the outputs are in radians.
        See Notes for line of sight convention.
    add_overviews : bool, optional
        If True, creates overviews of the new velocity raster.
        Default is True.
    extra_reference_date : datetime.datetime, optional
        If provided, makes another set of interferograms referenced to this
        for all dates later than it.

    Returns
    -------
    inverted_phase_paths : list[Path]
        list of Paths to inverted interferograms (single reference phase series).
    residual_paths : list[Path] | None
        list of Paths to timeseries inversion residuals.
        If no inversion is performed, this is be None.
    reference_point : ReferencePoint
        NamedTuple of reference (row, column) selected.
        If passed as input, simply returned back as output.
        Otherwise, the result is the auto-selection from `select_reference_point`.

    Notes
    -----
    When wavelength is provided, the output rasters are in meters and meters / year,
    where positive values indicate motion *toward* from the radar (i.e. positive values
    in both ascending and descending tracks imply uplift).

    """
    unwrapped_paths = sorted(unwrapped_paths, key=str)
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    if reference_point == (-1, -1):
        logger.info("Selecting a reference point for unwrapped interferograms")
        ref_point = select_reference_point(
            quality_file=quality_file,
            output_dir=Path(output_dir),
            candidate_threshold=reference_candidate_threshold,
            ccl_file_list=conncomp_paths,
        )
    else:
        ref_point = ReferencePoint(row=reference_point[0], col=reference_point[1])

    ifg_date_pairs = [get_dates(f) for f in unwrapped_paths]
    sar_dates = sorted(set(flatten(ifg_date_pairs)))
    # if we did single-reference interferograms, for `n` sar dates, we will only have
    # `n-1` interferograms. Any more than n-1 ifgs means we need to invert
    is_single_reference = len(unwrapped_paths) == len(sar_dates) - 1
    # TODO: Do we ever want to invert this case: the "trivial" network,
    # which has 1 ifg per date difference, but a moving reference date?
    # The extra condition to check is
    # ... and all(pair[0] == ifg_date_pairs[0][0] for pair in ifg_date_pairs)

    # check if we even need to invert, or if it was single reference
    inverted_phase_paths: list[Path] = []
    if is_single_reference:
        logger.info(
            "Skipping inversion step: only single reference interferograms exist."
        )
        # Copy over the unwrapped paths to `timeseries/`
        final_ts_paths = _convert_and_reference(
            unwrapped_paths,
            output_dir=output_dir,
            reference_point=ref_point,
            wavelength=wavelength,
        )
        final_residual_paths = None
    else:
        logger.info("Inverting network of %s unwrapped ifgs", len(unwrapped_paths))
        inverted_phase_paths, residual_paths = invert_unw_network(
            unw_file_list=unwrapped_paths,
            conncomp_file_list=conncomp_paths,
            reference=ref_point,
            output_dir=output_dir,
            block_shape=block_shape,
            num_threads=num_threads,
            wavelength=wavelength,
            method=method,
        )
        if extra_reference_date is None:
            final_ts_paths = inverted_phase_paths
            final_residual_paths = residual_paths
        else:
            final_ts_paths, final_residual_paths = _redo_reference(
                inverted_phase_paths, residual_paths, extra_reference_date
            )

    if add_overviews:
        logger.info("Creating overviews for timeseries images")
        create_overviews(final_ts_paths, image_type=ImageType.UNWRAPPED, max_workers=2)

    if run_velocity:
        logger.info("Estimating phase velocity")

        #  We can't pass the correlations after an inversion- the numbers don't match
        # TODO:
        # Is there a better weighting then?
        if not weight_velocity_by_corr or corr_paths is None:
            cor_file_list = None
        else:
            cor_file_list = (
                corr_paths if len(corr_paths) == len(final_ts_paths) else None
            )

        if velocity_file is None:
            velocity_file = Path(output_dir) / "velocity.tif"

        create_velocity(
            unw_file_list=final_ts_paths,
            output_file=velocity_file,
            reference=ref_point,
            cor_file_list=cor_file_list,
            cor_threshold=correlation_threshold,
            block_shape=block_shape,
            num_threads=num_threads,
        )

    return final_ts_paths, final_residual_paths, ref_point


def _redo_reference(
    inverted_phase_paths: Sequence[Path],
    residual_paths: Sequence[Path],
    extra_reference_date: datetime,
):
    """Reset the reference date in `inverted_phase_paths`.

    Affects all files whose secondary is after `extra_reference_date`.

    E.g Given the (day 1, day 2), ..., (day 1, day N) pairs, outputs
    (1, 2), (1, 3), ...(1, r), (r, r+1), ..., (r, N)
    where r is the index of the `extra_reference_date`

    Also resets the reference date in `residual_paths` with a simple renaming
    rather than a new difference calculation.
    We only need to rename because it's the *secondary* date that matters for
    the residual calculation.
    """
    output_path = inverted_phase_paths[0].parent
    inverted_date_pairs: list[tuple[datetime, datetime]] = [
        get_dates(p.stem)[:2] for p in inverted_phase_paths
    ]
    secondary_dates = [pair[1] for pair in inverted_date_pairs]
    extra_ref_idx = get_nearest_date_idx(
        secondary_dates, requested=extra_reference_date
    )
    ref_date = secondary_dates[extra_ref_idx]
    logger.info(f"Re-referencing later timeseries files to {ref_date}")
    extra_ref_img = inverted_phase_paths[extra_ref_idx]
    ref = io.load_gdal(extra_ref_img, masked=True)

    # Use a temp directory while re-referencing
    extra_out_dir = inverted_phase_paths[0].parent / "extra"
    extra_out_dir.mkdir(exist_ok=True)
    units = io.get_raster_units(inverted_phase_paths[0])

    # Copy the first part
    final_residual_paths = list(residual_paths[: extra_ref_idx + 1])
    for idx in range(extra_ref_idx + 1, len(inverted_date_pairs)):
        # To create the interferogram (r, r+1), we subtract
        # (1, r) from (1, r+1)
        cur_img = inverted_phase_paths[idx]
        new_stem = format_dates(ref_date, secondary_dates[idx])
        cur_output_name = extra_out_dir / f"{new_stem}.tif"
        cur = io.load_gdal(cur_img, masked=True)
        new_out = cur - ref
        io.write_arr(
            arr=new_out,
            like_filename=extra_ref_img,
            output_name=cur_output_name,
            units=units,
        )

        # rename the reference date in the residual timeseries
        new_residual_name = f"residuals_{new_stem}.tif"
        final_residual_paths.append(
            residual_paths[idx].rename(output_path / new_residual_name)
        )

    for idx, p in enumerate(inverted_phase_paths):
        if idx <= extra_ref_idx:
            p.rename(extra_out_dir / p.name)
        else:
            p.unlink()
    # Finally, move them back in to the `timeseries/` folder
    final_out = []
    for p in extra_out_dir.glob("*.tif"):
        final_out.append(p.rename(output_path / p.name))
    return sorted(final_out), sorted(final_residual_paths)


def _convert_and_reference(
    unwrapped_paths: Sequence[PathOrStr],
    *,
    output_dir: PathOrStr,
    reference_point: ReferencePoint,
    wavelength: float | None = None,
) -> list[Path]:
    if wavelength is not None:
        # Positive values are motion towards the radar
        constant = -1 * (wavelength / (4 * np.pi))
        units = "meters"
    else:
        constant = -1
        units = "radians"

    ref_row, ref_col = reference_point
    out_paths: list[Path] = []
    for p in unwrapped_paths:
        # if it ends in `.unw.tif`, change to just `.tif` for consistency
        # with the case where we run an inversion
        cur_name = Path(p).name
        unw_suffix = full_suffix(p)
        target_name = str(cur_name).replace(unw_suffix, ".tif")
        target = Path(output_dir) / target_name
        out_paths.append(target)

        if target.exists():  # Check to prevent overwriting
            continue

        arr_radians = io.load_gdal(p, masked=True)
        nodataval = io.get_raster_nodata(p)
        # Reference to the
        ref_value = arr_radians.filled(np.nan)[ref_row, ref_col]
        if np.isnan(ref_value):
            logger.warning(
                f"{ref_value!r} is NaN for {p} . Skipping reference subtraction."
            )
        else:
            arr_radians -= ref_value
        # Make sure we keep the same mask as the original
        out_arr = (arr_radians * constant).filled(nodataval)
        io.write_arr(
            arr=out_arr,
            output_name=target,
            units=units,
            like_filename=p,
            nodata=nodataval,
        )

    return out_paths


def argmin_index(arr: ArrayLike) -> tuple[int, ...]:
    """Get the index tuple of the minimum value of the array.

    If multiple occurrences of the minimum value exist, returns
    the index of the first such occurrence in the flattened array.

    Parameters
    ----------
    arr : array_like
        The input array.

    Returns
    -------
    tuple of int
        The index of the minimum value.

    """
    return np.unravel_index(np.argmin(arr), np.shape(arr))


def argmax_index(arr: ArrayLike) -> tuple[int, ...]:
    """Get the index tuple of the maximum value of the array.

    If multiple occurrences of the maximum value exist, returns
    the index of the first such occurrence in the flattened array.

    Parameters
    ----------
    arr : array_like
        The input array.

    Returns
    -------
    tuple of int
        The index of the maximum value.

    """
    return np.unravel_index(np.argmax(arr), np.shape(arr))


@jit
def censored_lstsq(A, B, M):
    """Solves least squares problem subject to missing data in the right hand side.

    Parameters
    ----------
    A : ndarray
        m x n system matrix.
    B : ndarray
        m x k matrix representing the k right hand side data vectors of size m.
    M : ndarray
        m x k boolean matrix of missing data (`False` indicate missing values)

    Returns
    -------
    X : ndarray
        n x k matrix that minimizes norm(M*(AX - B))
    residuals : np.array 1D
        Sums of (k,) squared residuals: squared Euclidean 2-norm for `b - A @ x`

    Reference
    ---------
    http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/

    """
    # if B is a vector, simply drop out corresponding rows in A
    if B.ndim == 1 or B.shape[1] == 1:
        return jnp.linalg.lstsq(A[M], B[M])[0]

    # else solve via tensor representation
    rhs = jnp.dot(A.T, M * B).T[:, :, None]  # k x n x 1 tensor
    T = jnp.matmul(A.T[None, :, :], M.T[:, :, None] * A[None, :, :])  # k x n x n tensor
    x = jnp.squeeze(jnp.linalg.solve(T, rhs)).T  # transpose to get n x k
    residuals = jnp.linalg.norm(A @ x - (B * M.astype(int)), axis=0)
    return x, residuals


@jit
def weighted_lstsq_single(
    A: ArrayLike,
    b: ArrayLike,
    weights: ArrayLike,
) -> Array:
    r"""Perform weighted least squares for one data vector.

    Minimizes the weighted 2-norm of the residual vector:

    \[
        || b - A x ||^2_W
    \]

    where \(W\) is a diagonal matrix of weights.

    Parameters
    ----------
    A : ArrayLike
        Incidence matrix of shape (n_ifgs, n_sar_dates - 1)
    b : ArrayLike, 1D
        The phase differences between the ifg pairs
    weights : ArrayLike, 1D, optional
        The weights for each element of `b`.

    Returns
    -------
    x : np.array 1D
        The estimated phase for each SAR acquisition
    residuals : np.array 1D
        Sums of squared residuals: Squared Euclidean 2-norm for `b - A @ x`
        For a 1D `b`, this is a scalar.

    """
    # scale both A and b by sqrt so we are minimizing
    sqrt_weights = jnp.sqrt(weights)
    # Multiply each data point by sqrt(weight)
    b_scaled = b * sqrt_weights
    # Multiply each row by sqrt(weight)
    A_scaled = A * sqrt_weights[:, None]

    # Run the weighted least squares
    x, residuals, rank, sing_vals = jnp.linalg.lstsq(A_scaled, b_scaled)
    # TODO: do we need special handling?
    # if rank < A.shape[1]:
    #     # logger.warning("Rank deficient solution")

    return x, residuals.ravel()


@jit
def invert_stack(
    A: ArrayLike,
    dphi: ArrayLike,
    weights: ArrayLike | None = None,
    missing_data_flags: ArrayLike | None = None,
) -> tuple[Array, Array]:
    """Solve the SBAS problem for a stack of unwrapped phase differences.

    Parameters
    ----------
    A : ArrayLike
        Incidence matrix of shape (n_ifgs, n_sar_dates - 1)
    dphi : ArrayLike
        The phase differences between the ifg pairs, shape=(n_ifgs, n_rows, n_cols)
    weights : ArrayLike, optional
        The weights for each element of `dphi`.
        Same shape as `dphi`.
        If not provided, all weights are set to 1 (ordinary least squares).
    missing_data_flags : ArrayLike, optional
        Boolean matrix, same shape as `dphi`, indicating a missing value in `dphi`.
        If provided, the least squares result will ignore these entries.
        Example may come from having connected component masks indicate unreliable
        values in `dphi` for certain interferograms.

    Returns
    -------
    phi : np.array 3D
        The estimated phase for each SAR acquisition
        Shape is (n_sar_dates - 1, n_rows, n_cols)
    residuals : np.array 2D
        Sums of squared residuals: Squared Euclidean 2-norm for `dphi - A @ x`
        Shape is (n_rows, n_cols)

    Notes
    -----
    To mask out data points of a pixel, the weight can be set to 0.
    When `A` remains full rank, setting the weight to zero is the same as removing
    the entry from the data vector and the corresponding row from `A`.

    """
    n_ifgs, n_rows, n_cols = dphi.shape

    if weights is None:
        # Can use ordinary least squares with no weights
        # Reshape to be size (M, K) instead of 3D
        b = dphi.reshape(n_ifgs, -1)
        phase_cols, residuals_cols, _, _ = jnp.linalg.lstsq(A, b)
        # Reshape the phase and residuals to be 3D
        phase = phase_cols.reshape(-1, n_rows, n_cols)
        residuals = residuals_cols.reshape(n_rows, n_cols)
    elif missing_data_flags is not None:
        b = dphi.reshape(n_ifgs, -1)
        missing_data = missing_data_flags.reshape(n_ifgs, -1)
        phase_cols, residuals_cols = censored_lstsq(A, b, missing_data)
        phase = phase_cols.reshape(-1, n_rows, n_cols)
        residuals = residuals_cols.reshape(n_rows, n_cols)
    else:
        # vectorize the solve function to work on 2D and 3D arrays
        # We are not vectorizing over the A matrix, only the dphi vector
        # Solve 2d shapes: (nrows, n_ifgs) -> (nrows, n_sar_dates)
        invert_2d = vmap(weighted_lstsq_single, in_axes=(None, 1, 1), out_axes=(1, 1))
        # Solve 3d shapes: (nrows, ncols, n_ifgs) -> (nrows, ncols, n_sar_dates)
        invert_3d = vmap(invert_2d, in_axes=(None, 2, 2), out_axes=(2, 2))
        phase, residuals = invert_3d(A, dphi, weights)
        # Reshape the residuals to be 2D
        residuals = residuals[0]

    return phase, residuals


def get_incidence_matrix(
    ifg_pairs: Sequence[tuple[T, T]],
    sar_idxs: Sequence[T] | None = None,
    delete_first_date_column: bool = True,
) -> np.ndarray:
    """Build the indicator matrix from a list of ifg pairs (index 1, index 2).

    Parameters
    ----------
    ifg_pairs : Sequence[tuple[T, T]]
        List of ifg pairs represented as tuples of (day 1, day 2)
        Can be ints, datetimes, etc.
    sar_idxs : Sequence[T], optional
        If provided, used as the total set of indexes which `ifg_pairs`
        were formed from.
        Otherwise, created from the unique entries in `ifg_pairs`.
        Only provide if there are some dates which are not present in `ifg_pairs`.
    delete_first_date_column : bool
        If True, removes the first column of the matrix to make it full column rank.
        Size will be `n_sar_dates - 1` columns.
        Otherwise, the matrix will have `n_sar_dates`, but rank `n_sar_dates - 1`.

    Returns
    -------
    A : np.array 2D
        The incident-like matrix for the system: A*phi = dphi
        Each row corresponds to an ifg, each column to a SAR date.
        The value will be -1 on the early (reference) ifgs, +1 on later (secondary)
        since the ifg phase = (later - earlier)
        Shape: (n_ifgs, n_sar_dates - 1)

    """
    if sar_idxs is None:
        sar_idxs = sorted(set(flatten(ifg_pairs)))

    M = len(ifg_pairs)
    col_iter = sar_idxs[1:] if delete_first_date_column else sar_idxs
    N = len(col_iter)
    A = np.zeros((M, N))

    # Create a dictionary mapping sar dates to matrix columns
    # We take the first SAR acquisition to be time 0, leave out of matrix
    date_to_col = {date: i for i, date in enumerate(col_iter)}
    for i, (early, later) in enumerate(ifg_pairs):
        if early in date_to_col:
            A[i, date_to_col[early]] = -1
        if later in date_to_col:
            A[i, date_to_col[later]] = +1

    return A


@jit
def estimate_velocity_pixel(x: ArrayLike, y: ArrayLike, w: ArrayLike) -> Array:
    """Estimate the velocity from a single pixel's time series.

    Parameters
    ----------
    x : np.array 1D
        The time values
    y : np.array 1D
        The unwrapped phase values
    w : np.array 1D
        The weights for each time value

    Returns
    -------
    velocity : np.array, 0D
        The estimated velocity in (unw unit) / year.

    """
    # Jax polyfit will grab the first *2* dimensions of y to solve in a batch
    return jnp.polyfit(x, y, deg=1, w=w.reshape(y.shape), rcond=None)[0]


@jit
def estimate_velocity(
    x_arr: ArrayLike, unw_stack: ArrayLike, weight_stack: ArrayLike | None
) -> Array:
    """Estimate the velocity from a stack of unwrapped interferograms.

    Parameters
    ----------
    x_arr : ArrayLike
        Array of time values corresponding to each unwrapped phase image.
        Length must match `unw_stack.shape[0]`.
    unw_stack : ArrayLike
        Array of unwrapped phase values at each pixel, shape=`(n_time, n_rows, n_cols)`.
    weight_stack : ArrayLike, optional
        Array of weights for each pixel, shape=`(n_time, n_rows, n_cols)`.
        If not provided, performs one batch unweighted linear fit.

    Returns
    -------
    velocity : np.array 2D
        The estimated velocity in (unw unit) / year calculated as 365.25 * rad/day.
        E.g. if the unwrapped phase is in radians, the velocity is in rad/year.

    """
    # TODO: weighted least squares using correlation?
    n_time, n_rows, n_cols = unw_stack.shape

    unw_pixels = unw_stack.reshape(n_time, -1)
    if weight_stack is None:
        # For jnp.polyfit(...), coeffs[0] is slope, coeffs[1] is the intercept
        velocities = jnp.polyfit(x_arr, unw_pixels, deg=1, rcond=None)[0]
    else:
        # We use the same x inputs for all output pixels
        if unw_stack.shape != weight_stack.shape:
            msg = (
                "unw_stack and weight_stack must have the same shape,"
                f" got {unw_stack.shape} and {weight_stack.shape}"
            )
            raise ValueError(msg)

        weights_pixels = weight_stack.reshape(n_time, 1, -1)

        velocities = vmap(estimate_velocity_pixel, in_axes=(None, -1, -1))(
            x_arr, unw_pixels, weights_pixels
        )
    # Currently `velocities` is in units / day,
    days_per_year = 365.25
    return velocities.reshape(n_rows, n_cols) * days_per_year


def datetime_to_float(dates: Sequence[DateOrDatetime]) -> np.ndarray:
    """Convert a sequence of datetime objects to a float representation.

    Output units are in days since the first item in `dates`.

    Parameters
    ----------
    dates : Sequence[DateOrDatetime]
        List of datetime objects to convert to floats

    Returns
    -------
    date_arr : np.array 1D
        The float representation of the datetime objects

    """
    sec_per_day = 60 * 60 * 24
    date_arr = np.asarray(dates).astype("datetime64[s]")
    # Reference the 0 to the first date
    date_arr = date_arr - date_arr[0]
    return date_arr.astype(float) / sec_per_day


def create_velocity(
    unw_file_list: Sequence[PathOrStr],
    output_file: PathOrStr,
    reference: ReferencePoint,
    date_list: Sequence[DateOrDatetime] | None = None,
    cor_file_list: Sequence[PathOrStr] | None = None,
    cor_threshold: float = 0.2,
    block_shape: tuple[int, int] = (256, 256),
    num_threads: int = 4,
    add_overviews: bool = True,
) -> None:
    """Perform pixel-wise (weighted) linear regression to estimate velocity.

    The units of `output_file` are in (unwrapped units) / year.
    E.g. if the files in `unw_file_list` are in radians, the output velocity
    is in radians / year, which is calculated as 365.25 * radians / day.

    Parameters
    ----------
    unw_file_list : Sequence[PathOrStr]
        List of unwrapped phase files.
    output_file : PathOrStr
        The output file to save the velocity to.
    reference : ReferencePoint
        The (row, col) to use as reference before fitting the velocity.
        This point will be subtracted from all other points before solving.
    date_list : Sequence[DateOrDatetime], optional
        List of dates corresponding to the unwrapped phase files.
        If not provided, will be parsed from filenames in `unw_file_list`.
    cor_file_list : Sequence[PathOrStr], optional
        List of correlation files to use for weighting the velocity estimation.
        If not provided, all weights are set to 1.
    cor_threshold : float, optional
        The correlation threshold to use for weighting the velocity estimation.
        Default is 0.2.
    block_shape : tuple[int, int], optional
        The shape of the blocks to process in parallel.
        Default is (256, 256)
    num_threads : int, optional
        The parallel blocks to process at once.
        Default is 4.
    add_overviews : bool, optional
        If True, creates overviews of the new velocity raster.
        Default is True.

    """
    if Path(output_file).exists():
        logger.info(f"Output file {output_file} already exists, skipping velocity")
        return

    if date_list is None:
        date_list = [get_dates(f)[1] for f in unw_file_list]
    x_arr = datetime_to_float(date_list)

    # Set up the input data readers
    out_dir = Path(output_file).parent
    unw_reader = io.VRTStack(
        file_list=unw_file_list,
        outfile=out_dir / "velocity_inputs.vrt",
        skip_size_check=True,
    )
    if cor_file_list is not None:
        if len(cor_file_list) != len(unw_file_list):
            msg = "Mismatch in number of input files provided:"
            msg += f"{len(cor_file_list) = }, but {len(unw_file_list) = }"
            raise ValueError(msg)

        logger.info("Using correlation to weight velocity fit")
        cor_reader = io.VRTStack(
            file_list=cor_file_list,
            outfile=out_dir / "cor_inputs.vrt",
            skip_size_check=True,
        )
    else:
        logger.info("Using unweighted fit for velocity.")
        cor_reader = None

    # Read in the reference point
    ref_row, ref_col = reference
    logger.info(f"Reading phase reference pixel {reference}")
    ref_data = unw_reader[:, ref_row, ref_col].reshape(-1, 1, 1)

    def read_and_fit(
        readers: Sequence[io.StackReader], rows: slice, cols: slice
    ) -> tuple[np.ndarray, slice, slice]:
        # Only use the cor_reader if it's the same shape as the unw_reader
        if len(readers) == 2:
            unw_reader, cor_reader = readers
            unw_stack = unw_reader[:, rows, cols]
            weights = cor_reader[:, rows, cols]
            weights[weights < cor_threshold] = 0
        else:
            unw_stack = readers[0][:, rows, cols]
            # weights = np.ones_like(unw_stack)
            weights = None
        # Reference the data
        unw_stack = unw_stack - ref_data
        # Fit a line to each pixel with weighted least squares
        return (
            estimate_velocity(x_arr=x_arr, unw_stack=unw_stack, weight_stack=weights),
            rows,
            cols,
        )

    # Note: For some reason, the `RasterStackReader` is much slower than the VRT
    # for files on S3:
    # ~300 files takes >2 min to open, >2 min to read each block
    # VRTStack seems to take ~30 secs to open, 1 min to read
    # Very possible there's a tuning param/rasterio config to fix, but not sure.
    readers = [unw_reader]
    if cor_reader is not None:
        readers.append(cor_reader)
    writer = io.BackgroundRasterWriter(output_file, like_filename=unw_file_list[0])
    io.process_blocks(
        readers=readers,
        writer=writer,
        func=read_and_fit,
        block_shape=block_shape,
        num_threads=num_threads,
    )

    writer.notify_finished()
    if add_overviews:
        logger.info("Creating overviews for velocity image")
        create_overviews([output_file])
    if units := io.get_raster_units(unw_file_list[0]):
        io.set_raster_units(output_file, units=f"{units} / year")
    logger.info("Completed create_velocity")


class AverageFunc(Protocol):
    """Protocol for temporally averaging a block of data."""

    def __call__(self, ArrayLike, axis: int) -> ArrayLike: ...


def create_temporal_average(
    file_list: Sequence[PathOrStr],
    output_file: PathOrStr,
    block_shape: tuple[int, int] = (256, 256),
    num_threads: int = 4,
    average_func: Callable[[ArrayLike, int], np.ndarray] = np.nanmean,
    read_masked: bool = False,
) -> None:
    """Average all images in `reader` to create a 2D image in `output_file`.

    Parameters
    ----------
    file_list : Sequence[PathOrStr]
        List of files to average
    output_file : PathOrStr
        The output file to save the average to
    block_shape : tuple[int, int], optional
        The shape of the blocks to process in parallel.
        Default is (256, 256)
    num_threads : int, optional
        The parallel blocks to process at once.
        Default is 4.
    average_func : Callable[[ArrayLike, int], np.ndarray], optional
        The function to use to average the images.
        Default is `np.nanmean`, which calls `np.nanmean(arr, axis=0)` on each block.
    read_masked : bool, optional
        If True, reads the data as a masked array based on the rasters' nodata values.
        Default is False.

    """

    def read_and_average(
        readers: Sequence[io.StackReader], rows: slice, cols: slice
    ) -> tuple[slice, slice, np.ndarray]:
        chunk = readers[0][:, rows, cols]
        return average_func(chunk, 0), rows, cols

    writer = io.BackgroundRasterWriter(output_file, like_filename=file_list[0])
    with NamedTemporaryFile(mode="w", suffix=".vrt") as f:
        reader = io.VRTStack(
            file_list=file_list,
            outfile=f.name,
            skip_size_check=True,
            read_masked=read_masked,
        )

        io.process_blocks(
            readers=[reader],
            writer=writer,
            func=read_and_average,
            block_shape=block_shape,
            num_threads=num_threads,
        )

    writer.notify_finished()


def invert_unw_network(
    unw_file_list: Sequence[PathOrStr],
    reference: ReferencePoint,
    output_dir: PathOrStr,
    conncomp_file_list: Sequence[PathOrStr] | None = None,
    cor_file_list: Sequence[PathOrStr] | None = None,
    cor_threshold: float = 0.0,
    n_cor_looks: int = 1,
    ifg_date_pairs: Sequence[Sequence[DateOrDatetime]] | None = None,
    wavelength: float | None = None,
    method: InversionMethod = InversionMethod.L2,
    block_shape: tuple[int, int] = (256, 256),
    num_threads: int = 4,
) -> tuple[list[Path], list[Path]]:
    """Perform pixel-wise inversion of unwrapped network to get phase per date.

    Parameters
    ----------
    unw_file_list : Sequence[PathOrStr]
        List of unwrapped phase files.
    reference : ReferencePoint
        The reference point to use for the inversion.
        The data vector from `unw_file_list` at this point will be subtracted
        from all other points when solving.
    output_dir : PathOrStr
        The directory to save the output files
    conncomp_file_list : Sequence[PathOrStr], optional
        Sequence connected component files, one per file in `unwrapped_paths`.
        Used to ignore interferogram pixels whose connected component label is zero.
    cor_file_list : Sequence[PathOrStr], optional
        List of correlation files to use for weighting the inversion.
        Cannot be used if `conncomp_file_list` is passed.
    cor_threshold : float, optional
        The correlation threshold to use for weighting the inversion
        Default is 0.0
    n_cor_looks : int, optional
        The number of looks used to form the input correlation data, used
        to convert correlation to phase variance.
        Default is 1.
    ifg_date_pairs : Sequence[Sequence[DateOrDatetime]], optional
        List of date pairs to use for the inversion. If not provided, will be
        parsed from filenames in `unw_file_list`.
    method : str, choices = "L1", "L2"
        Inversion method to use when solving Ax = b.
        Default is L2, which uses least squares to solve Ax = b (faster).
        "L1" minimizes |Ax - b|_1 at each pixel.
    wavelength : float, optional
        The wavelength of the radar signal, in meters.
        If provided, the output rasters are in meters.
        If not provided, the outputs are in radians.
    block_shape : tuple[int, int], optional
        The shape of the blocks to process in parallel.
        Default is (256, 256).
    num_threads : int
        The parallel blocks to process at once.
        Default is 4.

    Returns
    -------
    out_paths : list[Path]
        List of the output files created by the inversion.
    residual_paths : list[Path]
        List of the output files containing the residuals.

    """
    if ifg_date_pairs is None:
        ifg_date_pairs = [get_dates(f)[:2] for f in unw_file_list]

    try:
        # Ensure it's a list of pairs
        ifg_tuples = [(ref, sec) for (ref, sec) in ifg_date_pairs]
    except ValueError as e:
        raise ValueError(
            "Each item in `ifg_date_pairs` must be a sequence of length 2"
        ) from e

    # Make the names of the output files from the SAR dates to solve for
    sar_dates = sorted(set(flatten(ifg_tuples)))
    ref_date = sar_dates[0]
    suffix = ".tif"
    # Create the `n_sar_dates - 1` output files (skipping the 0 reference raster)
    out_paths = [
        Path(output_dir) / f"{format_dates(ref_date, d)}{suffix}" for d in sar_dates[1:]
    ]
    out_residuals_paths = [
        Path(output_dir) / f"residuals_{format_dates(ref_date, d)}{suffix}"
        for d in sar_dates[1:]
    ]
    if all(p.exists() for p in out_paths):
        logger.info("All output files already exist, skipping inversion")
        return out_paths, out_residuals_paths

    summed_residuals_path = Path(output_dir) / "unw_inversion_residuals.tif"
    logger.info(f"Inverting network using {method.upper()}-norm minimization")

    A = get_incidence_matrix(ifg_pairs=ifg_tuples, sar_idxs=sar_dates)

    unw_nodataval = io.get_raster_nodata(unw_file_list[0])
    out_vrt_name = Path(output_dir) / "unw_network.vrt"
    unw_reader = io.VRTStack(
        file_list=unw_file_list,
        outfile=out_vrt_name,
        skip_size_check=True,
        read_masked=True,
    )
    cor_vrt_name = Path(output_dir) / "cor_network.vrt"
    conncomp_vrt_name = Path(output_dir) / "conncomp_network.vrt"

    if conncomp_file_list is not None and method == "L2":
        conncomp_reader = io.VRTStack(
            file_list=conncomp_file_list,
            outfile=conncomp_vrt_name,
            skip_size_check=True,
            read_masked=True,
        )
        readers = [unw_reader, conncomp_reader]
        logger.info("Masking unw pixels during inversion using connected components.")
    elif cor_file_list is not None and method == "L2":
        cor_reader = io.VRTStack(
            file_list=cor_file_list, outfile=cor_vrt_name, skip_size_check=True
        )
        readers = [unw_reader, cor_reader]
        logger.info("Using correlation to weight unw inversion")
    else:
        readers = [unw_reader]
        logger.info("Using unweighted unw inversion")

    # Get the reference point data
    ref_row, ref_col = reference
    ref_data = unw_reader[:, ref_row, ref_col].reshape(-1, 1, 1)
    if ref_data.mask.sum() > 0:
        logger.warning(f"Masked data found at {ref_row}, {ref_col}.")
        logger.warning("Zeroing out reference pixel. Results may be wrong.")
        ref_data = 0 * ref_data.data

    if wavelength is not None:
        # Positive values are motion towards the radar
        constant = -1 * (wavelength / (4 * np.pi))
        units = "meters"
    else:
        constant = -1
        units = "radians"

    def read_and_solve(
        readers: Sequence[io.StackReader], rows: slice, cols: slice
    ) -> tuple[slice, slice, np.ndarray]:
        unw_reader = readers[0]
        stack = unw_reader[:, rows, cols]
        masked_pixel_sum: NDArray[np.bool_] = stack.mask.sum(axis=0)
        # Mask the output if any inputs are missing
        masked_pixels = masked_pixel_sum > 0
        # Setup the (optional) second reader: either conncomps, or correlation
        if len(readers) == 2 and method == "L2":
            if conncomp_file_list is not None:
                # Use censored least squares based on the conncomp labels
                missing_data_flags = readers[1][:, rows, cols].filled(0) != 0
                weights = None
            else:
                # Weight the inversion by correlation-derived variance
                cor = readers[1][:, rows, cols]
                weights = correlation_to_variance(cor, n_cor_looks)
                weights[cor < cor_threshold] = 0
                missing_data_flags = None
        else:
            weights = missing_data_flags = None

        # subtract the reference, convert to numpy
        stack = (stack - ref_data).filled(0)

        # TODO: do i want to write residuals too? Do i need
        # to have multiple writers then, or a StackWriter?
        if method.upper() == "L1":
            phases, residual_sum = invert_stack_l1(A, stack)
        else:
            phases, residual_sum = invert_stack(A, stack, weights, missing_data_flags)

        # Compute the full residuals, then sum them per date
        # residuals_per_date = np.asarray(_get_residuals_per_date(A, stack, phases))
        residuals_per_date = _get_residuals_per_date(A, stack, phases)

        # Convert to meters, with LOS convention:
        out_displacement = constant * np.asarray(phases)
        # Set the masked pixels to be nodata in the output, and in the residuals
        out_displacement[:, masked_pixels] = unw_nodataval
        residuals_per_date = np.asarray(
            residuals_per_date.at[:, masked_pixels].set(np.nan)
        )
        residual_sum = np.where(masked_pixels, np.nan, np.asarray(residual_sum))

        return (
            np.vstack([out_displacement, residuals_per_date, residual_sum[np.newaxis]]),
            rows,
            cols,
        )

    # Combined writer for all output files
    writer = io.BackgroundStackWriter(
        [*out_paths, *out_residuals_paths, summed_residuals_path],
        like_filename=unw_file_list[0],
        # Using np.nan for the residuals, since it's not a valid phase
        nodata=np.nan,
    )

    io.process_blocks(
        readers=readers,
        writer=writer,
        func=read_and_solve,
        block_shape=block_shape,
        num_threads=num_threads,
    )
    writer.notify_finished()
    # Set the nodata for the outputs back to `unw_nodataval`
    for p in out_paths:
        if unw_nodataval is not None:
            io.set_raster_nodata(p, unw_nodataval)

        io.set_raster_units(p, units)

    # Residuals are always radians
    for p in out_residuals_paths:
        io.set_raster_units(p, "radians")
    io.set_raster_units(summed_residuals_path, "radians")

    logger.info("Completed invert_unw_network")
    return out_paths, out_residuals_paths


@jit
def _get_residuals_per_date(
    A: ArrayLike, x_stack: ArrayLike, b_stack: ArrayLike
) -> Array:
    """Sum the time series inversion residuals per date.

    Parameters
    ----------
    A : ArrayLike
        The matrix A in the equation Ax = b.
    x_stack : ArrayLike
        The 3D stack of solved phases from the inversion.
        Shape is (n_dates, n_rows, n_cols)
    b_stack : ArrayLike
        The 3D input stack of data.
        shape is (n_ifgs, n_rows, n_cols)

    Returns
    -------
    Array
        Residuals per date, shape=(n_dates, n_rows, n_cols)

    """
    resids_all = jnp.abs(
        A @ b_stack.reshape(A.shape[1], -1) - x_stack.reshape(A.shape[0], -1)
    )
    # Running abs(A).T @ b sums only the residuals involved in each date
    return (jnp.abs(A).T @ resids_all).reshape(b_stack.shape)


def correlation_to_variance(correlation: ArrayLike, nlooks: int) -> Array:
    r"""Convert interferometric correlation to phase variance.

    Uses the Cramer-Rao Lower Bound (CRLB) formula from Rodriguez, 1992 [1]_ to
    get the phase variance, \sigma_{\phi}^2:

    \[
        \sigma_{\phi}^{2} = \frac{1}{2N_{L}} \frac{1 - \gamma^{2}}{\gamma^{2}}
    \]

    where \gamma is the correlation and N_L is the effective number of looks.

    References
    ----------
    .. [1] Rodriguez, E., and J. M. Martin. "Theory and design of interferometric
    synthetic aperture radars." IEEE Proceedings of Radar and Signal Processing.
    Vol. 139. No. 2. IET Digital Library, 1992.

    """
    return (1 - correlation**2) / (2 * nlooks * correlation**2 + 1e-6)


def select_reference_point(
    *,
    quality_file: PathOrStr,
    output_dir: Path,
    candidate_threshold: float = 0.95,
    ccl_file_list: Sequence[PathOrStr] | None = None,
    block_shape: tuple[int, int] = (256, 256),
    num_threads: int = 4,
) -> ReferencePoint:
    """Automatically select a reference point for a stack of unwrapped interferograms.

    Uses the quality file and (optionally) connected component labels.
    The point is selected which

    1. (optionally) is within intersection of all nonzero connected component labels
    2. Has value in `quality_file` above the threshold `candidate_threshold`

    Among all points which meet this, the centroid selected using the function
    `scipy.ndimage.center_of_mass`.

    Parameters
    ----------
    quality_file: PathOrStr
        A file with the same size as each raster in `ccl_file_list` containing a quality
        metric, such as temporal coherence.
    output_dir: Path
        Path to store the computed "conncomp_intersection.tif" raster
    candidate_threshold: float
        The threshold for the quality metric function to be considered a candidate
        reference point pixel.
        Only pixels with values in `quality_file` greater than `candidate_threshold` are
        considered a candidate.
        Default = 0.95
    ccl_file_list : Sequence[PathOrStr]
        List of connected component label phase files.
    block_shape : tuple[int, int]
        Size of blocks to read from while processing `ccl_file_list`
        Default = (256, 256)
    num_threads: int
        Number of parallel blocks to process.
        Default = 5

    Returns
    -------
    ReferencePoint
        The select (row, col) as a namedtuple

    Raises
    ------
    ReferencePointError
        Raised if no valid region is found in the intersection of the connected
        component label files

    """
    output_file = output_dir / "reference_point.txt"
    if output_file.exists():
        ref_point = _read_reference_point(output_file=output_file)
        logger.info(f"Read {ref_point!r} from existing {output_file}")
        return ref_point

    logger.info("Selecting reference point")
    quality_file_values = io.load_gdal(quality_file, masked=True)

    # Start with all points as valid candidates
    isin_largest_conncomp = np.ones(quality_file_values.shape, dtype=bool)
    if ccl_file_list:
        try:
            isin_largest_conncomp = _get_largest_conncomp_mask(
                ccl_file_list=ccl_file_list,
                output_dir=output_dir,
                block_shape=block_shape,
                num_threads=num_threads,
            )
        except ReferencePointError:
            msg = "Unable to find a connected component intersection."
            msg += f"Proceeding using only {quality_file = }"
            logger.warning(msg, exc_info=True)

    # Find pixels meeting the threshold criteria
    is_candidate = quality_file_values > candidate_threshold

    # Restrict candidates to the largest connected component region
    is_candidate &= isin_largest_conncomp

    # Find connected regions within candidate pixels
    labeled, n_objects = ndimage.label(is_candidate, structure=np.ones((3, 3)))

    if n_objects == 0:
        # If no candidates meet threshold, pick best available point
        logger.warning(
            f"No pixels above threshold={candidate_threshold}. Choosing best among"
            " available."
        )
        ref_row, ref_col = argmax_index(quality_file_values)
    else:
        # Find the largest region of connected candidate pixels
        label_counts = np.bincount(labeled.ravel())
        label_counts[0] = 0  # ignore background
        largest_label = label_counts.argmax()
        largest_component = labeled == largest_label

        # Select point closest to center of largest region
        row_c, col_c = ndimage.center_of_mass(largest_component)
        rows, cols = np.nonzero(largest_component)
        dist_sq = (rows - row_c) ** 2 + (cols - col_c) ** 2
        i_min = dist_sq.argmin()
        ref_row, ref_col = rows[i_min], cols[i_min]

    # Cast to `int` to avoid having `np.int64` types
    ref_point = ReferencePoint(int(ref_row), int(ref_col))
    logger.info(f"Saving {ref_point!r} to {output_file}")
    _write_reference_point(output_file=output_file, ref_point=ref_point)
    return ref_point


def _write_reference_point(output_file: Path, ref_point: ReferencePoint) -> None:
    output_file.write_text(",".join(list(map(str, ref_point))))


def _read_reference_point(output_file: Path):
    return ReferencePoint(*[int(n) for n in output_file.read_text().split(",")])


def _get_largest_conncomp_mask(
    output_dir: Path,
    ccl_file_list: Sequence[PathOrStr] | None = None,
    block_shape: tuple[int, int] = (256, 256),
    num_threads: int = 4,
) -> np.ndarray:
    def intersect_conncomp(arr: np.ma.MaskedArray, axis: int) -> np.ndarray:
        # Track where input is nodata
        any_masked = np.any(arr.mask, axis=axis)
        # Get the logical AND of all nonzero conncomp labels
        fillval = arr.fill_value
        is_valid_conncomp = arr.filled(0) > 0
        all_are_valid = np.all(is_valid_conncomp, axis=axis).astype(arr.dtype)
        # Reset nodata
        all_are_valid[any_masked] = fillval
        return all_are_valid

    conncomp_intersection_file = Path(output_dir) / "conncomp_intersection.tif"
    if ccl_file_list and not conncomp_intersection_file.exists():
        logger.info("Creating intersection of connected components")
        create_temporal_average(
            file_list=ccl_file_list,
            output_file=conncomp_intersection_file,
            block_shape=block_shape,
            num_threads=num_threads,
            average_func=intersect_conncomp,
            read_masked=True,
        )

    conncomp_intersection = io.load_gdal(conncomp_intersection_file, masked=True)

    # Find the largest conncomp region in the intersection
    label, n_labels = ndimage.label(
        conncomp_intersection.filled(0), structure=np.ones((3, 3))
    )
    if n_labels == 0:
        raise ReferencePointError(
            "Connected components intersection left no valid regions"
        )
    logger.info("Found %d connected components in intersection", n_labels)

    # Make a mask of the largest conncomp:
    # Find the label with the most pixels using bincount
    label_counts = np.bincount(label.ravel())
    # (ignore the 0 label)
    largest_idx = np.argmax(label_counts[1:]) + 1
    # Create a mask of pixels with this label
    isin_largest_conncomp = label == largest_idx
    return isin_largest_conncomp


@jax.jit
def _shrinkage(a: jnp.ndarray, kappa: float) -> jnp.ndarray:
    """Apply the shrinkage operator element-wise."""
    return jnp.maximum(0, a - kappa) - jnp.maximum(0, -a - kappa)


@jax.jit
def least_absolute_deviations(
    A: jnp.ndarray,
    b: jnp.ndarray,
    R: jnp.ndarray,
    rho: float = 0.4,
    alpha: float = 1.0,
    max_iter: int = 20,
) -> jnp.ndarray:
    """Solve Least Absolute Deviations (LAD) via ADMM.

    Solves the following problem via ADMM:

        minimize ||Ax - b||_1

    See [@Boyd2010DistributedOptimizationStatistical] for more on ADMM.

    Parameters
    ----------
    A : jnp.ndarray
        The matrix A in the equation Ax = b.
        Shape is (M, N)
    b : jnp.ndarray
        The vector b in the equation Ax = b.
        Shape is (M,)
    R : jnp.ndarray
        Precomputed lower-triangular Cholesky factor of A^T A.
        Shape is (N, N)
    rho : float, optional
        The augmented Lagrangian parameter
        By default 0.4.
    alpha : float, optional
        The over-relaxation parameter (typical values are between 1.0 and 1.8)
        By default 1.0.
    max_iter : int, optional
        The maximum number of iterations, by default 15.

    Returns
    -------
    x_solution : jnp.ndarray
        The solution vector x of shape (N, )
    residual : jnp.ndarray, scalar
        The objective residual `sum(abs(b - A @ x_solution))`


    Notes
    -----
    The implementation is based on the [MATLAB implementation of LAD here](https://web.stanford.edu/~boyd/papers/admm/least_abs_deviations/lad.html).
    One caveat is that there are a fixed number of iterations used for this
    problem here. Inverting interferogram network have a very similar structure
    each time, and the results converge quickly relative to other large-scale
    LAD problems which ADMM can solve.

    """
    m, n = A.shape
    x0 = jnp.zeros(n)
    z0 = jnp.zeros(m)
    u0 = jnp.zeros(m)

    def body_fun(_i, state):
        x, z, z_old, u = state
        # x-update
        q = A.T @ (b + z - u)
        x = jax.scipy.linalg.cho_solve((R, True), q)

        # z-update with relaxation
        Ax_hat = alpha * (A @ x) + (1 - alpha) * (z_old + b)
        z_new = _shrinkage(Ax_hat - b + u, 1 / rho)

        # u-update
        u = u + Ax_hat - z_new - b

        return x, z_new, z, u

    x_final, _, _, _ = lax.fori_loop(0, max_iter, body_fun, (x0, z0, z0, u0))
    residual = jnp.sum(jnp.abs(b - A @ x_final))
    return x_final, residual


@jit
def invert_stack_l1(A: ArrayLike, dphi: ArrayLike) -> tuple[Array, Array]:
    R = jax.scipy.linalg.cholesky(A.T @ A, lower=True)

    # vectorize the solve function to work on 2D and 3D arrays
    # We are not vectorizing over the A matrix, only the dphi vector
    # Solve 2d shapes: (nrows, n_ifgs) -> (nrows, n_sar_dates)
    invert_2d = vmap(
        least_absolute_deviations, in_axes=(None, 1, None), out_axes=(1, 0)
    )
    # Solve 3d shapes: (nrows, ncols, n_ifgs) -> (nrows, ncols, n_sar_dates)
    invert_3d = vmap(invert_2d, in_axes=(None, 2, None), out_axes=(2, 1))
    phase, residuals = invert_3d(A, dphi, R)
    # Reshape the residuals to be 2D
    # residuals = jnp.sum(residual_vecs, axis=0)

    return phase, residuals


def create_nonzero_conncomp_counts(
    conncomp_file_list: Sequence[PathOrStr],
    output_dir: PathOrStr,
    ifg_date_pairs: Sequence[Sequence[DateOrDatetime]] | None = None,
    block_shape: tuple[int, int] = (256, 256),
    num_threads: int = 4,
) -> list[Path]:
    """Count the number of valid interferograms per date.

    Parameters
    ----------
    conncomp_file_list : Sequence[PathOrStr]
        List of connected component files
    output_dir : PathOrStr
        The directory to save the output files
    ifg_date_pairs : Sequence[Sequence[DateOrDatetime]], optional
        List of date pairs corresponding to the interferograms.
        If not provided, will be parsed from filenames.
    block_shape : tuple[int, int], optional
        The shape of the blocks to process in parallel.
    num_threads : int
        The number of parallel blocks to process at once.

    Returns
    -------
    out_paths : list[Path]
        List of output files, one per unique date

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if ifg_date_pairs is None:
        ifg_date_pairs = [get_dates(str(f))[:2] for f in conncomp_file_list]
    try:
        # Ensure it's a list of pairs
        ifg_tuples = [(ref, sec) for (ref, sec) in ifg_date_pairs]
    except ValueError as e:
        raise ValueError(
            "Each item in `ifg_date_pairs` must be a sequence of length 2"
        ) from e

    # Get unique dates and create the counting matrix
    sar_dates: list[DateOrDatetime] = sorted(set(flatten(ifg_date_pairs)))

    date_counting_matrix = np.abs(
        get_incidence_matrix(ifg_tuples, sar_dates, delete_first_date_column=False)
    )

    # Create output paths for each date
    suffix = "_valid_count.tif"
    out_paths = [output_dir / f"{d.strftime('%Y%m%d')}{suffix}" for d in sar_dates]

    if all(p.exists() for p in out_paths):
        logger.info("All output files exist, skipping counting")
        return out_paths

    logger.info("Counting valid interferograms per date")

    # Create VRT stack for reading
    vrt_name = Path(output_dir) / "conncomp_network.vrt"
    conncomp_reader = io.VRTStack(
        file_list=conncomp_file_list,
        outfile=vrt_name,
        skip_size_check=True,
        read_masked=True,
    )

    def count_by_date(
        readers: Sequence[io.StackReader], rows: slice, cols: slice
    ) -> tuple[np.ndarray, slice, slice]:
        """Process each block by counting valid interferograms per date."""
        stack = readers[0][:, rows, cols]
        valid_mask = stack.filled(0) != 0  # Shape: (n_ifgs, block_rows, block_cols)

        # Use the counting matrix to map from interferograms to dates
        # For each pixel, multiply the valid_mask to get counts per date
        # Reshape valid_mask to (n_ifgs, -1) to handle all pixels at once
        valid_flat = valid_mask.reshape(valid_mask.shape[0], -1)
        # Matrix multiply to get counts per date
        # (date_counting_matrix.T) is shape (n_sar_dates, n_ifgs), and each row
        # has a number of 1s equal to the nonzero conncomps for that date.
        date_count_cols = date_counting_matrix.T @ valid_flat
        date_counts = date_count_cols.reshape(-1, *valid_mask.shape[1:])

        return date_counts, rows, cols

    # Setup writer for all output files
    writer = io.BackgroundStackWriter(
        out_paths, like_filename=conncomp_file_list[0], dtype=np.uint16, units="count"
    )

    # Process the blocks
    io.process_blocks(
        readers=[conncomp_reader],
        writer=writer,
        func=count_by_date,
        block_shape=block_shape,
        num_threads=num_threads,
    )
    writer.notify_finished()

    logger.info("Completed counting valid interferograms per date")
    return out_paths
