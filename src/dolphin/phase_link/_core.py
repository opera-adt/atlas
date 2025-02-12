from __future__ import annotations

import logging
from enum import IntEnum
from functools import partial
from typing import NamedTuple, Optional

import jax.numpy as jnp
import numpy as np
from jax import Array, jit, lax
from jax.scipy.linalg import cho_factor, cho_solve
from jax.typing import ArrayLike

from dolphin._types import HalfWindow, Strides
from dolphin.utils import take_looks

from . import covariance, metrics
from ._eigenvalues import eigh_largest_stack, eigh_smallest_stack
from ._ps_filling import fill_ps_pixels

logger = logging.getLogger("dolphin")


DEFAULT_STRIDES = Strides(1, 1)


class PhaseLinkRuntimeError(Exception):
    """Exception raised while running the MLE solver."""


class EstimatorType(IntEnum):
    """Type of estimator used for phase linking."""

    EVD = 0
    EMI = 1


class PhaseLinkOutput(NamedTuple):
    """Output of the MLE solver."""

    cpx_phase: np.ndarray
    """Estimated linked phase."""

    temp_coh: np.ndarray
    """Temporal coherence of the optimization.
    A goodness of fit parameter from 0 to 1 at each pixel.
    """

    shp_counts: np.ndarray
    """Number of neighbor pixels used in adaptive multilooking."""

    eigenvalues: np.ndarray
    """The smallest (largest) eigenvalue resulting from EMI (EVD)."""

    estimator: np.ndarray  # dtype: np.int8
    """The estimator type used for phase linking at each pixel."""

    avg_coh: np.ndarray | None = None
    """Average coherence across dates for each SLC."""


def run_phase_linking(
    slc_stack: ArrayLike,
    half_window: HalfWindow,
    strides: Strides = DEFAULT_STRIDES,
    use_evd: bool = False,
    beta: float = 0.0,
    zero_correlation_threshold: float = 0.0,
    reference_idx: int = 0,
    nodata_mask: ArrayLike | None = None,
    mask_input_ps: bool = False,
    ps_mask: ArrayLike | None = None,
    use_max_ps: bool = True,
    neighbor_arrays: ArrayLike | None = None,
    avg_mag: ArrayLike | None = None,
    use_slc_amp: bool = False,
    calc_average_coh: bool = False,
    baseline_lag: Optional[int] = None,
) -> PhaseLinkOutput:
    """Estimate the linked phase for a stack of SLCs.

    If passing a `ps_mask`, will combine the PS phases with the
    estimated DS phases.

    Parameters
    ----------
    slc_stack : ArrayLike
        The SLC stack, with shape (n_images, n_rows, n_cols)
    half_window : HalfWindow, or tuple[int, int]
        A (named) tuple of (y, x) sizes for the half window.
        The full window size is 2 * half_window + 1 for x, y.
    strides : tuple[int, int], optional
        The (y, x) strides (in pixels) to use for the sliding window.
        By default (1, 1)
    use_evd : bool, default = False
        Use eigenvalue decomposition on the covariance matrix instead of
        the EMI algorithm.
    beta : float, optional
        The regularization parameter, by default 0 (no regularization).
    zero_correlation_threshold : float, optional
        Snap correlation values in the coherence matrix below this value to 0.
        Default is 0 (no clipping).
    reference_idx : int, optional
        The index of the (non compressed) reference SLC, by default 0
    nodata_mask : ArrayLike, optional
        A mask of bad/nodata pixels to ignore when estimating the covariance.
        Pixels with `True` (or 1) are ignored, by default None
        If None, all pixels are used, by default None.
    mask_input_ps : bool
        If True, pixels labeled as PS will get set to NaN during phase linking to
        avoid summing their phase. Default of False means that the SHP algorithm
        will decide if a pixel should be included, regardless of its PS label.
    ps_mask : ArrayLike, optional
        A mask of pixels marking persistent scatterers (PS) to
        skip when multilooking.
        Pixels with `True` (or 1) are PS and will be ignored
        (combined with `nodata_mask`).
        The phase from these pixels will be inserted back
        into the final estimate directly from `slc_stack`.
    use_max_ps : bool, optional
        Whether to use the maximum PS phase for the first pixel, or average all
        PS within the look window.
        By default True.
    neighbor_arrays : ArrayLike, optional
        The neighbor arrays to use for SHP, shape = (n_rows, n_cols, *window_shape).
        If None, a rectangular window is used. By default None.
    avg_mag : ArrayLike, optional
        The average magnitude of the SLC stack, used to to find the brightest
        PS pixels to fill within each look window.
        If None, the average magnitude will be computed from `slc_stack`.
    use_slc_amp : bool, optional
        Whether to use the SLC amplitude when outputting the MLE estimate,
        or to set the SLC amplitude to 1.0. By default False.
    calc_average_coh : bool, optional, default = False
        Whether to calculate the average coherence for each SLC date.
    baseline_lag : int, optional, default=None
        lag for temporal baseline to do short temporal baseline inversion (STBAS)


    Returns
    -------
    PhaseLinkOutput:
        A Named tuple with the following fields
    linked_phase : np.ndarray[np.complex64]
        The estimated linked phase, with shape (n_images, n_rows, n_cols)
    temp_coh : np.ndarray[np.float32]
        The temporal coherence at each pixel, shape (n_rows, n_cols)
    eigenvalues : np.ndarray[np.float32]
        The smallest (largest) eigenvalue resulting from EMI (EVD).
    `avg_coh` : np.ndarray[np.float32]
        (only If `calc_average_coh` is True) the average coherence for each SLC date

    """
    _, rows, cols = slc_stack.shape
    # Common pre-processing for both CPU and GPU versions:

    # Mask nodata pixels if given
    if nodata_mask is None:
        nodata_mask = np.zeros((rows, cols), dtype=bool)
    else:
        nodata_mask = nodata_mask.astype(bool)

    # Track the PS pixels, if given, and remove them from the stack
    # This will prevent the large amplitude PS pixels from dominating
    # the covariance estimation.
    if ps_mask is None:
        ps_mask = np.zeros((rows, cols), dtype=bool)
    else:
        ps_mask = ps_mask.astype(bool)
    _raise_if_all_nan(slc_stack)

    # Make sure we also are ignoring pixels which are nans for all SLCs
    if nodata_mask.shape != (rows, cols) or ps_mask.shape != (rows, cols):
        msg = (
            f"nodata_mask.shape={nodata_mask.shape}, ps_mask.shape={ps_mask.shape},"
            f" but != SLC (rows, cols) {rows, cols}"
        )
        raise ValueError(msg)
    # for any area that has nans in the SLC stack, mark it as nodata
    nodata_mask |= np.any(np.isnan(slc_stack), axis=0)
    # Make sure the PS mask didn't have extra burst borders that are nodata here
    ps_mask[nodata_mask] = False

    # Make a copy, and set the masked pixels to np.nan
    slc_stack_masked = slc_stack.copy()
    if mask_input_ps:
        ignore_mask = np.logical_or.reduce((nodata_mask, ps_mask))
        slc_stack_masked[:, ignore_mask] = np.nan
    else:
        slc_stack_masked[:, nodata_mask] = np.nan

    cpl_out = run_cpl(
        slc_stack=slc_stack_masked,
        half_window=half_window,
        strides=strides,
        use_evd=use_evd,
        beta=beta,
        zero_correlation_threshold=zero_correlation_threshold,
        reference_idx=reference_idx,
        neighbor_arrays=neighbor_arrays,
        calc_average_coh=calc_average_coh,
        baseline_lag=baseline_lag,
    )

    # Get the smaller, looked versions of the masks
    # We zero out nodata if all pixels within the window had nodata
    mask_looked = take_looks(nodata_mask, *strides, func_type="all")

    # Convert from jax array back to np
    temp_coh = np.array(cpl_out.temp_coh)

    # Set as unit-magnitude
    cpx_phase = np.exp(1j * np.angle(cpl_out.cpx_phase))
    # Fill in the PS pixels from the original SLC stack, if it was given
    if np.any(ps_mask):
        fill_ps_pixels(
            cpx_phase,
            temp_coh,
            slc_stack,
            ps_mask,
            strides,
            avg_mag,
            reference_idx,
            use_max_ps=use_max_ps,
        )

    if use_slc_amp:
        # use the amplitude from the original SLCs
        # account for the strides when grabbing original data
        # we need to match `io.compute_out_shape` here
        slcs_decimated = decimate(slc_stack, strides)
        cpx_phase = np.exp(1j * np.angle(cpx_phase)) * np.abs(slcs_decimated)

    # Finally, ensure the nodata regions are 0
    cpx_phase[:, mask_looked] = np.nan
    temp_coh[mask_looked] = np.nan

    return PhaseLinkOutput(
        cpx_phase=cpx_phase,
        temp_coh=temp_coh,
        shp_counts=np.asarray(cpl_out.shp_counts),
        # Convert the rest to numpy for writing
        eigenvalues=np.asarray(cpl_out.eigenvalues),
        estimator=np.asarray(cpl_out.estimator),
        avg_coh=cpl_out.avg_coh,
    )


def run_cpl(
    slc_stack: np.ndarray,
    half_window: HalfWindow,
    strides: Strides,
    use_evd: bool = False,
    beta: float = 0,
    zero_correlation_threshold: float = 0.0,
    reference_idx: int = 0,
    neighbor_arrays: Optional[np.ndarray] = None,
    calc_average_coh: bool = False,
    baseline_lag: Optional[int] = None,
) -> PhaseLinkOutput:
    """Run the Combined Phase Linking (CPL) algorithm.

    Estimates a coherence matrix for each SLC pixel, then
    runs the EMI/EVD solver.

    Parameters
    ----------
    slc_stack : np.ndarray
        The SLC stack, with shape (n_slc, n_rows, n_cols)
    half_window : HalfWindow, or tuple[int, int]
        A (named) tuple of (y, x) sizes for the half window.
        The full window size is 2 * half_window + 1 for x, y.
    strides : tuple[int, int], optional
        The (y, x) strides (in pixels) to use for the sliding window.
        By default (1, 1)
    use_evd : bool, default = False
        Use eigenvalue decomposition on the covariance matrix instead of
        the EMI algorithm.
    beta : float, optional
        The regularization parameter, by default 0 (no regularization).
    zero_correlation_threshold : float, optional
        Snap correlation values in the coherence matrix below this value to 0.
        Default is 0 (no clipping).
    reference_idx : int, optional
        The index of the (non compressed) reference SLC, by default 0
    use_slc_amp : bool, optional
        Whether to use the SLC amplitude when outputting the MLE estimate,
        or to set the SLC amplitude to 1.0. By default False.
    neighbor_arrays : np.ndarray, optional
        The neighbor arrays to use for SHP, shape = (n_rows, n_cols, *window_shape).
        If None, a rectangular window is used. By default None.
    calc_average_coh : bool, default=False
        If requested, the average of each row of the covariance matrix is computed
        for the purposes of finding the best reference (highest coherence) date
    baseline_lag : int, optional, default=None
        StBAS parameter to include only nearest-N interferograms for phase linking.
        A `baseline_lag` of `n` will only include the closest `n` interferograms.
        `baseline_line` must be positive.

    Returns
    -------
    cpx_phase : Array
        Optimized SLC phase, shape same as `slc_stack` unless Strides are requested.
    temp_coh : Array
        Temporal coherence of the optimization.
        A goodness of fit parameter from 0 to 1 at each pixel.
        shape = (out_rows, out_cols)
    eigenvalues : Array
        The eigenvalues of the coherence matrices.
        If `use_evd` is True, these are the largest eigenvalues;
        Otherwise, for EMI they are the smallest.
        shape = (out_rows, out_cols)
    estimator : Array
        The estimator used at each pixel.
        0 = EVD, 1 = EMI
        shape = (out_rows, out_cols)
    avg_coh : np.ndarray | None
        The average coherence of each row of the coherence matrix,
        if requested.
        shape = (nslc, out_rows, out_cols)

    """
    C_arrays = covariance.estimate_stack_covariance(
        slc_stack,
        half_window,
        strides,
        neighbor_arrays=neighbor_arrays,
    )
    ns = slc_stack.shape[0]
    if baseline_lag:
        u_rows, u_cols = jnp.triu_indices(ns, baseline_lag)
        l_rows, l_cols = jnp.tril_indices(ns, -baseline_lag)
        C_arrays = C_arrays.at[:, :, u_rows, u_cols].set(0.0 + 0j)
        C_arrays = C_arrays.at[:, :, l_rows, l_cols].set(0.0 + 0j)

    cpx_phase, eigenvalues, estimator = process_coherence_matrices(
        C_arrays,
        use_evd=use_evd,
        beta=beta,
        zero_correlation_threshold=zero_correlation_threshold,
        reference_idx=reference_idx,
    )
    # Get the temporal coherence
    temp_coh = metrics.estimate_temp_coh(cpx_phase, C_arrays)

    # Reshape the (rows, cols, nslcs) output to be same as input stack
    cpx_phase_reshaped = jnp.moveaxis(cpx_phase, -1, 0)

    # Get the SHP counts for each pixel (if not using Rect window)
    if neighbor_arrays is None:
        shp_counts = jnp.zeros(temp_coh.shape, dtype=np.int16)
    else:
        shp_counts = jnp.sum(neighbor_arrays, axis=(-2, -1))

    if calc_average_coh:
        # If requested, average the Cov matrix at each row for reference selection
        avg_coh_per_date = jnp.abs(C_arrays).mean(axis=3)
        avg_coh = np.argmax(avg_coh_per_date, axis=2)
    else:
        avg_coh = None

    return PhaseLinkOutput(
        cpx_phase=cpx_phase_reshaped,
        temp_coh=temp_coh,
        shp_counts=shp_counts,
        eigenvalues=eigenvalues,
        estimator=estimator,
        avg_coh=avg_coh,
    )


@partial(jit, static_argnames=("use_evd", "beta", "reference_idx"))
def process_coherence_matrices(
    C_arrays,
    use_evd: bool = False,
    beta: float = 0.0,
    zero_correlation_threshold: float = 0.0,
    reference_idx: int = 0,
) -> tuple[Array, Array, Array]:
    """Estimate the linked phase for a stack of coherence matrices.

    This function is used after coherence estimation to estimate the
    optimized SLC phase.

    Parameters
    ----------
    C_arrays : ndarray, shape = (rows, cols, nslc, nslc)
        The sample coherence matrix at each pixel
        (e.g. from [dolphin.phase_link.covariance.estimate_stack_covariance][])
    use_evd : bool, default = False
        Use eigenvalue decomposition on the covariance matrix instead of
        the EMI algorithm of [@Ansari2018EfficientPhaseEstimation].
    beta : float, optional
        The regularization parameter for inverting Gamma = |C|
        The regularization is applied as (1 - beta) * Gamma + beta * I
        Default is 0 (no regularization).
    zero_correlation_threshold : float, optional
        Snap correlation values in the coherence matrix below this value to 0.
        Default is 0 (no clipping).
    reference_idx : int, optional
        The index of the reference acquisition, by default 0
        All outputs are multiplied by the conjugate of the data at this index.

    Returns
    -------
    eig_vecs : ndarray[float32], shape = (rows, cols, nslc)
        The phase resulting from the optimization at each output pixel.
        Shape is same as input slcs unless Strides > (1, 1)
    eig_vals : ndarray[float], shape = (rows, cols)
        The smallest (largest) eigenvalue as solved by EMI (EVD).
    estimator : Array
        The estimator used at each pixel.
        0 = EVD, 1 = EMI

    """
    rows, cols, n, _ = C_arrays.shape

    evd_eig_vals, evd_eig_vecs = eigh_largest_stack(C_arrays * jnp.abs(C_arrays))
    if use_evd:
        # EVD
        eig_vals, eig_vecs = evd_eig_vals, evd_eig_vecs
        estimator = jnp.zeros(eig_vals.shape, dtype=bool)
    else:
        # EMI
        # estimate the wrapped phase based on the EMI paper
        # *smallest* eigenvalue decomposition of the (|Gamma|^-1  *  C) matrix

        # Identity used for regularization and for solving
        Id = jnp.eye(n, dtype=C_arrays.dtype)
        # repeat the identity matrix for each pixel
        Id = jnp.tile(Id, (rows, cols, 1, 1))

        Gamma = jnp.abs(C_arrays)
        if beta > 0:
            # Perform regularization
            Gamma = (1 - beta) * Gamma + beta * Id
        # Assume correlation below `zero_correlation_threshold` is 0
        Gamma = jnp.where(Gamma < zero_correlation_threshold, 0, Gamma)

        # Attempt to invert Gamma
        cho, is_lower = cho_factor(Gamma)

        # Check: If it fails the cholesky factor, it's close to singular and
        # we should just fall back to EVD
        # Use the already- factored |Gamma|^-1, solving Ax = I gives the inverse
        Gamma_inv = cho_solve((cho, is_lower), Id)
        # We're looking for the lambda nearest to 1. So shift by 0.99
        # Also, use the evd vectors as iteration starting point:
        mu = 0.99
        emi_eig_vals, emi_eig_vecs = eigh_smallest_stack(Gamma_inv * C_arrays, mu)
        # From the EMI paper, normalize the eigenvectors to have norm sqrt(n)
        emi_eig_vecs = (
            jnp.sqrt(n)
            * emi_eig_vecs
            / jnp.linalg.norm(emi_eig_vecs, axis=-1, keepdims=True)
        )
        # is the output is the inverse of the eigenvectors? or inverse conj?

        # Use https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.select.html
        # Note that `if` would fail the jit tracing
        # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#cond
        inv_has_nans = jnp.any(jnp.isnan(Gamma_inv), axis=(-1, -2))

        # Must broadcast the 2D boolean array so it's the same size as the outputs
        inv_has_nans_3d = jnp.tile(inv_has_nans[:, :, None], (1, 1, n))

        # For EVD, or places where inverting |Gamma| failed: fall back to computing EVD
        eig_vecs = lax.select(
            inv_has_nans_3d,
            # Run this on True: EVD, since we failed to invert:
            evd_eig_vecs,
            # Otherwise, on False, we're fine to use EMI
            emi_eig_vecs,
        )

        eig_vals = lax.select(inv_has_nans, evd_eig_vals, emi_eig_vals)
        # Make array of ints to indicate which estimator was used for each pixel
        # 0 means EVD, 1 mean EMI
        evd_used = jnp.zeros(emi_eig_vals.shape, dtype=jnp.int8)
        emi_used = jnp.ones(emi_eig_vals.shape, dtype=jnp.int8)
        estimator = lax.select(inv_has_nans, evd_used, emi_used)

    # Now the shape of eig_vecs is (rows, cols, nslc)
    # at pixel (r, c), eig_vecs[r, c] is the largest (smallest) eigenvector if
    # we picked EVD (EMI)
    # The phase estimate on the reference day will be size (rows, cols)
    ref = eig_vecs[:, :, reference_idx]
    # Make sure each still has 3 dims, then reference all phases to `ref`
    evd_estimate = eig_vecs * jnp.exp(-1j * jnp.angle(ref[:, :, None]))

    return evd_estimate, eig_vals, estimator.astype("uint8")


def decimate(arr: ArrayLike, strides: Strides) -> Array:
    """Decimate an array by strides in the x and y directions.

    Output will match [`io.compute_out_shape`][dolphin.io.compute_out_shape]

    Parameters
    ----------
    arr : ArrayLike
        2D or 3D array to decimate.
    strides : dict[str, int]
        The strides in the x and y directions.

    Returns
    -------
    ArrayLike
        The decimated array.

    """
    ys, xs = strides
    rows, cols = arr.shape[-2:]
    start_r = ys // 2
    start_c = xs // 2
    end_r = (rows // ys) * ys + 1
    end_c = (cols // xs) * xs + 1
    return arr[..., start_r:end_r:ys, start_c:end_c:xs]


def _raise_if_all_nan(slc_stack: np.ndarray):
    """Check for all NaNs in each SLC of the stack."""
    nans = np.isnan(slc_stack)
    # Check that there are no SLCS which are all nans:
    bad_slc_idxs = np.where(np.all(nans, axis=(1, 2)))[0]
    if bad_slc_idxs.size > 0:
        msg = f"slc_stack[{bad_slc_idxs}] out of {len(slc_stack)} are all NaNs."
        raise PhaseLinkRuntimeError(msg)
