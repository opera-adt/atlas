"""Find the persistent scatterers in a stack of SLCS."""

from __future__ import annotations

import logging
import shutil
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike
from osgeo import gdal

from dolphin import io, utils
from dolphin._types import Filename
from dolphin.io import EagerLoader, StackReader, repack_raster

gdal.UseExceptions()

logger = logging.getLogger("dolphin")

NODATA_VALUES = {"ps": 255, "amp_dispersion": 0.0, "amp_mean": 0.0}

FILE_DTYPES = {"ps": np.uint8, "amp_dispersion": np.float32, "amp_mean": np.float32}
_EXTRA_COMPRESSION = {
    "keep_bits": 10,
    "predictor": 3,
}
REPACK_OPTIONS = {
    "ps": {},
    "amp_dispersion": _EXTRA_COMPRESSION,
    "amp_mean": _EXTRA_COMPRESSION,
}


def create_ps(
    *,
    reader: StackReader,
    output_file: Filename,
    output_amp_mean_file: Filename,
    output_amp_dispersion_file: Filename,
    like_filename: Filename,
    amp_dispersion_threshold: float = 0.25,
    existing_amp_mean_file: Optional[Filename] = None,
    existing_amp_dispersion_file: Optional[Filename] = None,
    nodata_mask: Optional[np.ndarray] = None,
    update_existing: bool = False,
    block_shape: tuple[int, int] = (512, 512),
    **tqdm_kwargs,
):
    """Create the amplitude dispersion, mean, and PS files.

    Parameters
    ----------
    reader : StackReader
        A dataset reader for the 3D SLC stack.
    output_file : Filename
        The output PS file (dtype: Byte)
    output_amp_dispersion_file : Filename
        The output amplitude dispersion file.
    output_amp_mean_file : Filename
        The output mean amplitude file.
    like_filename : Filename
        The filename to use for the output files' spatial reference.
    amp_dispersion_threshold : float, optional
        The threshold for the amplitude dispersion. Default is 0.25.
    existing_amp_mean_file : Optional[Filename], optional
        An existing amplitude mean file to use, by default None.
    existing_amp_dispersion_file : Optional[Filename], optional
        An existing amplitude dispersion file to use, by default None.
    nodata_mask : Optional[np.ndarray]
        If provided, skips computing PS over areas where the mask is False
        Otherwise, loads input data from everywhere and calculates.
    update_existing : bool, optional
        If providing existing amp mean/dispersion files, combine them with the
        data from the current SLC stack.
        If False, simply uses the existing files to create as PS mask.
        Default is False.
    block_shape : tuple[int, int], optional
        The 2D block size to load all bands at a time.
        Default is (512, 512)
    **tqdm_kwargs : optional
        Arguments to pass to `tqdm`, (e.g. `position=n` for n parallel bars)
        See https://tqdm.github.io/docs/tqdm/#tqdm-objects for all options.

    """
    if existing_amp_dispersion_file and existing_amp_mean_file and not update_existing:
        logger.info("Using existing amplitude dispersion file, skipping calculation.")
        # Just use what's there, copy to the expected output locations
        _use_existing_files(
            existing_amp_mean_file=existing_amp_mean_file,
            existing_amp_dispersion_file=existing_amp_dispersion_file,
            output_file=output_file,
            output_amp_mean_file=output_amp_mean_file,
            output_amp_dispersion_file=output_amp_dispersion_file,
            amp_dispersion_threshold=amp_dispersion_threshold,
        )
        return

    # Otherwise, we need to calculate the PS files from the SLC stack
    # Initialize the output files with zeros
    file_list = [output_file, output_amp_dispersion_file, output_amp_mean_file]
    for fn, dtype, nodata in zip(
        file_list, FILE_DTYPES.values(), NODATA_VALUES.values()
    ):
        io.write_arr(
            arr=None,
            like_filename=like_filename,
            output_name=fn,
            nbands=1,
            dtype=dtype,
            nodata=nodata,
        )
    # Initialize the intermediate arrays for the calculation
    magnitude = np.zeros((reader.shape[0], *block_shape), dtype=np.float32)

    writer = io.BackgroundBlockWriter()
    # Make the generator for the blocks
    block_gen = EagerLoader(reader, block_shape=block_shape, nodata_mask=nodata_mask)
    for cur_data, (rows, cols) in block_gen.iter_blocks(**tqdm_kwargs):
        cur_rows, cur_cols = cur_data.shape[-2:]

        if not (np.all(cur_data == 0) or np.all(np.isnan(cur_data))):
            magnitude_cur = np.abs(cur_data, out=magnitude[:, :cur_rows, :cur_cols])
            mean, amp_disp, ps = calc_ps_block(
                # use min_count == size of stack so that ALL need to be not Nan
                magnitude_cur,
                amp_dispersion_threshold,
                min_count=len(magnitude_cur),
            )

            # Use the UInt8 type for the PS to save.
            # For invalid pixels, set to max Byte value
            ps = ps.astype(FILE_DTYPES["ps"])
            ps[amp_disp == 0] = NODATA_VALUES["ps"]
        else:
            # Fill the block with nodata
            ps = (
                np.ones((cur_rows, cur_cols), dtype=FILE_DTYPES["ps"])
                * NODATA_VALUES["ps"]
            )
            mean = np.full(
                (cur_rows, cur_cols),
                NODATA_VALUES["amp_mean"],
                dtype=FILE_DTYPES["amp_mean"],
            )
            amp_disp = np.full(
                (cur_rows, cur_cols),
                NODATA_VALUES["amp_dispersion"],
                dtype=FILE_DTYPES["amp_dispersion"],
            )

        # Write amp dispersion and the mean blocks
        writer.queue_write(mean, output_amp_mean_file, rows.start, cols.start)
        writer.queue_write(amp_disp, output_amp_dispersion_file, rows.start, cols.start)
        writer.queue_write(ps, output_file, rows.start, cols.start)

    logger.info(f"Waiting to write {writer.num_queued} blocks of data.")
    writer.notify_finished()
    # Repack for better compression
    logger.info("Repacking PS rasters for better compression")
    for fn, opt in zip(file_list, REPACK_OPTIONS.values()):
        # Repack to a temp, then overwrite
        repack_raster(Path(fn), output_dir=None, **opt)
    logger.info("Finished writing out PS files")


def calc_ps_block(
    stack_mag: ArrayLike,
    amp_dispersion_threshold: float = 0.25,
    min_count: Optional[int] = None,
):
    r"""Calculate the amplitude dispersion for a block of data.

    The amplitude dispersion is defined as the standard deviation of a pixel's
    magnitude divided by the mean of the magnitude:

    \[
    d_a = \frac{\sigma(|Z|)}{\mu(|Z|)}
    \]

    where $Z \in \mathbb{R}^{N}$ is one pixel's complex data for $N$ SLCs.

    Parameters
    ----------
    stack_mag : ArrayLike
        The magnitude of the stack of SLCs.
    amp_dispersion_threshold : float, optional
        The threshold for the amplitude dispersion to label a pixel as a PS:
            ps = amp_disp < amp_dispersion_threshold
        Default is 0.25.
    min_count : int, optional
        The minimum number of valid pixels to calculate the mean and standard
        deviation. If the number of valid pixels is less than `min_count`,
        then the mean and standard deviation are set to 0 (and the pixel is
        not a PS). Default is 90% the number of SLCs: `int(0.9 * stack_mag.shape[0])`.

    Returns
    -------
    mean : np.ndarray
        The mean amplitude for the block.
        dtype: float32
    amp_disp : np.ndarray
        The amplitude dispersion for the block.
        dtype: float32
    ps : np.ndarray
        The persistent scatterers for the block.
        dtype: bool

    Notes
    -----
    The min_count is used to prevent the mean and standard deviation from being
    calculated for pixels that are not valid for most of the SLCs. This happens
    when the burst footprints shift around and pixels near the edge get only one or
    two acquisitions.
    Since fewer samples are used to calculate the mean and standard deviation,
    there is a higher false positive risk for these edge pixels.

    """
    if np.iscomplexobj(stack_mag):
        msg = "The input `stack_mag` must be real-valued."
        raise ValueError(msg)

    if min_count is None:
        min_count = int(0.9 * stack_mag.shape[0])

    with warnings.catch_warnings():
        # ignore the warning about nansum/nanmean of empty slice
        warnings.simplefilter("ignore", category=RuntimeWarning)

        mean = np.nanmean(stack_mag, axis=0)
        std_dev = np.nanstd(stack_mag, axis=0)
        count = np.count_nonzero(~np.isnan(stack_mag), axis=0)
        amp_disp = std_dev / mean
    # Mask out the pixels with too few valid pixels
    amp_disp[count < min_count] = np.nan
    # replace nans/infinities with 0s, which will mean nodata
    mean = np.nan_to_num(mean, nan=0, posinf=0, neginf=0, copy=False)
    amp_disp = np.nan_to_num(amp_disp, nan=0, posinf=0, neginf=0, copy=False)

    ps = amp_disp < amp_dispersion_threshold
    ps[amp_disp == 0] = False
    return mean, amp_disp, ps


def _use_existing_files(
    *,
    existing_amp_mean_file: Filename,
    existing_amp_dispersion_file: Filename,
    output_file: Filename,
    output_amp_mean_file: Filename,
    output_amp_dispersion_file: Filename,
    amp_dispersion_threshold: float,
) -> None:
    amp_disp = io.load_gdal(existing_amp_dispersion_file, masked=True)
    ps = amp_disp < amp_dispersion_threshold
    ps = ps.astype(np.uint8)
    # Set the PS nodata value to the max uint8 value
    ps[(amp_disp == 0) | amp_disp.mask] = NODATA_VALUES["ps"]
    io.write_arr(
        arr=ps,
        like_filename=existing_amp_dispersion_file,
        output_name=output_file,
        nodata=NODATA_VALUES["ps"],
    )
    # Copy the existing amp mean file/amp dispersion file
    shutil.copy(existing_amp_dispersion_file, output_amp_dispersion_file)
    shutil.copy(existing_amp_mean_file, output_amp_mean_file)


def multilook_ps_files(
    strides: dict[str, int],
    ps_mask_file: Filename,
    amp_dispersion_file: Filename,
) -> tuple[Path, Path]:
    """Create a multilooked version of the full-res PS mask/amplitude dispersion.

    Parameters
    ----------
    strides : dict[str, int]
        Decimation factor for 'x', 'y'
    ps_mask_file : Filename
        Name of input full-res uint8 PS mask file
    amp_dispersion_file : Filename
        Name of input full-res float32 amplitude dispersion file

    Returns
    -------
    output_ps_file : Path
        Multilooked PS mask file
        Will be same as `ps_mask_file`, but with "_looked" added before suffix.
    output_amp_disp_file : Path
        Multilooked amplitude dispersion file
        Similar naming scheme to `output_ps_file`

    """
    if strides == {"x": 1, "y": 1}:
        logger.info("No striding request, skipping multilook.")
        return Path(ps_mask_file), Path(amp_dispersion_file)
    full_cols, full_rows = io.get_raster_xysize(ps_mask_file)
    out_rows, out_cols = full_rows // strides["y"], full_cols // strides["x"]

    ps_suffix = Path(ps_mask_file).suffix
    ps_out_path = Path(str(ps_mask_file).replace(ps_suffix, f"_looked{ps_suffix}"))
    logger.info(f"Saving a looked PS mask to {ps_out_path}")

    if Path(ps_out_path).exists():
        logger.info(f"{ps_out_path} exists, skipping.")
    else:
        ps_mask = io.load_gdal(ps_mask_file, masked=True).astype(bool)
        ps_mask_looked = utils.take_looks(
            ps_mask, strides["y"], strides["x"], func_type="any", edge_strategy="pad"
        )
        # make sure it's the same size as the MLE result/temp_coh after padding
        ps_mask_looked = ps_mask_looked[:out_rows, :out_cols]
        ps_mask_looked = ps_mask_looked.astype("uint8").filled(NODATA_VALUES["ps"])
        io.write_arr(
            arr=ps_mask_looked,
            like_filename=ps_mask_file,
            output_name=ps_out_path,
            strides=strides,
            nodata=NODATA_VALUES["ps"],
        )

    amp_disp_suffix = Path(amp_dispersion_file).suffix
    amp_disp_out_path = Path(
        str(amp_dispersion_file).replace(amp_disp_suffix, f"_looked{amp_disp_suffix}")
    )
    if amp_disp_out_path.exists():
        logger.info(f"{amp_disp_out_path} exists, skipping.")
    else:
        amp_disp = io.load_gdal(amp_dispersion_file, masked=True)
        # We use `nanmin` assuming that the multilooked PS is using
        # the strongest PS (the one with the lowest amplitude dispersion)
        amp_disp_looked = utils.take_looks(
            amp_disp,
            strides["y"],
            strides["x"],
            func_type="nanmin",
            edge_strategy="pad",
        )
        amp_disp_looked = amp_disp_looked[:out_rows, :out_cols]
        amp_disp_looked = amp_disp_looked.filled(NODATA_VALUES["amp_dispersion"])
        io.write_arr(
            arr=amp_disp_looked,
            like_filename=amp_dispersion_file,
            output_name=amp_disp_out_path,
            strides=strides,
            nodata=NODATA_VALUES["amp_dispersion"],
        )
    return ps_out_path, amp_disp_out_path


def combine_means(means: ArrayLike, N: ArrayLike) -> np.ndarray:
    r"""Compute the combined mean from multiple `mu_i` values.

    This function calculates the weighted average of amplitudes based on the
    number of original data points (N) that went into each mean.

    Parameters
    ----------
    means : ArrayLike
        A 3D array of mean values.
        Shape: (n_images, rows, cols)
    N : np.ndarray
        A list/array of weights indicating the number of original images.
        Shape: (depth,)

    Returns
    -------
    np.ndarray
        The combined mean.
        Shape: (height, width)

    Notes
    -----
    Both input arrays are expected to have the same shape.
    The operation is performed along axis=0.

    The combined mean is calculated as

    \begin{equation}
        E[X] = \frac{\sum_i N_i\mu_i}{\sum_i N_i}
    \end{equation}

    """
    N = np.asarray(N)
    if N.shape[0] != means.shape[0]:
        raise ValueError("Size of N must match the number of images in means.")
    if N.ndim == 1:
        N = N[:, None, None]

    weighted_sum = np.sum(means * N, axis=0)
    total_N = np.sum(N, axis=0)

    return weighted_sum / total_N


def combine_amplitude_dispersions(
    dispersions: np.ndarray, means: np.ndarray, N: ArrayLike | Sequence
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute the combined amplitude dispersion from multiple groups.

    Given several ADs where difference numbers of images, N, went in,
    the function computes a weighted mean/variance to calculate the combined AD.

    Parameters
    ----------
    dispersions : np.ndarray
        A 3D array of amplitude dispersion values for each group.
        Shape: (depth, height, width)
    means : np.ndarray
        A 3D array of mean values for each group.
        Shape: (depth, height, width)
    N : np.ndarray
        An array sample sizes for each group.
        Shape: (depth, )

    Returns
    -------
    np.ndarray
        The combined amplitude dispersion.
        Shape: (height, width)
    np.ndarray
        The combined amplitude mean.
        Shape: (height, width)

    Notes
    -----
    All input arrays are expected to have the same shape.
    The operation is performed along `axis=0`.

    Let $X_i$ be the random variable for group $i$, with mean $\mu_i$ and variance
    $\sigma_i^2$, and $N_i$ be the number of samples in group $i$.

    The combined variance $\sigma^2$ uses the formula

    \begin{equation}
        \sigma^2 = E[X^2] - (E[X])^2
    \end{equation}

    where $E[X]$ is the combined mean, and $E[X^2]$ is the expected value of
    the squared random variable.

    The combined mean is calculated as:

    \begin{equation}
        E[X] = \frac{\sum_i N_i\mu_i}{\sum_i N_i}
    \end{equation}

    For $E[X^2]$, we use the property $E[X^2] = \sigma^2 + \mu^2$:

    \begin{equation}
        E[X^2] = \frac{\sum_i N_i(\sigma_i^2 + \mu_i^2)}{\sum_i N_i}
    \end{equation}

    Substituting these into the variance formula gives:

    \begin{equation}
        \sigma^2 = \frac{\sum_i N_i(\sigma_i^2 + \mu_i^2)}{\sum_i N_i} -
        \left(\frac{\sum_i N_i\mu_i}{\sum_i N_i}\right)^2
    \end{equation}

    """
    N = np.asarray(N)
    if N.ndim == 1:
        N = N[:, None, None]
    if not (means.shape == dispersions.shape):
        raise ValueError("Input arrays must have the same shape.")
    if means.shape[0] != N.shape[0]:
        raise ValueError("Size of N must match the number of groups in means.")

    combined_mean = combine_means(means, N)

    # Compute combined variance
    variances = (dispersions * means) ** 2
    total_N = np.sum(N, axis=0).squeeze()
    sum_N_var_meansq = np.sum(N * (variances + means**2), axis=0)
    combined_variance = (sum_N_var_meansq / total_N) - combined_mean**2

    return np.sqrt(combined_variance) / combined_mean, combined_mean
