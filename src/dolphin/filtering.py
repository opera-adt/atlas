from __future__ import annotations

import multiprocessing as mp
import tempfile
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from osgeo_utils.gdal_fillnodata import gdal_fillnodata
from scipy import fft, ndimage
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from dolphin import io
from dolphin._overviews import Resampling, create_image_overviews
from dolphin.utils import DummyProcessPoolExecutor


def filter_long_wavelength(
    unwrapped_phase: ArrayLike,
    bad_pixel_mask: ArrayLike,
    wavelength_cutoff: float = 50 * 1e3,
    pixel_spacing: float = 30,
    workers: int = 1,
    fill_value: float | None = None,
    scratch_dir: Path | None = None,
) -> np.ndarray:
    """Filter out signals with spatial wavelength longer than a threshold.

    Parameters
    ----------
    unwrapped_phase : ArrayLike
        Unwrapped interferogram phase to filter.
    bad_pixel_mask : ArrayLike
        Boolean array with same shape as `unwrapped_phase` where `True` indicates a
        pixel to ignore during ramp fitting
    wavelength_cutoff : float
        Spatial wavelength threshold to filter the unwrapped phase.
        Signals with wavelength longer than 'wavelength_cutoff' are filtered out.
        The default is 50*1e3 (m).
    pixel_spacing : float
        Pixel spatial spacing. Assume same spacing for x, y axes.
        The default is 30 (m).
    workers : int
        Number of `fft` workers to use for `scipy.fft.fft2`.
        Default is 1.
    fill_value : float, optional
        Value to place in output pixels which were masked.
        If `None`, masked pixels are filled with the ramp value fitted
        before filtering to suppress outliers.
    scratch_dir : Path, optional
        Directory to use for temporary files. If not provided, uses system default
        for Python's tempfile module.

    Returns
    -------
    filtered_ifg : 2D complex array
        filtered interferogram that does not contain signals with spatial wavelength
        longer than a threshold.

    Raises
    ------
    ValueError
        If wavelength_cutoff too large for image size/pixel spacing.

    """
    from dolphin import io

    # Find the filter `sigma` which gives the correct cutoff in meters
    sigma = _compute_filter_sigma(wavelength_cutoff, pixel_spacing, cutoff_value=0.5)
    rows, cols = unwrapped_phase.shape

    if sigma > rows or sigma > cols:
        msg = f"{wavelength_cutoff = } too large for image."
        msg += f"Shape = {(rows, cols)}, and {pixel_spacing = }"
        raise ValueError(msg)

    # Work on a copy of the displacement field to avoid modifying the input
    displacement = np.nan_to_num(unwrapped_phase)

    # Take either nan or 0 pixels in `unwrapped_phase` to be nodata
    nodata_mask = displacement == 0
    in_bounds_pixels = np.logical_not(nodata_mask)

    # Convert bad pixels to NaN for GDAL fillnodata
    displacement[bad_pixel_mask] = np.nan

    # Calculate the number of pixels for max_distance based on wavelength
    max_distance_pixels = int((wavelength_cutoff / 2) / pixel_spacing)

    # Create temporary files for GDAL processing
    temp_src = Path("temp_src.tif")
    temp_dst = Path("temp_filled.tif")

    scratch_dir = Path(scratch_dir) if scratch_dir is not None else None

    # Create temporary directory in the specified location or system default
    with tempfile.TemporaryDirectory(dir=scratch_dir) as temp_dir:
        tmp_path = Path(temp_dir)
        # Create paths for temporary files in the temp directory
        temp_src = tmp_path / "src.tif"
        temp_dst = tmp_path / "filled.tif"

        # Save the array to a temporary GeoTIFF
        io.write_arr(
            arr=displacement,
            nodata=np.nan,
            output_name=temp_src,
            shape=(rows, cols),
            dtype=displacement.dtype,
        )

        # Fill nodata using GDAL
        gdal_fillnodata(
            src_filename=str(temp_src),
            dst_filename=str(temp_dst),
            max_distance=max_distance_pixels,
            smoothing_iterations=0,
            interpolation="nearest",
            quiet=True,
        )

        filled_data = io.load_gdal(temp_dst, masked=True).filled(0)

    # Apply Gaussian filter
    lowpass_filtered = fft.fft2(filled_data, workers=workers)
    lowpass_filtered = ndimage.fourier_gaussian(lowpass_filtered, sigma=sigma)
    # Make sure to only take the real part (ifft returns complex)
    lowpass_filtered = fft.ifft2(lowpass_filtered, workers=workers).real.astype(
        unwrapped_phase.dtype
    )

    filtered_ifg = filled_data - lowpass_filtered * in_bounds_pixels
    if fill_value is not None:
        good_pixel_mask = np.logical_not(bad_pixel_mask)
        total_valid_mask = in_bounds_pixels & good_pixel_mask
        return np.where(total_valid_mask, filtered_ifg, fill_value)
    else:
        return filtered_ifg


def _compute_filter_sigma(
    wavelength_cutoff: float, pixel_spacing: float, cutoff_value: float = 0.5
) -> float:
    sigma_f = 1 / wavelength_cutoff / np.sqrt(np.log(1 / cutoff_value))
    sigma_x = 1 / np.pi / 2 / sigma_f
    sigma = sigma_x / pixel_spacing
    return sigma


def filter_rasters(
    unw_filenames: list[Path],
    cor_filenames: list[Path] | None = None,
    conncomp_filenames: list[Path] | None = None,
    temporal_coherence_filename: Path | None = None,
    output_dir: Path | None = None,
    wavelength_cutoff: float = 50_000,
    fill_value: float | None = None,
    correlation_cutoff: float = 0.5,
    temporal_coherence_cutoff: float = 0.5,
    max_workers: int = 4,
) -> list[Path]:
    """Filter a list of unwrapped interferogram files using a long-wavelength filter.

    Remove long-wavelength components from each unwrapped interferogram.
    It can optionally use temporal coherence, correlation, and connected component
    information for masking.

    Parameters
    ----------
    unw_filenames : list[Path]
        List of paths to unwrapped interferogram files to be filtered.
    cor_filenames : list[Path] | None
        List of paths to correlation files
        Passing None skips filtering on correlation.
    conncomp_filenames : list[Path] | None
        List of paths to connected component files, filters any 0 labeled pixels.
        Passing None skips filtering on connected component labels.
    temporal_coherence_filename : Path | None
        Path to the temporal coherence file for masking.
        Passing None skips filtering on temporal coherence.
    wavelength_cutoff : float, optional
        Spatial wavelength cutoff (in meters) for the filter. Default is 50,000 meters.
    correlation_cutoff : float, optional
        Threshold of correlation (if passing `cor_filenames`) to use to ignore pixels
        during filtering.
    temporal_coherence_cutoff : float, optional
        Threshold of temporal_coherence to use to ignore pixels during filtering.
        Default is 0.5.
    fill_value : float, optional
        Value to place in output pixels which were masked.
        If `None`, masked pixels are filled with the ramp value fitted
        before filtering to suppress outliers.
    output_dir : Path | None, optional
        Directory to save the filtered results.
        If None, saves in the same location as inputs with .filt.tif extension.
    max_workers : int, optional
        Number of parallel images to process. Default is 4.

    Returns
    -------
    list[Path]
        Output filtered rasters.

    """
    bad_pixel_mask = np.zeros(io.get_raster_shape(unw_filenames[0])[-2:], dtype="bool")
    if temporal_coherence_filename:
        bad_pixel_mask = bad_pixel_mask | (
            io.load_gdal(temporal_coherence_filename) < temporal_coherence_cutoff
        )

    if output_dir is None:
        assert unw_filenames
        output_dir = unw_filenames[0].parent
    output_dir.mkdir(exist_ok=True)

    num_parallel = min(max_workers, len(unw_filenames))
    Executor = ProcessPoolExecutor if num_parallel > 1 else DummyProcessPoolExecutor
    ctx = mp.get_context("spawn")
    tqdm.set_lock(ctx.RLock())

    with Executor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=tqdm.set_lock,
        initargs=(tqdm.get_lock(),),
    ) as exc:
        return list(
            exc.map(
                _filter_and_save,
                unw_filenames,
                cor_filenames or repeat(None),
                conncomp_filenames or repeat(None),
                repeat(output_dir),
                repeat(wavelength_cutoff),
                repeat(bad_pixel_mask),
                repeat(correlation_cutoff),
                repeat(fill_value),
            )
        )


def _filter_and_save(
    unw_filename: Path,
    cor_path: Path | None,
    conncomp_path: Path | None,
    output_dir: Path,
    wavelength_cutoff: float,
    bad_pixel_mask: NDArray[np.bool_],
    correlation_cutoff: float = 0.5,
    fill_value: float | None = None,
) -> Path:
    """Filter one interferogram (wrapper for multiprocessing)."""
    # Average for the pixel spacing for filtering
    _, x_res, _, _, _, y_res = io.get_raster_gt(unw_filename)
    pixel_spacing = (abs(x_res) + abs(y_res)) / 2

    if cor_path is not None:
        bad_pixel_mask |= io.load_gdal(cor_path) < correlation_cutoff
    if conncomp_path is not None:
        bad_pixel_mask |= io.load_gdal(conncomp_path, masked=True).astype(bool) == 0

    unw = io.load_gdal(unw_filename)
    filt_arr = filter_long_wavelength(
        unwrapped_phase=unw,
        wavelength_cutoff=wavelength_cutoff,
        bad_pixel_mask=bad_pixel_mask,
        pixel_spacing=pixel_spacing,
        workers=1,
        fill_value=fill_value,
    )
    io.round_mantissa(filt_arr, keep_bits=9)
    output_name = output_dir / Path(unw_filename).with_suffix(".filt.tif").name
    io.write_arr(arr=filt_arr, like_filename=unw_filename, output_name=output_name)

    create_image_overviews(output_name, resampling=Resampling.AVERAGE)

    return output_name


def gaussian_filter_nan(
    image: ArrayLike, sigma: float | Sequence[float], mode="constant", **kwargs
) -> np.ndarray:
    """Apply a gaussian filter to an image with NaNs (avoiding all nans).

    The scipy.ndimage `gaussian_filter` will make the output all NaNs if
    any of the pixels in the input that touches the kernel is NaN

    Source:
    https://stackoverflow.com/a/36307291

    Parameters
    ----------
    image : ndarray
        Image with nans to filter
    sigma : float
        Size of filter kernel. passed into `gaussian_filter`
    mode : str, default = "constant"
        Boundary mode for `[scipy.ndimage.gaussian_filter][]`
    **kwargs : Any
        Passed into `[scipy.ndimage.gaussian_filter][]`

    Returns
    -------
    ndarray
        Filtered version of `image`.

    """
    if np.sum(np.isnan(image)) == 0:
        return gaussian_filter(image, sigma=sigma, mode=mode, **kwargs)

    V = image.copy()
    nan_idxs = np.isnan(image)
    V[nan_idxs] = 0
    V_filt = gaussian_filter(V, sigma, **kwargs)

    W = np.ones(image.shape)
    W[nan_idxs] = 0
    W_filt = gaussian_filter(W, sigma, **kwargs)

    return V_filt / W_filt
