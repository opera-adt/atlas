from __future__ import annotations

import logging
import multiprocessing
import shutil
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen

import numpy as np
import rasterio
from numpy.typing import NDArray
from opera_utils import get_dates
from scipy import ndimage, signal

from dolphin import io
from dolphin._types import PathOrStr
from dolphin.workflows.config import SpurtOptions

from ._constants import CONNCOMP_SUFFIX, DEFAULT_CCL_NODATA, UNW_SUFFIX
from ._post_process import interpolate_masked_gaps

logger = logging.getLogger("dolphin")

DEFAULT_OPTIONS = SpurtOptions()


def unwrap_spurt(
    ifg_filenames: Sequence[PathOrStr],
    output_path: PathOrStr,
    temporal_coherence_filename: PathOrStr,
    # cor_filenames: Sequence[PathOrStr] | None = None,
    mask_filename: PathOrStr | None = None,
    similarity_filename: PathOrStr | None = None,
    options: SpurtOptions = DEFAULT_OPTIONS,
    scratchdir: PathOrStr | None = None,
    num_retries: int = 3,
) -> tuple[list[Path], list[Path]]:
    """Perform 3D unwrapping using `spurt` via subprocess call."""
    # NOTE: we are working around spurt currently wanting "temporal_coherence.tif",
    # and a temporal coherence threshold.
    # we'll make our own mask of 0=bad, 1=good, then pass a threshold of 0.5
    temp_coh = io.load_gdal(temporal_coherence_filename, masked=True).filled(0)
    # Mark the "bad" pixels (good=1, bad=0, following the unwrapper mask convention)
    temp_coh_mask = temp_coh > options.temporal_coherence_threshold
    combined_mask = temp_coh_mask
    if similarity_filename and options.similarity_threshold:
        sim = io.load_gdal(similarity_filename, masked=True).filled(0)
        sim_mask = sim > options.similarity_threshold
        # A good pixel can have good similarity, or good temp. coherence
        combined_mask = combined_mask | sim_mask

    if mask_filename:
        nodata_mask = io.load_gdal(mask_filename).astype(bool)
        # A good pixel has to be 1 in both masks
        combined_mask = combined_mask & nodata_mask

    # We name it "temporal_coherence.tif" so spurt reads it.
    # Also make it float32 as though it were temp coh
    scratch_path = Path(scratchdir) if scratchdir else Path(output_path) / "scratch"
    scratch_path.mkdir(exist_ok=True, parents=True)
    combined_mask_filename = scratch_path / "temporal_coherence.tif"
    io.write_arr(
        arr=combined_mask.astype("float32"), output_name=combined_mask_filename
    )

    # Symlink the interferograms to the same scratch path so spurt finds everything
    # expected in the one directory
    for fn in ifg_filenames:
        new_path = scratch_path / Path(fn).name
        if not new_path.exists():
            new_path.symlink_to(fn)

    cmd = [
        "python",
        "-m",
        "spurt.workflows.emcf",
        "-i",
        str(scratch_path),
        "--log-file",
        f"{scratch_path}/spurt-unwrap.log",
        "-o",
        str(output_path),
        "--tempdir",
        str(scratch_path / "emcf_tmp"),
        "-c",
        str(0.5),  # arbitrary, since we are passing a 0/1 file anyway
    ]
    if not options.general_settings.use_tiles:
        cmd.append("--singletile")

    # Tiler Settings
    cmd.extend(
        [
            "--pts-per-tile",
            str(options.tiler_settings.target_points_per_tile),
            "--max-tiles",
            str(options.tiler_settings.max_tiles),
        ]
    )

    # Solver Settings
    cmd.extend(
        [
            "-w",
            str(options.solver_settings.t_worker_count),
            "--s-workers",
            str(options.solver_settings.s_worker_count),
            "-b",
            str(options.solver_settings.links_per_batch),
            "--t-cost-type",
            options.solver_settings.t_cost_type,
            "--t-cost-scale",
            str(int(options.solver_settings.t_cost_scale)),
            "--unwrap-parallel-tiles",
            str(options.solver_settings.num_parallel_tiles),
        ]
    )

    # Merger Settings
    cmd.extend(
        [
            "--merge-parallel-ifgs",
            str(options.merger_settings.num_parallel_ifgs),
        ]
    )

    def run_with_retry(cmd: list[str], num_retries: int = 3) -> int:
        for attempt in range(num_retries):
            process = Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True)
            assert process.stdout is not None
            for line in process.stdout:
                logger.info(line.strip())

            return_code = process.wait()
            if return_code != 0:
                logging.warning(f"spurt attempt {attempt + 1} failed")
                continue

            logging.info(f"Command succeeded on attempt {attempt + 1}")
            return return_code

        # If we've exhausted all retries
        logging.error(f"Command failed after {num_retries} attempts")
        raise RuntimeError(f"Command '{cmd}' failed after {num_retries} attempts")

    run_with_retry(cmd, num_retries=num_retries)

    # Return paths to output files
    output_path = Path(output_path)
    unw_filenames = sorted(output_path.glob("*[0-9].unw.tif"))
    conncomp_filenames = _create_conncomps_from_mask(
        temporal_coherence_filename,
        options.temporal_coherence_threshold,
        unw_filenames=unw_filenames,
    )

    if options.run_ambiguity_interpolation:
        filled_masked_unw_regions(unw_filenames, ifg_filenames)
    return unw_filenames, conncomp_filenames


def _create_conncomps_from_mask(
    temporal_coherence_filename: PathOrStr,
    temporal_coherence_threshold: float,
    unw_filenames: Sequence[PathOrStr],
    dilate_by: int = 25,
) -> list[Path]:
    arr = io.load_gdal(temporal_coherence_filename, masked=True)
    good_pixels = arr > temporal_coherence_threshold
    strel = np.ones((dilate_by, dilate_by))
    # "1" pixels will be spread out and have (approximately) 1.0 in surrounding pixels
    # Note: `ndimage.binary_dilation` scales ~quadratically with `dilate_by`,
    # whereas FFT-based convolution is roughly constant for any `dilate_by`.
    good_pixels_dilated = signal.fftconvolve(good_pixels, strel, mode="same") > 0.95
    # Label the contiguous areas based on the dilated version
    labels, nlabels = ndimage.label(good_pixels_dilated)
    logger.debug("Labeled %s connected components", nlabels)
    # Now these labels will extend to areas which are not "good" in the original.
    labels[~good_pixels] = 0
    # An make a masked version based on the original nodata, then fill with final output
    conncomp_arr = (
        np.ma.MaskedArray(data=labels, mask=arr.mask)
        .astype("uint16")
        .filled(DEFAULT_CCL_NODATA)
    )

    conncomp_files = [
        Path(str(outf).replace(UNW_SUFFIX, CONNCOMP_SUFFIX)) for outf in unw_filenames
    ]
    # Write the first one with geo-metadata
    io.write_arr(
        arr=conncomp_arr,
        like_filename=unw_filenames[0],
        output_name=conncomp_files[0],
        nodata=DEFAULT_CCL_NODATA,
    )
    for f in conncomp_files[1:]:
        shutil.copy(conncomp_files[0], f)
    return conncomp_files


def _process_single_unw(
    unw_filename: PathOrStr,
    ifg_filenames: Sequence[PathOrStr],
    output_dir: Path,
    profile: dict,
):
    unw, wrapped_phase = _reform_wrapped_phase(unw_filename, ifg_filenames)
    interpolate_masked_gaps(unw, wrapped_phase)
    # Save the updated unwrapped phase
    kwargs = profile | {
        "count": 1,
        "height": unw.shape[0],
        "width": unw.shape[1],
        "dtype": "float32",
    }
    with rasterio.open(output_dir / Path(unw_filename).name, "w", **kwargs) as src:
        src.write(unw, 1)


def filled_masked_unw_regions(
    unw_filenames: Sequence[PathOrStr],
    ifg_filenames: Sequence[PathOrStr],
    output_dir: Path | None = None,
    max_workers: int = 3,
) -> None:
    """Fill the nan gaps in `unw_filenames` using the wrapped `ifg_filenames`.

    This function iterates through the nearest-3 unwrapped filenames from spurt,
    calculates the wrapped phase difference from 2 `ifg_filenames` and interpolates
    the unwrapped ambiguity number to fill the gaps.

    Parameters
    ----------
    unw_filenames : Sequence[PathOrStr]
        List of the nearest-3 unwrapped filenames from spurt, containing nan gaps.
    ifg_filenames : Sequence[PathOrStr]
        Wrapped, single-reference interferogram filenames used as input to spurt.
    output_dir : Path, optional
        Separate folder to write output files after filling gaps.
        If None, overwrites the `unw_filenames`.
    max_workers : int
        Number of parallel unwrapped files to process at once.
        Default is 3.

    """
    if output_dir is None:
        output_dir = Path(unw_filenames[0]).parent
    with rasterio.open(unw_filenames[0]) as src:
        profile = src.profile.copy()

    process_func = partial(
        _process_single_unw,
        ifg_filenames=ifg_filenames,
        output_dir=output_dir,
        profile=profile,
    )

    # Use multiprocessing to process files in parallel
    with multiprocessing.get_context("spawn").Pool(max_workers) as pool:
        pool.map(process_func, unw_filenames)


def _reform_wrapped_phase(
    unw_filename: PathOrStr, ifg_filenames: Sequence[PathOrStr]
) -> tuple[NDArray[np.float64], NDArray[np.complex64]]:
    """Load unwrapped phase, and re-calculate the corresponding wrapped phase.

    Finds the matching ifg to `unw_filename`, or uses 2 to compute the correct
    wrapped phase. For example, if `unw_filename` is like (day4_day5), then we load
    the `ifg1 = (day1_day4)`, `ifg2 = (day1_day5)`, and compute `a * b.conj()`.
    """
    # Extract dates from unw_filename
    unw_dates = get_dates(Path(unw_filename))

    date1, date2 = unw_dates

    ifg_date_tuples = [get_dates(p) for p in ifg_filenames]
    if len({tup[0] for tup in ifg_date_tuples}) > 1:
        raise ValueError(
            "ifg_filenames must contain only single-reference interferograms"
        )

    # Find the required interferogram filenames
    ifg1_name = None
    ifg2_name = None
    for ifg in ifg_filenames:
        ifg_dates = get_dates(Path(ifg))
        if len(ifg_dates) != 2:
            continue
        if ifg_dates == unw_dates:
            ifg1_name = ifg
            break

        _ref, sec_date = ifg_dates
        if sec_date == date1:
            ifg1_name = ifg
        if sec_date == date2:
            ifg2_name = ifg

    if ifg1_name is None and ifg2_name is None:
        raise ValueError(f"Could not find required interferograms for {unw_filename}")

    logger.info(f"Interpolating nodata in {unw_filename} with {ifg1_name}, {ifg2_name}")
    with rasterio.open(unw_filename) as src:
        unw = src.read(1)

    with rasterio.open(ifg1_name) as src:
        ifg1 = src.read(1)

    if ifg2_name is not None:
        with rasterio.open(ifg2_name) as src:
            ifg2 = src.read(1)
        # Calculate the wrapped phase difference
        wrapped_phase = np.angle(ifg1 * np.conj(ifg2))
    else:
        wrapped_phase = ifg1

    return unw, wrapped_phase
