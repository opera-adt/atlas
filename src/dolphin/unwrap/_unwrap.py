from __future__ import annotations

import itertools
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from tqdm.auto import tqdm

from dolphin import io
from dolphin._types import Filename
from dolphin.goldstein import goldstein
from dolphin.interpolation import interpolate
from dolphin.utils import DummyProcessPoolExecutor, full_suffix
from dolphin.workflows import UnwrapMethod, UnwrapOptions

from ._constants import (
    CONNCOMP_SUFFIX,
    CONNCOMP_SUFFIX_ZEROED,
    DEFAULT_CCL_NODATA,
    DEFAULT_UNW_NODATA,
    UNW_SUFFIX,
    UNW_SUFFIX_ZEROED,
)
from ._snaphu_py import grow_conncomp_snaphu, unwrap_snaphu_py
from ._tophu import multiscale_unwrap
from ._unwrap_3d import unwrap_spurt
from ._utils import create_combined_mask, set_nodata_values
from ._whirlwind import unwrap_whirlwind

logger = logging.getLogger("dolphin")

__all__ = ["run", "unwrap"]

DEFAULT_OPTIONS = UnwrapOptions()


def run(
    ifg_filenames: Sequence[Filename],
    cor_filenames: Sequence[Filename],
    output_path: Filename,
    *,
    unwrap_options: UnwrapOptions = DEFAULT_OPTIONS,
    nlooks: float = 5,
    temporal_coherence_filename: Filename | None = None,
    similarity_filename: Filename | None = None,
    mask_filename: Filename | None = None,
    unw_nodata: float | None = DEFAULT_UNW_NODATA,
    ccl_nodata: int | None = DEFAULT_CCL_NODATA,
    scratchdir: Filename | None = None,
    delete_intermediate: bool = True,
    overwrite: bool = False,
) -> tuple[list[Path], list[Path]]:
    """Run snaphu on all interferograms in a directory.

    Parameters
    ----------
    ifg_filenames : Sequence[Filename]
        Paths to input interferograms.
    cor_filenames : Sequence[Filename]
        Paths to input correlation files. Order must match `ifg_filenames`.
    output_path : Filename
        Path to output directory.
    unwrap_options : UnwrapOptions, optional
        [`UnwrapOptions`][dolphin.workflows.config.UnwrapOptions] config object
        with parameters and settings for unwrapping.
    nlooks : int, optional
        Effective number of looks used to form the input correlation data.
    temporal_coherence_filename : Filename, optional
        Path to temporal coherence file from phase linking.
    similarity_filename : Filename, optional
        Path to phase cosine similarity file from phase linking.
    mask_filename : Filename, optional
        Path to binary byte mask file, by default None.
        Assumes that 1s are valid pixels and 0s are invalid.
    unw_nodata : float , optional.
        Requested nodata value for the unwrapped phase.
        Default = 0
    ccl_nodata : float, optional
        Requested nodata value for connected component labels.
        Default = max value of UInt16 (65535)
    scratchdir : Filename, optional
        Path to scratch directory to hold intermediate files.
        If None, uses the unwrapper's default.
    delete_intermediate : bool, default = True
        Delete the temporary files made in the scratchdir after completion.
        If True, will make separate folders inside `scratchdir` for cleaner
        removals (in case `scratchdir`) has other contents.
        Must specify `scratchdir` for this option to be used.
    overwrite : bool, optional, default = False
        Overwrite existing unwrapped files.

    Returns
    -------
    unw_paths : list[Path]
        list of unwrapped files names
    conncomp_paths : list[Path]
        list of connected-component-label files names

    """
    if scratchdir is None:
        delete_intermediate = False

    if len(cor_filenames) != len(ifg_filenames):
        msg = (
            "Number of correlation files does not match number of interferograms."
            f" Found {len(cor_filenames)} correlation files and"
            f" {len(ifg_filenames)} interferograms."
        )
        raise ValueError(msg)

    output_path = Path(output_path)
    if unwrap_options.unwrap_method == UnwrapMethod.SPURT:
        if temporal_coherence_filename is None:
            # TODO: we should make this a mask, instead of requiring this.
            # we'll need to change spurt
            raise ValueError("temporal coherence required for spurt unwrapping")
        unw_paths, conncomp_paths = unwrap_spurt(
            ifg_filenames=ifg_filenames,
            output_path=output_path,
            temporal_coherence_filename=temporal_coherence_filename,
            similarity_filename=similarity_filename,
            # cor_filenames=cor_filenames,
            mask_filename=mask_filename,
            options=unwrap_options.spurt_options,
            scratchdir=scratchdir,
        )
        for f in unw_paths:
            io.set_raster_units(f, "radians")
        return unw_paths, conncomp_paths

    ifg_suffixes = [full_suffix(f) for f in ifg_filenames]
    all_out_files = [
        (output_path / Path(f).name.replace(suf, UNW_SUFFIX))
        for f, suf in zip(ifg_filenames, ifg_suffixes)
    ]
    in_files, out_files = [], []
    for inf, outf in zip(ifg_filenames, all_out_files):
        if Path(outf).exists() and not overwrite:
            logger.info(f"{outf} exists. Skipping.")
            continue

        in_files.append(inf)
        out_files.append(outf)
    logger.info(f"{len(out_files)} left to unwrap")

    if mask_filename:
        mask_filename = Path(mask_filename).resolve()

    if delete_intermediate:
        assert scratchdir is not None  # can't be none from the previous check
        scratch_dirs: list[Path | None] = [
            Path(scratchdir) / f"scratch-{Path(ifg_file).stem}" for ifg_file in in_files
        ]
    else:
        scratch_dirs = itertools.repeat(scratchdir)  # type: ignore[assignment]
    # This keeps it from spawning a new process for a single job.
    max_jobs = unwrap_options.n_parallel_jobs

    Executor = ThreadPoolExecutor if max_jobs > 1 else DummyProcessPoolExecutor
    with Executor(max_workers=max_jobs) as exc:
        futures = [
            exc.submit(
                unwrap,
                ifg_filename=ifg_file,
                corr_filename=cor_file,
                unw_filename=out_file,
                nlooks=nlooks,
                mask_filename=mask_filename,
                similarity_filename=similarity_filename,
                unwrap_options=unwrap_options,
                unw_nodata=unw_nodata,
                ccl_nodata=ccl_nodata,
                scratchdir=cur_scratch,
                delete_scratch=delete_intermediate,
            )
            for ifg_file, out_file, cor_file, cur_scratch in zip(
                in_files, out_files, cor_filenames, scratch_dirs
            )
        ]
        for fut in tqdm(as_completed(futures)):
            # We're not passing all the unw files in, so we need to tally up below
            _unw_path, _cc_path = fut.result()

    if unwrap_options.zero_where_masked and mask_filename is not None:
        all_out_files = [
            Path(str(outf).replace(UNW_SUFFIX, UNW_SUFFIX_ZEROED))
            for outf in all_out_files
        ]
        conncomp_files = [
            Path(str(outf).replace(UNW_SUFFIX_ZEROED, CONNCOMP_SUFFIX_ZEROED))
            for outf in all_out_files
        ]
    else:
        conncomp_files = [
            Path(str(outf).replace(UNW_SUFFIX, CONNCOMP_SUFFIX))
            for outf in all_out_files
        ]
    for f in all_out_files:
        io.set_raster_units(f, "radians")
    return all_out_files, conncomp_files


def transfer_ambiguities(wrapped: np.ndarray, unw_est: np.ndarray) -> np.ndarray:
    """Compute unwrapped phase by transferring ambiguities from an unwrapped estimate.

    Transfer the ambiguities from an unwrapped phase estimate to the original wrapped
    phase in order to form a new unwrapped phase array that is congruent (i.e. differs
    from the wrapped phase only by multiples of 2pi).

    Parameters
    ----------
    wrapped : numpy.ndarray
        The initial wrapped phase array, in radians. A 2-D, real-valued array.
    unw_est : numpy.ndarray
        An estimate of the unwrapped phase, in radians. A 2-D, real-valued array with
        the same shape as `wrapped`. May differ from the wrapped phase by non-integer
        cycles.

    Returns
    -------
    numpy.ndarray
        The unwrapped phase data, rounded pixel-wise to the nearest value that differs
        from the wrapped phase by a multiple of 2pi.

    """
    # Measure the difference between the unwrapped & wrapped phase, rounded to the
    # nearest phase cycle.
    ambiguity = np.round((unw_est - wrapped) / (2 * np.pi))

    # Convert ambiguities back to radians and add them to the wrapped phase.
    return wrapped + 2 * np.pi * ambiguity


def unwrap(
    ifg_filename: Filename,
    corr_filename: Filename,
    unw_filename: Filename,
    nlooks: float,
    mask_filename: Optional[Filename] = None,
    similarity_filename: Optional[Filename] = None,
    unwrap_options: UnwrapOptions = DEFAULT_OPTIONS,
    log_to_file: bool = True,
    unw_nodata: float | None = DEFAULT_UNW_NODATA,
    ccl_nodata: int | None = DEFAULT_CCL_NODATA,
    scratchdir: Optional[Filename] = None,
    delete_scratch: bool = False,
) -> tuple[Path, Path]:
    """Unwrap a single interferogram.

    Parameters
    ----------
    ifg_filename : Filename
        Path to input interferogram.
    corr_filename : Filename
        Path to input correlation file.
    unw_filename : Filename
        Path to output unwrapped phase file.
    unwrap_options : UnwrapOptions, optional
        [`UnwrapOptions`][dolphin.workflows.config.UnwrapOptions] config object
        with parameters and settings for unwrapping.
    nlooks : float
        Effective number of looks used to form the input correlation data.
    mask_filename : Filename, optional
        Path to binary byte mask file, by default None.
        Assumes that 1s are valid pixels and 0s are invalid.
    similarity_filename : Filename, optional
        Path to phase cosine similarity file from phase linking.
    log_to_file : bool, optional
        Redirect isce3 logging output to file, by default True
    unw_nodata : float , optional.
        Requested nodata value for the unwrapped phase.
        Default = 0
    ccl_nodata : float, optional
        Requested nodata value for connected component labels.
        Default = max value of UInt16 (65535)
    scratchdir : Filename, optional
        Path to scratch directory to hold intermediate files.
        If None, uses `tophu`'s `/tmp/...` default.
    delete_scratch : bool, default = False
        After unwrapping, delete the contents inside `scratchdir`.

    Returns
    -------
    unw_path : Path
        Path to output unwrapped phase file.
    conncomp_path : Path
        Path to output connected component label file.

    """
    unwrap_method = unwrap_options.unwrap_method
    preproc_options = unwrap_options.preprocess_options
    if scratchdir is None:
        # Let the unwrappers handle the scratch if we don't specify.
        delete_scratch = False
    else:
        Path(scratchdir).mkdir(parents=True, exist_ok=True)

    # Check for a nodata mask
    if io.get_raster_nodata(ifg_filename) is None or mask_filename is None:
        # With no marked `nodata`, just use the passed in mask
        combined_mask_file = mask_filename
    else:
        combined_mask_file = Path(ifg_filename).with_suffix(".mask.tif")
        create_combined_mask(
            mask_filename=mask_filename,
            image_filename=ifg_filename,
            output_filename=combined_mask_file,
        )

    unwrapper_ifg_filename = Path(ifg_filename)
    unwrapper_unw_filename = Path(unw_filename)
    name_change = "."

    ifg = io.load_gdal(ifg_filename, masked=True)
    if unwrap_options.run_goldstein:
        suf = Path(unw_filename).suffix
        if suf == ".tif":
            driver = "GTiff"
            opts = list(io.DEFAULT_TIFF_OPTIONS)
        else:
            driver = "ENVI"
            opts = list(io.DEFAULT_ENVI_OPTIONS)

        name_change = ".filt" + name_change
        # If we're running Goldstein filtering, the intermediate
        # filtered/unwrapped rasters are temporary rasters in the scratch dir.
        filt_ifg_filename = Path(scratchdir or ".") / (
            Path(ifg_filename).stem.split(".")[0] + (name_change + "int" + suf)
        )
        filt_unw_filename = Path(
            str(unw_filename).split(".")[0] + (name_change + "unw" + suf)
        )

        logger.info(f"Goldstein filtering {ifg_filename} -> {filt_ifg_filename}")
        modified_ifg = goldstein(ifg.filled(0), alpha=preproc_options.alpha)
        logger.info(f"Writing filtered output to {filt_ifg_filename}")
        io.write_arr(
            arr=modified_ifg,
            output_name=filt_ifg_filename,
            like_filename=ifg_filename,
            driver=driver,
            options=opts,
        )
        unwrapper_ifg_filename = filt_ifg_filename
        unwrapper_unw_filename = filt_unw_filename

    if unwrap_options.run_interpolation:
        suf = Path(ifg_filename).suffix
        if suf == ".tif":
            driver = "GTiff"
            opts = list(io.DEFAULT_TIFF_OPTIONS)
        else:
            driver = "ENVI"
            opts = list(io.DEFAULT_ENVI_OPTIONS)

        pre_interp_ifg_filename = unwrapper_ifg_filename
        pre_interp_unw_filename = unwrapper_unw_filename
        name_change = ".interp" + name_change

        # temporarily storing the intermediate interpolated rasters in the scratch dir.
        interp_ifg_filename = Path(scratchdir or ".") / (
            pre_interp_ifg_filename.stem.split(".")[0] + (name_change + "int" + suf)
        )
        interp_unw_filename = Path(
            str(pre_interp_unw_filename).split(".")[0] + (name_change + "unw" + suf)
        )

        pre_interp_ifg = io.load_gdal(pre_interp_ifg_filename, masked=True).filled(0)

        corr = io.load_gdal(corr_filename)
        if similarity_filename and preproc_options.interpolation_similarity_threshold:
            cutoff = preproc_options.interpolation_similarity_threshold
            logger.info(f"Masking pixels with similarity below {cutoff}")
            sim = io.load_gdal(similarity_filename, masked=True).filled(0)
            coherent_pixel_mask = sim[:] >= cutoff
        else:
            cutoff = preproc_options.interpolation_cor_threshold
            logger.info(f"Masking pixels with correlation below {cutoff}")
            coherent_pixel_mask = corr[:] >= cutoff

        logger.info(f"Interpolating {pre_interp_ifg_filename} -> {interp_ifg_filename}")
        modified_ifg = interpolate(
            ifg=pre_interp_ifg,
            weights=coherent_pixel_mask,
            weight_cutoff=cutoff,
            max_radius=preproc_options.max_radius,
        )

        logger.info(f"Writing interpolated output to {interp_ifg_filename}")
        io.write_arr(
            arr=modified_ifg,
            output_name=interp_ifg_filename,
            like_filename=ifg_filename,
            driver=driver,
            options=opts,
        )
        unwrapper_ifg_filename = interp_ifg_filename
        unwrapper_unw_filename = interp_unw_filename

    if unwrap_method == UnwrapMethod.SNAPHU:
        snaphu_opts = unwrap_options.snaphu_options
        # Pass everything to snaphu-py
        unw_path, conncomp_path = unwrap_snaphu_py(
            unwrapper_ifg_filename,
            corr_filename,
            unwrapper_unw_filename,
            nlooks,
            ntiles=snaphu_opts.ntiles,
            tile_overlap=snaphu_opts.tile_overlap,
            mask_file=combined_mask_file,
            nproc=snaphu_opts.n_parallel_tiles,
            zero_where_masked=unwrap_options.zero_where_masked,
            unw_nodata=unw_nodata,
            ccl_nodata=ccl_nodata,
            init_method=snaphu_opts.init_method,
            cost=snaphu_opts.cost,
            single_tile_reoptimize=snaphu_opts.single_tile_reoptimize,
            scratchdir=scratchdir,
        )
    elif unwrap_method == UnwrapMethod.WHIRLWIND:
        unw_path, conncomp_path = unwrap_whirlwind(
            unwrapper_ifg_filename,
            corr_filename,
            unwrapper_unw_filename,
            nlooks,
            mask_file=combined_mask_file,
            zero_where_masked=unwrap_options.zero_where_masked,
            unw_nodata=unw_nodata,
            ccl_nodata=ccl_nodata,
            scratchdir=scratchdir,
        )
    elif (unwrap_method == UnwrapMethod.ICU) or (unwrap_method == UnwrapMethod.PHASS):
        tophu_opts = unwrap_options.tophu_options
        unw_path, conncomp_path = multiscale_unwrap(
            unwrapper_ifg_filename,
            corr_filename,
            unwrapper_unw_filename,
            tophu_opts.downsample_factor,
            ntiles=tophu_opts.ntiles,
            nlooks=nlooks,
            mask_file=combined_mask_file,
            zero_where_masked=unwrap_options.zero_where_masked,
            unw_nodata=unw_nodata,
            ccl_nodata=ccl_nodata,
            init_method=tophu_opts.init_method,
            cost=tophu_opts.cost,
            unwrap_method=unwrap_method,
            scratchdir=scratchdir,
            log_to_file=log_to_file,
        )
    else:
        # Should be unreachable.
        raise AssertionError(f"unexpected unwrap method {unwrap_method}")

    # post-processing steps go here:

    # Transfer ambiguity numbers from filtered/interpolated unwrapped interferogram
    # back to original interferogram
    if unwrap_options.run_goldstein or unwrap_options.run_interpolation:
        logger.info(
            "Transferring ambiguity numbers from filtered/interpolated"
            f" ifg {unwrapper_unw_filename}"
        )
        unw_arr = io.load_gdal(unwrapper_unw_filename, masked=True).filled(unw_nodata)

        final_arr = transfer_ambiguities(np.angle(ifg), unw_arr)
        final_arr[ifg.mask] = unw_nodata

        io.write_arr(
            arr=final_arr,
            output_name=unw_filename,
            like_filename=unwrapper_unw_filename,
            dtype=np.float32,
            driver=driver,
            options=opts,
        )

        # Regrow connected components after phase modification
        # TODO decide whether we want to have the
        # 'min_conncomp_frac' option in the config
        conncomp_path = grow_conncomp_snaphu(
            unw_filename=unw_filename,
            corr_filename=corr_filename,
            nlooks=nlooks,
            mask_filename=combined_mask_file,
            ccl_nodata=ccl_nodata,
            cost=unwrap_options.snaphu_options.cost,
            scratchdir=scratchdir,
        )

        # Move the intermediate ".interp" or ".goldstein" into the scratch directory
        if scratchdir is not None:
            shutil.move(unwrapper_unw_filename, scratchdir)

    # Reset the input nodata values to be nodata in the unwrapped and CCL
    logger.info(f"Setting nodata values of {unw_path} file")
    set_nodata_values(
        filename=unw_filename, output_nodata=unw_nodata, like_filename=ifg_filename
    )
    logger.info(f"Setting nodata values of {conncomp_path} file")
    set_nodata_values(
        filename=conncomp_path, output_nodata=ccl_nodata, like_filename=ifg_filename
    )

    if delete_scratch:
        assert scratchdir is not None
        shutil.rmtree(scratchdir, ignore_errors=True)

    return Path(unw_filename), conncomp_path
