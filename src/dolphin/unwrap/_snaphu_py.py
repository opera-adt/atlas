from __future__ import annotations

import logging
from contextlib import ExitStack
from pathlib import Path
from typing import Optional

from dolphin._types import Filename
from dolphin.io._core import DEFAULT_TIFF_OPTIONS_RIO
from dolphin.utils import full_suffix

from ._constants import CONNCOMP_SUFFIX, DEFAULT_CCL_NODATA, DEFAULT_UNW_NODATA
from ._utils import _zero_from_mask

logger = logging.getLogger("dolphin")


def unwrap_snaphu_py(
    ifg_filename: Filename,
    corr_filename: Filename,
    unw_filename: Filename,
    nlooks: float,
    ntiles: tuple[int, int] = (1, 1),
    tile_overlap: tuple[int, int] = (0, 0),
    nproc: int = 1,
    mask_file: Optional[Filename] = None,
    zero_where_masked: bool = False,
    unw_nodata: Optional[float] = DEFAULT_UNW_NODATA,
    ccl_nodata: Optional[int] = DEFAULT_CCL_NODATA,
    init_method: str = "mst",
    single_tile_reoptimize: bool = False,
    min_conncomp_frac: float = 0.001,
    cost: str = "smooth",
    scratchdir: Optional[Filename] = None,
) -> tuple[Path, Path]:
    """Unwrap an interferogram using `SNAPHU`.

    Parameters
    ----------
    ifg_filename : Filename
        Path to input interferogram.
    corr_filename : Filename
        Path to input correlation file.
    unw_filename : Filename
        Path to output unwrapped phase file.
    nlooks : float
        Effective number of looks used to form the input correlation data.
    ntiles : tuple[int, int], optional
        Number of (row, column) tiles to split for full image into.
        If `ntiles` is an int, will use `(ntiles, ntiles)`
    tile_overlap : tuple[int, int], optional
        Number of pixels to overlap in the (row, col) direction.
        Default = (0, 0)
    nproc : int, optional
        If specifying `ntiles`, number of processes to spawn to unwrap the
        tiles in parallel.
        Default = 1, which unwraps each tile in serial.
    mask_file : Filename, optional
        Path to binary byte mask file, by default None.
        Assumes that 1s are valid pixels and 0s are invalid.
    zero_where_masked : bool, optional
        Set wrapped phase/correlation to 0 where mask is 0 before unwrapping.
        If not mask is provided, this is ignored.
        By default False.
    unw_nodata : float, optional
        If providing `unwrap_callback`, provide the nodata value for your
        unwrapping function.
    ccl_nodata : float, optional
        Nodata value for the connected component labels.
    init_method : str, choices = {"mcf", "mst"}
        initialization method, by default "mst"
    single_tile_reoptimize : bool
        If True, after unwrapping with multiple tiles, an additional post-processing
        unwrapping step is performed to re-optimize the unwrapped phase using a single
        tile. This option is disregarded when `ntiles` is (1, 1). It supersedes the
        `regrow_conncomps` option -- if both are enabled, only the single-tile
        re-optimization step will be performed in order to avoid redundant computation.
        Defaults to False.
    min_conncomp_frac : float, optional
        Minimum size of a single connected component, as a fraction of the total number
        of pixels in the tile. Defaults to 1e-3
    cost : str
        Statistical cost mode.
        Default = "smooth"
    scratchdir : Filename, optional
        If provided, uses a scratch directory to save the intermediate files
        during unwrapping.

    Returns
    -------
    unw_path : Path
        Path to output unwrapped phase file.
    conncomp_path : Path
        Path to output connected component label file.

    """
    import snaphu

    unw_suffix = full_suffix(unw_filename)
    cc_filename = str(unw_filename).replace(unw_suffix, CONNCOMP_SUFFIX)
    with ExitStack() as stack:
        if zero_where_masked and (mask_file is not None):
            logger.info(f"Zeroing phase/corr of pixels masked in {mask_file}")
            zeroed_ifg_file, zeroed_corr_file = _zero_from_mask(
                ifg_filename, corr_filename, mask_file
            )
            igram = stack.enter_context(snaphu.io.Raster(zeroed_ifg_file))
            corr = stack.enter_context(snaphu.io.Raster(zeroed_corr_file))
        else:
            igram = stack.enter_context(snaphu.io.Raster(ifg_filename))
            corr = stack.enter_context(snaphu.io.Raster(corr_filename))

        if mask_file is None:
            mask = None
        else:
            mask = stack.enter_context(snaphu.io.Raster(mask_file))

        unw, conncomp = snaphu.unwrap(
            igram,
            corr,
            nlooks=nlooks,
            init=init_method,
            cost=cost,
            mask=mask,
            ntiles=ntiles,
            tile_overlap=tile_overlap,
            nproc=nproc,
            scratchdir=scratchdir,
            single_tile_reoptimize=single_tile_reoptimize,
            min_conncomp_frac=min_conncomp_frac,
            # https://github.com/isce-framework/snaphu-py/commit/a77cbe1ff115d96164985523987b1db3278970ed
            # On frame-sized ifgs, especially with decorrelation, defaults of
            # (500, 100) for (tile_cost_thresh, min_region_size) lead to
            # "Exceeded maximum number of secondary arcs"
            # "Decrease TILECOSTTHRESH and/or increase MINREGIONSIZE"
            tile_cost_thresh=500,
            # ... "and/or increase MINREGIONSIZE"
            min_region_size=300,
        )

        # Save the numpy results
        with snaphu.io.Raster.create(
            unw_filename,
            like=igram,
            nodata=unw_nodata,
            dtype="f4",
            **DEFAULT_TIFF_OPTIONS_RIO,
        ) as unw_raster:
            unw_raster[:, :] = unw
        with snaphu.io.Raster.create(
            cc_filename,
            like=igram,
            nodata=ccl_nodata,
            dtype="u2",
            **DEFAULT_TIFF_OPTIONS_RIO,
        ) as conncomp_raster:
            conncomp_raster[:, :] = conncomp

    if zero_where_masked and mask_file is not None:
        logger.info(f"Zeroing unw/conncomp of pixels masked in {mask_file}")

        return _zero_from_mask(unw_filename, cc_filename, mask_file)

    return Path(unw_filename), Path(cc_filename)


def grow_conncomp_snaphu(
    unw_filename: Filename,
    corr_filename: Filename,
    nlooks: float,
    mask_filename: Optional[Filename] = None,
    ccl_nodata: Optional[int] = DEFAULT_CCL_NODATA,
    cost: str = "smooth",
    min_conncomp_frac: float = 0.0001,
    scratchdir: Optional[Filename] = None,
) -> Path:
    """Compute connected component labels using SNAPHU.

    Parameters
    ----------
    unw_filename : Filename
        Path to output unwrapped phase file.
    corr_filename : Filename
        Path to input correlation file.
    nlooks : float
        Effective number of looks used to form the input correlation data.
    mask_filename : Filename, optional
        Path to binary byte mask file, by default None.
        Assumes that 1s are valid pixels and 0s are invalid.
    ccl_nodata : float, optional
        Nodata value for the connected component labels.
    cost : str
        Statistical cost mode.
        Default = "smooth"
    min_conncomp_frac : float, optional
        Minimum size of a single connected component, as a fraction of the total number
        of pixels in the array. Defaults to 0.0001.
    scratchdir : Filename, optional
        If provided, uses a scratch directory to save the intermediate files
        during unwrapping.

    Returns
    -------
    conncomp_path : Filename
        Path to output connected component label file.

    """
    import snaphu

    unw_suffix = full_suffix(unw_filename)
    cc_filename = str(unw_filename).replace(unw_suffix, CONNCOMP_SUFFIX)

    with ExitStack() as stack:
        unw = stack.enter_context(snaphu.io.Raster(unw_filename))
        corr = stack.enter_context(snaphu.io.Raster(corr_filename))
        conncomp = stack.enter_context(
            snaphu.io.Raster.create(
                cc_filename,
                like=unw,
                nodata=ccl_nodata,
                dtype="u2",
                **DEFAULT_TIFF_OPTIONS_RIO,
            )
        )

        if mask_filename is not None:
            mask = stack.enter_context(snaphu.io.Raster(mask_filename))
        else:
            mask = None

        snaphu.grow_conncomps(
            unw=unw,
            corr=corr,
            nlooks=nlooks,
            mask=mask,
            cost=cost,
            min_conncomp_frac=min_conncomp_frac,
            scratchdir=scratchdir,
            conncomp=conncomp,
        )

    return Path(cc_filename)
