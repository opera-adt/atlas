from __future__ import annotations

import logging
from contextlib import ExitStack
from pathlib import Path
from typing import Optional

import numpy as np

from dolphin._types import Filename
from dolphin.io._core import DEFAULT_TIFF_OPTIONS_RIO
from dolphin.utils import full_suffix

from ._constants import CONNCOMP_SUFFIX, DEFAULT_CCL_NODATA, DEFAULT_UNW_NODATA
from ._utils import _zero_from_mask

__all__ = [
    "unwrap_whirlwind",
]


logger = logging.getLogger("dolphin")


def unwrap_whirlwind(
    ifg_filename: Filename,
    corr_filename: Filename,
    unw_filename: Filename,
    nlooks: float,
    mask_file: Optional[Filename] = None,
    zero_where_masked: bool = False,
    unw_nodata: Optional[float] = DEFAULT_UNW_NODATA,
    ccl_nodata: Optional[int] = DEFAULT_CCL_NODATA,
    scratchdir: Optional[Filename] = None,
) -> tuple[Path, Path]:
    """Unwrap an interferogram using `whirlwind`.

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
    mask_file : Filename, optional
        Path to binary byte mask file, by default None.
        Assumes that 1s are valid pixels and 0s are invalid.
    zero_where_masked : bool, optional
        Set wrapped phase/correlation to 0 where mask is 0 before unwrapping.
        If not mask is provided, this is ignored.
        By default True.
    unw_nodata : float, optional
        If providing `unwrap_callback`, provide the nodata value for your
        unwrapping function.
    ccl_nodata : float, optional
        Nodata value for the connected component labels.
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
    import whirlwind as ww

    # Create a context manager that combines other context managers -- one for each
    # input raster file. Upon exiting the context block, each context manager in the
    # stack will be closed in LIFO order.
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

        logger.info("Unwrapping using whirlwind")
        # FIXME: Ad hoc kludge to prevent NaN's in whirlwind cost computation.
        # Remove this when the issue is fixed upstream.
        nlooks = np.clip(nlooks / 2, 1.0, 20.0)
        unw = ww.unwrap(igram, corr, nlooks, mask=mask)

        logger.info("Writing unwrapped phase to raster file")
        with snaphu.io.Raster.create(
            unw_filename,
            like=igram,
            nodata=unw_nodata,
            dtype=np.float32,
            **DEFAULT_TIFF_OPTIONS_RIO,
        ) as unw_raster:
            unw_raster[:, :] = unw

        unw_suffix = full_suffix(unw_filename)
        cc_filename = str(unw_filename).replace(unw_suffix, CONNCOMP_SUFFIX)

        # XXX Whirlwind does not yet provide connected components. Instead, grow
        # connected components using SNAPHU's 'SMOOTH' cost function for now.
        logger.info("Growing connected component labels using SNAPHU")
        with snaphu.io.Raster.create(
            cc_filename,
            like=igram,
            nodata=ccl_nodata,
            dtype=np.uint16,
            **DEFAULT_TIFF_OPTIONS_RIO,
        ) as conncomp:
            snaphu.grow_conncomps(
                unw=unw,
                corr=corr,
                nlooks=nlooks,
                mask=mask,
                cost="smooth",
                scratchdir=scratchdir,
                conncomp=conncomp,
            )

    if zero_where_masked and (mask_file is not None):
        logger.info(f"Zeroing unw/conncomp of pixels masked in {mask_file}")
        return _zero_from_mask(unw_filename, cc_filename, mask_file)

    return Path(unw_filename), Path(cc_filename)
