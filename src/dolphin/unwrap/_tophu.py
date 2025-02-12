from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from dolphin import io
from dolphin._types import Filename
from dolphin.io._core import DEFAULT_TIFF_OPTIONS_RIO
from dolphin.utils import full_suffix
from dolphin.workflows import UnwrapMethod

from ._constants import CONNCOMP_SUFFIX, DEFAULT_CCL_NODATA, DEFAULT_UNW_NODATA
from ._utils import _redirect_unwrapping_log, _zero_from_mask

logger = logging.getLogger("dolphin")


def multiscale_unwrap(
    ifg_filename: Filename,
    corr_filename: Filename,
    unw_filename: Filename,
    downsample_factor: tuple[int, int],
    ntiles: tuple[int, int],
    nlooks: float,
    mask_file: Filename | None = None,
    zero_where_masked: bool = False,
    unwrap_method: UnwrapMethod = UnwrapMethod.SNAPHU,
    unwrap_callback=None,  # type is `tophu.UnwrapCallback`
    unw_nodata: float | None = DEFAULT_UNW_NODATA,
    ccl_nodata: int | None = DEFAULT_CCL_NODATA,
    init_method: str = "mst",
    cost: str = "smooth",
    scratchdir: Filename | None = None,
    log_to_file: bool = True,
) -> tuple[Path, Path]:
    """Unwrap an interferogram using at multiple scales using `tophu`.

    Parameters
    ----------
    ifg_filename : Filename
        Path to input interferogram.
    corr_filename : Filename
        Path to input correlation file.
    unw_filename : Filename
        Path to output unwrapped phase file.
    downsample_factor : tuple[int, int]
        Downsample the interferograms by this factor to unwrap faster, then upsample
    ntiles : tuple[int, int]
        Number of tiles to split for full image into for high res unwrapping.
        If `ntiles` is an int, will use `(ntiles, ntiles)`
    nlooks : float
        Effective number of looks used to form the input correlation data.
    mask_file : Filename, optional
        Path to binary byte mask file, by default None.
        Assumes that 1s are valid pixels and 0s are invalid.
    zero_where_masked : bool, optional
        Set wrapped phase/correlation to 0 where mask is 0 before unwrapping.
        If not mask is provided, this is ignored.
        By default False.
    unwrap_method : UnwrapMethod or str, optional, default = "snaphu"
        Choice of unwrapping algorithm to use.
        Choices: {"snaphu", "icu", "phass"}
    unwrap_callback : tophu.UnwrapCallback
        Alternative to `unwrap_method`: directly provide a callable
        function usable in `tophu`. See [tophu.UnwrapCallback] docs for interface.
    unw_nodata : float, optional.
        If providing `unwrap_callback`, provide the nodata value for your
        unwrapping function.
    ccl_nodata : float, optional
        Nodata value for the connected component labels.
    init_method : str, choices = {"mcf", "mst"}
        SNAPHU initialization method, by default "mst"
    cost : str, choices = {"smooth", "defo", "p-norm",}
        SNAPHU cost function, by default "smooth"
    scratchdir : Filename, optional
        Path to scratch directory to hold intermediate files.
        If None, uses `tophu`'s `/tmp/...` default.
    log_to_file : bool, optional
        Redirect isce3's logging output to file, by default True

    Returns
    -------
    unw_path : Path
        Path to output unwrapped phase file.
    conncomp_path : Path
        Path to output connected component label file.

    """
    import rasterio as rio
    import tophu

    def _get_rasterio_crs_transform(filename: Filename):
        with rio.open(filename) as src:
            return src.crs, src.transform

    def _get_cb_and_nodata(unwrap_method, unwrap_callback, nodata):
        if unwrap_callback is not None:
            # Pass through what the user gave
            return unwrap_callback, nodata
        # Otherwise, set defaults depending on the method
        unwrap_method = UnwrapMethod(unwrap_method)
        if unwrap_method == UnwrapMethod.ICU:
            unwrap_callback = tophu.ICUUnwrap()
            nodata = 0  # TODO: confirm this?
        elif unwrap_method == UnwrapMethod.PHASS:
            unwrap_callback = tophu.PhassUnwrap()
            nodata = -10_000
        elif unwrap_method == UnwrapMethod.SNAPHU:
            unwrap_callback = tophu.SnaphuUnwrap(
                cost=cost,
                init_method=init_method,
            )
            nodata = 0
        else:
            msg = f"Unknown {unwrap_method = }"
            raise ValueError(msg)
        return unwrap_callback, nodata

    # Used to track if we can redirect logs or not
    _user_gave_callback = unwrap_callback is not None
    unwrap_callback, unw_nodata = _get_cb_and_nodata(
        unwrap_method, unwrap_callback, unw_nodata
    )
    # Used to track if we can redirect logs or not

    (width, height) = io.get_raster_xysize(ifg_filename)
    crs, transform = _get_rasterio_crs_transform(ifg_filename)

    unw_suffix = full_suffix(unw_filename)
    conncomp_filename = str(unw_filename).replace(unw_suffix, CONNCOMP_SUFFIX)

    # SUFFIX=ADD
    # Convert to something rasterio understands
    logger.debug(f"Saving conncomps to {conncomp_filename}")
    conncomp_rb = tophu.RasterBand(
        conncomp_filename,
        height=height,
        width=width,
        dtype=np.uint16,
        driver="GTiff",
        crs=crs,
        transform=transform,
        nodata=ccl_nodata,
        **DEFAULT_TIFF_OPTIONS_RIO,
    )
    unw_rb = tophu.RasterBand(
        unw_filename,
        height=height,
        width=width,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=unw_nodata,
        **DEFAULT_TIFF_OPTIONS_RIO,
    )

    if zero_where_masked and mask_file is not None:
        logger.info(f"Zeroing phase/corr of pixels masked in {mask_file}")
        zeroed_ifg_file, zeroed_corr_file = _zero_from_mask(
            ifg_filename, corr_filename, mask_file
        )
        igram_rb = tophu.RasterBand(zeroed_ifg_file)
        coherence_rb = tophu.RasterBand(zeroed_corr_file)
    else:
        igram_rb = tophu.RasterBand(ifg_filename)
        coherence_rb = tophu.RasterBand(corr_filename)

    if log_to_file and not _user_gave_callback:
        # Note that if they gave an arbitrary callback, we dont know the logger
        _redirect_unwrapping_log(unw_filename, unwrap_method.value)

    tophu.multiscale_unwrap(
        unw_rb,
        conncomp_rb,
        igram_rb,
        coherence_rb,
        nlooks=nlooks,
        unwrap_func=unwrap_callback,
        downsample_factor=downsample_factor,
        ntiles=ntiles,
        scratchdir=scratchdir,
    )
    if zero_where_masked and mask_file is not None:
        logger.info(f"Zeroing unw/conncomp of pixels masked in {mask_file}")
        return _zero_from_mask(unw_filename, conncomp_filename, mask_file)
    elif unwrap_method == UnwrapMethod.PHASS:
        # Fill in the nan pixels with the nearest ambiguities
        from ._post_process import interpolate_masked_gaps

        with (
            rio.open(unw_filename, mode="r+") as u_src,
            rio.open(igram_rb.filepath) as i_src,
        ):
            unw = u_src.read(1)
            ifg = i_src.read(1)
            # nodata_mask = i_src.read_masks(1) != 0
            interpolate_masked_gaps(unw, ifg)
            u_src.write(unw, 1)

    return Path(unw_filename), Path(conncomp_filename)
