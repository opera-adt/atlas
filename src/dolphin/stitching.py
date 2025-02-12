"""stitching.py: utilities for combining interferograms into larger images."""

from __future__ import annotations

import logging
import math
import subprocess
import tempfile
from datetime import datetime
from os import fspath
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from numpy.typing import DTypeLike
from opera_utils import group_by_date
from osgeo import gdal, osr
from rasterio.warp import transform_bounds
from tqdm.contrib.concurrent import thread_map

from dolphin import io, utils
from dolphin._types import Bbox, Filename
from dolphin.io import DEFAULT_DATETIME_FORMAT

logger = logging.getLogger("dolphin")


def merge_by_date(
    image_file_list: Iterable[Filename],
    file_date_fmt: str = DEFAULT_DATETIME_FORMAT,
    output_dir: Filename = ".",
    driver: str = "GTiff",
    output_suffix: str = ".int.tif",
    out_nodata: Optional[float] = 0,
    in_nodata: Optional[float] = None,
    out_bounds: Optional[Bbox] = None,
    out_bounds_epsg: Optional[int] = None,
    options: Optional[Sequence[str]] = io.DEFAULT_TIFF_OPTIONS,
    num_workers: int = 1,
    overwrite: bool = False,
) -> dict[tuple[datetime, ...], Path]:
    """Group images from the same datetime and merge into one image per datetime.

    Parameters
    ----------
    image_file_list : Iterable[Filename]
        list of paths to images.
    file_date_fmt : Optional[str]
        Format of the datetime in the filename. Default is %Y%m%d
    output_dir : Filename
        Path to output directory
    driver : str
        GDAL driver to use for output. Default is ENVI.
    output_suffix : str
        Suffix to use to output stitched filenames. Default is ".int"
    out_nodata : Optional[float | str]
        Nodata value to use for output file. Default is 0.
    in_nodata : Optional[float | str]
        Override the files' `nodata` and use `in_nodata` during merging.
    out_bounds: Optional[tuple[float]]
        if provided, forces the output image bounds to
            (left, bottom, right, top).
        Otherwise, computes from the outside of all input images.
    out_bounds_epsg: Optional[int]
        EPSG code for the `out_bounds`.
        If not provided, assumed to match the projections of `file_list`.
    options : Optional[Sequence[str]]
        Driver-specific creation options passed to GDAL.
        Default is [dolphin.io.DEFAULT_TIFF_OPTIONS][].
    num_workers : int
        Number of dates to stitch in separate threads in parallel.
        Default is 1.
    overwrite : bool
        Overwrite existing files. Default is False.

    Returns
    -------
    dict
        key: the datetime of the SLC acquisitions/datetime pair of the interferogram.
        value: the path to the stitched image

    Notes
    -----
    This function is intended to be used with filenames that contain datetime pairs
    (from interferograms).

    """
    image_path_list = [Path(f) for f in image_file_list]
    grouped_images = group_by_date(image_path_list, file_date_fmt=file_date_fmt)
    stitched_acq_times = {}
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for dates, cur_images in grouped_images.items():
        logger.info(f"{dates}: Stitching {len(cur_images)} images.")
        if len(dates) == 2:
            date_str = utils.format_date_pair(*dates)
        elif len(dates) == 1:
            date_str = dates[0].strftime(file_date_fmt)
        else:
            msg = f"Expected 1 or 2 dates: {dates}."
            raise ValueError(msg)
        outfile = Path(output_dir) / (date_str + output_suffix)
        stitched_acq_times[dates] = outfile

    def process_date(args):
        cur_images, outfile = args
        merge_images(
            cur_images,
            outfile=outfile,
            driver=driver,
            overwrite=overwrite,
            out_nodata=out_nodata,
            out_bounds=out_bounds,
            out_bounds_epsg=out_bounds_epsg,
            in_nodata=in_nodata,
            options=options,
        )

    # loop over the merging in parallel
    thread_map(
        process_date,
        list(zip(grouped_images.values(), stitched_acq_times.values())),
        max_workers=num_workers,
        desc="Merging images by date",
    )

    return stitched_acq_times


def merge_images(
    file_list: Sequence[Filename],
    outfile: Filename,
    target_aligned_pixels: bool = True,
    out_bounds: Optional[Bbox] = None,
    out_bounds_epsg: Optional[int] = None,
    strides: Optional[dict[str, int]] = None,
    driver: str = "GTiff",
    out_nodata: Optional[float] = 0,
    out_dtype: Optional[DTypeLike] = None,
    in_nodata: Optional[float] = None,
    resample_alg: str = "lanczos",
    overwrite: bool = False,
    options: Optional[Sequence[str]] = io.DEFAULT_TIFF_OPTIONS,
    create_only: bool = False,
) -> None:
    """Combine multiple SLC images on the same date into one image.

    Parameters
    ----------
    file_list : list[Filename]
        list of raster filenames
    outfile : Filename
        Path to output file
    target_aligned_pixels: bool
        If True, adjust output image bounds so that pixel coordinates
        are integer multiples of pixel size, matching the ``-tap``
        options of GDAL utilities.
        Default is True.
    out_bounds: Optional[tuple[float]]
        if provided, forces the output image bounds to
            (left, bottom, right, top).
        Otherwise, computes from the outside of all input images.
        Note that using `resample_alg='nearest'` may result in bounds not
        equaling the exact `out_bounds` due to the nearest-neighbor
        resampling algorithm in GDAL.
    out_bounds_epsg: Optional[int]
        EPSG code for the `out_bounds`.
        If not provided, assumed to match the projections of `file_list`.
    strides : dict[str, int]
        subsample factor: {"x": x strides, "y": y strides}
    driver : str
        GDAL driver to use for output file. Default is GTiff.
    out_nodata : Optional[float | str]
        Nodata value to use for output file. Default is 0.
    out_dtype : Optional[DTypeLike]
        Output data type. Default is None, which will use the data type
        of the first image in the list.
    in_nodata : Optional[float | str]
        Override the files' `nodata` and use `in_nodata` during merging.
    resample_alg : str, default="lanczos"
        Method for gdal to use for reprojection.
        Default is lanczos (sinc-kernel)
    overwrite : bool
        Overwrite existing files. Default is False.
    options : Optional[Sequence[str]]
        Driver-specific creation options passed to GDAL.
        Default is [dolphin.io.DEFAULT_TIFF_OPTIONS][].
    create_only : bool
        If True, creates an empty output file, does not write data. Default is False.

    """
    if strides is None:
        strides = {"x": 1, "y": 1}
    if Path(outfile).exists():
        if not overwrite:
            logger.info(f"{outfile} already exists, skipping")
            return
        else:
            logger.info(f"Overwrite=True: removing {outfile}")
            Path(outfile).unlink()

    if len(file_list) == 1 and out_bounds is None:
        logger.info("Only one image, no stitching needed")
        logger.info(f"Copying {file_list[0]} to {outfile} and zeroing nodata values.")
        _copy_set_nodata(
            file_list[0],
            outfile=outfile,
            driver=driver,
            creation_options=options,
            out_nodata=out_nodata or 0,
        )
        return

    # Make sure all the files are in the same projection.
    projection = _get_mode_projection(file_list)
    # If not, warp them to the most common projection using VRT files in a tempdir
    temp_dir = tempfile.TemporaryDirectory()

    tmp_path = Path(temp_dir.name)
    if strides is not None and strides["x"] > 1 and strides["y"] > 1:
        file_list = get_downsampled_vrts(
            file_list,
            strides=strides,
            dirname=tmp_path,
        )

    warped_file_list = warp_to_projection(
        file_list,
        dirname=tmp_path,
        projection=projection,
        resample_alg=resample_alg,
    )
    # Compute output array shape. We guarantee it will cover the output
    # bounds completely
    bounds, combined_nodata = get_combined_bounds_nodata(
        *warped_file_list,
        target_aligned_pixels=target_aligned_pixels,
        out_bounds=out_bounds,
        out_bounds_epsg=out_bounds_epsg,
    )
    (xmin, ymin, xmax, ymax) = bounds
    proj_win = (xmin, ymax, xmax, ymin)  # ul_lr = ulx, uly, lrx, lry

    # Write out the files for gdal_merge using the --optfile flag
    optfile = tmp_path / "file_list.txt"
    optfile.write_text("\n".join(map(str, warped_file_list)))
    suffix = Path(outfile).suffix
    merge_output = (tmp_path / "merged").with_suffix(suffix)
    args = [
        "gdal_merge.py",
        "-quiet",
        "-o",
        merge_output,
        "--optfile",
        optfile,
        "-of",
        driver,
    ]

    if out_nodata is not None:
        args.extend(["-a_nodata", str(out_nodata)])
    if in_nodata is not None or combined_nodata is not None:
        ndv = str(in_nodata) if in_nodata is not None else str(combined_nodata)
        args.extend(["-n", ndv])
    if out_dtype is not None:
        out_gdal_dtype = gdal.GetDataTypeName(utils.numpy_to_gdal_type(out_dtype))
        args.extend(["-ot", out_gdal_dtype])
    if create_only:
        args.append("-create")
    if options is not None:
        for option in options:
            args.extend(["-co", option])

    arg_list = [str(a) for a in args]
    logger.debug(f"Running {' '.join(arg_list)}")
    subprocess.check_call(arg_list)

    # Now clip to the provided bounding box
    gdal.Translate(
        destName=fspath(outfile),
        srcDS=fspath(merge_output),
        projWin=proj_win,
        # TODO: https://github.com/OSGeo/gdal/issues/10536
        # Figure out if we really want to resample here, or just
        # do a nearest neighbor (which is default)
        resampleAlg="bilinear",
        format=driver,
        creationOptions=options,
    )

    temp_dir.cleanup()


def get_downsampled_vrts(
    filenames: Sequence[Filename],
    strides: dict[str, int],
    dirname: Filename,
) -> list[Path]:
    """Create downsampled VRTs from a list of files.

    Does not reproject, only uses `gdal_translate`.


    Parameters
    ----------
    filenames : Sequence[Filename]
        list of filenames to warp.
    strides : dict[str, int]
        subsample factor: {"x": x strides, "y": y strides}
    dirname : Filename
        The directory to write the warped files to.

    Returns
    -------
    list[Filename]
        The warped filenames.

    """
    if not filenames:
        return []
    warped_files = []
    res = _get_resolution(filenames)
    for idx, fn in enumerate(filenames):
        p = Path(fn)
        warped_fn = Path(dirname) / _get_temp_filename(p, idx, "_downsampled")
        logger.debug(f"Downsampling {p} by {strides}")
        warped_files.append(warped_fn)
        left, bottom, right, top = io.get_raster_bounds(p)
        gdal.Translate(
            fspath(warped_fn),
            fspath(p),
            format="VRT",  # Just creates a file that will warp on the fly
            resampleAlg="nearest",  # nearest neighbor for resampling
            xRes=res[0] * strides["x"],
            yRes=res[1] * strides["y"],
            projWin=(left, top, right, bottom),
        )

    return warped_files


def _get_temp_filename(fn: Path, idx: int, extra: str = ""):
    base = utils._get_path_from_gdal_str(fn).stem
    return f"{base}_{idx}{extra}.vrt"


def warp_to_projection(
    filenames: Sequence[Filename],
    dirname: Filename,
    projection: str,
    res: Optional[tuple[float, float]] = None,
    resample_alg: str = "lanczos",
) -> list[Path]:
    """Warp a list of files to `projection`.

    If the input file's projection matches `projection`, the same file is returned.
    Otherwise, a new file is created in `dirname` with the same name as the input file,
    but with '_warped' appended.

    Parameters
    ----------
    filenames : Sequence[Filename]
        list of filenames to warp.
    dirname : Filename
        The directory to write the warped files to.
    projection : str
        The desired projection, as a WKT string or 'EPSG:XXXX' string.
    res : tuple[float, float]
        The desired [x, y] resolution.
    resample_alg : str, default="lanczos"
        Method for gdal to use for reprojection.
        Default is lanczos (sinc-kernel)

    Returns
    -------
    list[Filename]
        The warped filenames.

    """
    if projection is None:
        projection = _get_mode_projection(filenames)
    if res is None:
        res = _get_resolution(filenames)

    warped_files = []
    for idx, fn in enumerate(filenames):
        p = Path(fn)
        ds = gdal.Open(fspath(p))
        proj_in = ds.GetProjection()
        if proj_in == projection:
            warped_files.append(p)
            continue
        warped_fn = Path(dirname) / _get_temp_filename(p, idx, "_warped")
        warped_fn = Path(dirname) / f"{p.stem}_{idx}_warped.vrt"
        from_srs_name = ds.GetSpatialRef().GetName()
        to_srs_name = osr.SpatialReference(projection).GetName()
        logger.info(
            f"Reprojecting {p} from {from_srs_name} to match mode projection"
            f" {to_srs_name}"
        )
        warped_files.append(warped_fn)
        gdal.Warp(
            fspath(warped_fn),
            fspath(p),
            format="VRT",  # Just creates a file that will warp on the fly
            dstSRS=projection,
            resampleAlg=resample_alg,
            targetAlignedPixels=True,  # align in multiples of dx, dy
            xRes=res[0],
            yRes=res[1],
        )

    return warped_files


def _get_mode_projection(filenames: Iterable[Filename]) -> str:
    """Get the most common projection in the list."""
    projs = [gdal.Open(fspath(fn)).GetProjection() for fn in filenames]
    return max(set(projs), key=projs.count)


def _get_resolution(filenames: Iterable[Filename]) -> tuple[float, float]:
    """Get the most common resolution in the list."""
    gts = [gdal.Open(fspath(fn)).GetGeoTransform() for fn in filenames]
    res = [(dx, dy) for (_, dx, _, _, _, dy) in gts]
    if len(set(res)) > 1:
        msg = f"The input files have different resolutions: {res}. "
        raise ValueError(msg)
    return res[0]


def get_combined_bounds_nodata(
    *filenames: Filename,
    target_aligned_pixels: bool = False,
    out_bounds: Optional[Bbox] = None,
    out_bounds_epsg: Optional[int] = None,
    strides: Optional[dict[str, int]] = None,
) -> tuple[Bbox, Optional[float]]:
    """Get the bounds and nodata of the combined image.

    Parameters
    ----------
    filenames : list[Filename]
        list of filenames to combine
    target_aligned_pixels : bool
        if True, adjust output image bounds so that pixel coordinates
        are integer multiples of pixel size, matching the `-tap` GDAL option.
    out_bounds: Optional[Bbox]
        if provided, forces the output image bounds to
            (left, bottom, right, top).
        Otherwise, computes from the outside of all input images.
    out_bounds_epsg: Optional[int]
        The EPSG of `out_bounds`. If not provided, assumed to be the same
        as the EPSG of all `*filenames`.
    strides : dict[str, int]
        subsample factor: {"x": x strides, "y": y strides}

    Returns
    -------
    bounds : Bbox
        (min_x, min_y, max_x, max_y)
    nodata : float | None
        Nodata value of the input files

    Raises
    ------
    ValueError:
        If the inputs files have different resolutions/projections/nodata values

    """
    # scan input files
    if strides is None:
        strides = {"x": 1, "y": 1}
    xs = []
    ys = []
    resolutions = set()
    projs = set()
    nodatas = set()

    # Check all files match in resolution/projection
    for fn in filenames:
        ds = gdal.Open(fspath(fn))
        left, bottom, right, top = io.get_raster_bounds(fn)
        gt = ds.GetGeoTransform()
        dx, dy = gt[1], gt[5]

        resolutions.add((abs(dx), abs(dy)))  # dy is negative for north-up
        projs.add(ds.GetProjection())

        xs.extend([left, right])
        ys.extend([bottom, top])

        nd = io.get_raster_nodata(fn)
        # Need to stringify 'nan', or it is repeatedly added
        nodatas.add(str(nd) if (nd is not None and np.isnan(nd)) else nd)

    if len(resolutions) > 1:
        msg = f"The input files have different resolutions: {resolutions}. "
        raise ValueError(msg)
    if len(projs) > 1:
        msg = f"The input files have different projections: {projs}. "
        raise ValueError(msg)
    if len(nodatas) > 1:
        msg = f"The input files have different nodata values: {nodatas}. "
        raise ValueError(msg)
    res = (abs(dx) * strides["x"], abs(dy) * strides["y"])

    if out_bounds is not None:
        if out_bounds_epsg is not None:
            dst_epsg = io.get_raster_crs(filenames[0]).to_epsg()
            bounds = Bbox(*transform_bounds(out_bounds_epsg, dst_epsg, *out_bounds))
        else:
            bounds = out_bounds
    else:
        bounds = Bbox(min(xs), min(ys), max(xs), max(ys))

    if target_aligned_pixels:
        bounds = _align_bounds(bounds, res)

    nodata = next(iter(nodatas))
    # Convert back from string "nan"
    ndv: float | None = np.nan if nodata == "nan" else nodata  # type: ignore[assignment]
    return bounds, ndv


def _align_bounds(bounds: Bbox, res: tuple[float, float]) -> Bbox:
    """Align boundary with an integer multiple of the resolution."""
    left, bottom, right, top = bounds
    left = math.floor(left / res[0]) * res[0]
    right = math.ceil(right / res[0]) * res[0]
    bottom = math.floor(bottom / res[1]) * res[1]
    top = math.ceil(top / res[1]) * res[1]
    return Bbox(left, bottom, right, top)


def _copy_set_nodata(
    infile: Filename,
    outfile: Optional[Filename] = None,
    ext: Optional[str] = None,
    in_band: int = 1,
    out_nodata: float = 0,
    driver="ENVI",
    creation_options=io.DEFAULT_ENVI_OPTIONS,
):
    """Make a copy of infile and replace NaNs/input nodata with `out_nodata`."""
    in_p = Path(infile)
    if outfile is None:
        if ext is None:
            ext = in_p.suffix
        out_dir = in_p.parent
        outfile = out_dir / (in_p.stem + "_tmp" + ext)

    ds_in = gdal.Open(fspath(infile))
    drv = gdal.GetDriverByName(driver)
    ds_out = drv.CreateCopy(fspath(outfile), ds_in, options=creation_options)

    bnd = ds_in.GetRasterBand(in_band)
    nodata = bnd.GetNoDataValue()
    arr = bnd.ReadAsArray()
    # also make sure to replace NaNs, even if nodata is not set
    mask = np.logical_or(np.isnan(arr), arr == nodata)
    arr[mask] = out_nodata

    bnd1 = ds_out.GetRasterBand(1)
    bnd1.WriteArray(arr)
    bnd1.SetNoDataValue(out_nodata)
    ds_out = bnd1 = None

    return outfile


def warp_to_match(
    input_file: Filename,
    match_file: Filename,
    output_file: Optional[Filename] = None,
    resample_alg: str = "near",
    output_format: Optional[str] = None,
) -> Path:
    """Reproject `input_file` to align with the `match_file`.

    Uses the bounds, resolution, and CRS of `match_file`.

    Parameters
    ----------
    input_file: Filename
        Path to the image to be reprojected.
    match_file: Filename
        Path to the input image to serve as a reference for the reprojected image.
        Uses the bounds, resolution, and CRS of this image.
    output_file: Filename
        Path to the output, reprojected image.
        If None, creates an in-memory warped VRT using the `/vsimem/` protocol.
    resample_alg: str, optional, default = "near"
        Resampling algorithm to be used during reprojection.
        See https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r for choices.
    output_format: str, optional, default = None
        Output format to be used for the output image.
        If None, gdal will try to infer the format from the output file extension, or
        (if the extension of `output_file` matches `input_file`) use the input driver.

    Returns
    -------
    Path
        Path to the output image.
        Same as `output_file` if provided, otherwise a path to the in-memory VRT.

    """
    bounds = io.get_raster_bounds(match_file)
    crs_wkt = io.get_raster_crs(match_file).to_wkt()
    gt = io.get_raster_gt(match_file)
    resolution = (gt[1], gt[5])

    if output_file is None:
        output_file = f"/vsimem/warped_{Path(input_file).stem}.vrt"
        logger.debug(f"Creating in-memory warped VRT: {output_file}")

    if output_format is None and Path(input_file).suffix == Path(output_file).suffix:
        output_format = io.get_raster_driver(input_file)

    options = gdal.WarpOptions(
        dstSRS=crs_wkt,
        format=output_format,
        xRes=resolution[0],
        yRes=resolution[1],
        outputBounds=bounds,
        outputBoundsSRS=crs_wkt,
        resampleAlg=resample_alg,
    )
    gdal.Warp(
        fspath(output_file),
        fspath(input_file),
        options=options,
    )

    return Path(output_file)
