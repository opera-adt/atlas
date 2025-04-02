"""Functions for reading from and writing to raster files.

This module heavily relies on GDAL and provides many convenience/
wrapper functions to write/iterate over blocks of large raster files.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from os import fspath
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import h5py
import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from osgeo import gdal
from pyproj import CRS

from dolphin._types import Bbox, Filename, Strides
from dolphin.utils import compute_out_shape, gdal_to_numpy_type, numpy_to_gdal_type

from ._paths import S3Path

gdal.UseExceptions()
logger = logging.getLogger("dolphin")

__all__ = [
    "DEFAULT_DATETIME_FORMAT",
    "DEFAULT_ENVI_OPTIONS",
    "DEFAULT_HDF5_OPTIONS",
    "DEFAULT_TIFF_OPTIONS",
    "DEFAULT_TILE_SHAPE",
    "copy_projection",
    "format_nc_filename",
    "get_raster_bounds",
    "get_raster_bounds",
    "get_raster_chunk_size",
    "get_raster_crs",
    "get_raster_description",
    "get_raster_driver",
    "get_raster_dtype",
    "get_raster_gt",
    "get_raster_metadata",
    "get_raster_nodata",
    "get_raster_units",
    "get_raster_xysize",
    "load_gdal",
    "set_raster_description",
    "set_raster_metadata",
    "set_raster_nodata",
    "set_raster_units",
    "write_arr",
    "write_block",
]


DEFAULT_DATETIME_FORMAT = "%Y%m%d"
DEFAULT_TILE_SHAPE = [128, 128]
# For use in rasterio
DEFAULT_TIFF_OPTIONS_RIO = {
    "compress": "lzw",
    "zlevel": 4,
    "bigtiff": "yes",
    "tiled": "yes",
    "interleave": "band",
    "blockxsize": DEFAULT_TILE_SHAPE[1],
    "blockysize": DEFAULT_TILE_SHAPE[0],
}
# For gdal's bindings
DEFAULT_TIFF_OPTIONS = tuple(
    f"{k.upper()}={v}" for k, v in DEFAULT_TIFF_OPTIONS_RIO.items()
)

DEFAULT_ENVI_OPTIONS = ("SUFFIX=ADD",)
DEFAULT_HDF5_OPTIONS = {
    # https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline
    "chunks": DEFAULT_TILE_SHAPE,
    "compression": "gzip",
    "compression_opts": 4,
    "shuffle": True,
}


def _get_gdal_ds(filename: Filename, update: bool = False) -> gdal.Dataset:
    mode = gdal.GA_Update if update else gdal.GA_ReadOnly
    if str(filename).startswith("s3://"):
        return gdal.Open(S3Path(str(filename)).to_gdal(), mode)
    return gdal.Open(fspath(filename))


def load_gdal(
    filename: Filename,
    *,
    band: Optional[int] = None,
    subsample_factor: Union[int, tuple[int, int]] = 1,
    rows: Optional[slice] = None,
    cols: Optional[slice] = None,
) -> np.ndarray:
    """Load a gdal file into a numpy array.

    Parameters
    ----------
    filename : str or Path
        Path to the file to load.
    band : int, optional
        Band to load. If None, load all bands as 3D array.
    subsample_factor : int or tuple[int, int], optional
        Subsample the data by this factor. Default is 1 (no subsampling).
        Uses nearest neighbor resampling.
    rows : slice, optional
        Rows to load. Default is None (load all rows).
    cols : slice, optional
        Columns to load. Default is None (load all columns).

    Returns
    -------
    arr : np.ndarray
        Array of shape (bands, y, x) or (y, x) if `band` is specified,
        where y = height // subsample_factor and x = width // subsample_factor.

    """
    ds = _get_gdal_ds(filename)
    nrows, ncols = ds.RasterYSize, ds.RasterXSize

    # if rows or cols are not specified, load all rows/cols
    rows = slice(0, nrows) if rows in (None, slice(None)) else rows
    cols = slice(0, ncols) if cols in (None, slice(None)) else cols
    # Help out mypy:
    assert rows is not None
    assert cols is not None

    dt = gdal_to_numpy_type(ds.GetRasterBand(1).DataType)

    if isinstance(subsample_factor, int):
        subsample_factor = (subsample_factor, subsample_factor)

    xoff, yoff = int(cols.start), int(rows.start)
    row_stop = min(rows.stop, nrows)
    col_stop = min(cols.stop, ncols)
    xsize, ysize = int(col_stop - cols.start), int(row_stop - rows.start)
    if xsize <= 0 or ysize <= 0:
        msg = (
            f"Invalid row/col slices: {rows}, {cols} for file {filename} of size"
            f" {nrows}x{ncols}"
        )
        raise IndexError(msg)
    nrows_out, ncols_out = (
        ysize // subsample_factor[0],
        xsize // subsample_factor[1],
    )

    # Read the data, and decimate if specified
    resamp = gdal.GRA_NearestNeighbour
    if band is None:
        count = ds.RasterCount
        out = np.empty((count, nrows_out, ncols_out), dtype=dt)
        ds.ReadAsArray(xoff, yoff, xsize, ysize, buf_obj=out, resample_alg=resamp)
        if count == 1:
            out = out[0]
    else:
        out = np.empty((nrows_out, ncols_out), dtype=dt)
        bnd = ds.GetRasterBand(band)
        bnd.ReadAsArray(xoff, yoff, xsize, ysize, buf_obj=out, resample_alg=resamp)

    return out


def load_masked(
    filename: Filename,
    *,
    band: Optional[int] = None,
    subsample_factor: Union[int, tuple[int, int]] = 1,
    rows: Optional[slice] = None,
    cols: Optional[slice] = None,
) -> np.ma.MaskedArray:
    """Load a gdal file into a numpy array, and mask the nodata value.

    Parameters
    ----------
    filename : str or Path
        Path to the file to load.
    band : int, optional
        Band to load. If None, load all bands as 3D array.
    subsample_factor : int or tuple[int, int], optional
        Subsample the data by this factor. Default is 1 (no subsampling).
        Uses nearest neighbor resampling.
    rows : slice, optional
        Rows to load. Default is None (load all rows).
    cols : slice, optional
        Columns to load. Default is None (load all columns).

    Returns
    -------
    arr : np.ma.MaskedArray
        Array of shape (bands, y, x) or (y, x) if `band` is specified,
        where y = height // subsample_factor and x = width // subsample_factor.
        The nodata value is masked.

    """
    data = load_gdal(
        filename,
        band=band,
        subsample_factor=subsample_factor,
        rows=rows,
        cols=cols,
    )
    nd = get_raster_nodata(filename)
    if nd is not None and np.isnan(nd):
        return np.ma.masked_invalid(data)
    else:
        return np.ma.masked_equal(data, nd)


def load_overview(filename: Filename, *, level: int = 1) -> np.ndarray:
    """Load an overview from a gdal file into a numpy array.

    Parameters
    ----------
    filename : str or Path
        Path to the file to load.
    level : int, optional
        Level of the overview to load.

    Returns
    -------
    arr : np.ndarray
        Array of shape (bands, y, x) or (y, x) if `band` is specified,
        where y = height // 2**level and x = width // 2**level.

    Raises
    ------
    FileNotFoundError
        If the overview doesn't exist.

    """
    ds = _get_gdal_ds(filename)
    bnd = ds.GetRasterBand(1)
    ovr_count = bnd.GetOverviewCount()
    assert isinstance(ovr_count, int)
    if ovr_count == 0:
        msg = f"No overviews found for {filename}"
        raise ValueError(msg)
    idx = ovr_count + level if level < 0 else level
    if idx < 0 or idx >= ovr_count:
        msg = (
            f"Requested {level = }, but only {ovr_count} overviews found for {filename}"
        )
        raise ValueError(msg)
    out = bnd.GetOverview(idx).ReadAsArray()
    bnd = ds = None
    return out


def format_nc_filename(filename: Filename, ds_name: Optional[str] = None) -> str:
    """Format an HDF5/NetCDF filename with dataset for reading using GDAL.

    If `filename` is already formatted, or if `filename` is not an HDF5/NetCDF
    file (based on the file extension), it is returned unchanged.

    Parameters
    ----------
    filename : str or PathLike
        Filename to format.
    ds_name : str, optional
        Dataset name to use. If not provided for a .h5 or .nc file, an error is raised.

    Returns
    -------
    str
        Formatted filename.

    Raises
    ------
    ValueError
        If `ds_name` is not provided for a .h5 or .nc file.

    """
    # If we've already formatted the filename, return it
    fname_clean = fspath(filename).lstrip('"').lstrip("'").rstrip('"').rstrip("'")
    if fname_clean.startswith(("NETCDF:", "HDF5:")):
        return fspath(filename)

    if not (fname_clean.endswith((".nc", ".h5"))):
        return fspath(filename)

    # Now we're definitely dealing with an HDF5/NetCDF file
    if ds_name is None:
        msg = "Must provide dataset name for HDF5/NetCDF files"
        raise ValueError(msg)

    return f'NETCDF:"{filename}":"//{ds_name.lstrip("/")}"'


def copy_projection(src_file: Filename, dst_file: Filename) -> None:
    """Copy projection/geotransform from `src_file` to `dst_file`."""
    ds_src = _get_gdal_ds(src_file)
    projection = ds_src.GetProjection()
    geotransform = ds_src.GetGeoTransform()
    nodata = ds_src.GetRasterBand(1).GetNoDataValue()

    if projection is None and geotransform is None:
        logger.info("No projection or geotransform found on file %s", input)
        return
    ds_dst = gdal.Open(fspath(dst_file), gdal.GA_Update)

    if geotransform is not None and geotransform != (0, 1, 0, 0, 0, 1):
        ds_dst.SetGeoTransform(geotransform)

    if projection is not None and projection != "":
        ds_dst.SetProjection(projection)

    if nodata is not None:
        ds_dst.GetRasterBand(1).SetNoDataValue(nodata)

    ds_src = ds_dst = None


def get_raster_xysize(filename: Filename) -> tuple[int, int]:
    """Get the xsize/ysize of a GDAL-readable raster."""
    ds = _get_gdal_ds(filename)
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    ds = None
    return xsize, ysize


def get_raster_nodata(filename: Filename, band: int = 1) -> Optional[float]:
    """Get the nodata value from a file.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.
    band : int, optional
        Band to get nodata value for, by default 1.

    Returns
    -------
    Optional[float]
        Nodata value, or None if not found.

    """
    ds = _get_gdal_ds(filename)
    return ds.GetRasterBand(band).GetNoDataValue()


def set_raster_nodata(filename: Filename, nodata: float, band: int | None = None):
    """Set the nodata value for a raster.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.
    nodata : float
        The nodata value to set.
    band : int, optional
        The band to set the nodata value for, by default None
        (sets the nodata value for all bands).

    """
    ds = gdal.Open(fspath(filename), gdal.GA_Update)
    if band is None:
        for i in range(ds.RasterCount):
            ds.GetRasterBand(i + 1).SetNoDataValue(nodata)
    else:
        ds.GetRasterBand(band).SetNoDataValue(nodata)
    ds = None


def get_raster_crs(filename: Filename) -> CRS:
    """Get the CRS from a file.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.

    Returns
    -------
    CRS
        CRS.

    """
    ds = _get_gdal_ds(filename)
    return CRS.from_wkt(ds.GetProjection())


def get_raster_gt(filename: Filename) -> list[float]:
    """Get the geotransform from a file.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.

    Returns
    -------
    List[float]
        6 floats representing a GDAL Geotransform.

    """
    ds = _get_gdal_ds(filename)
    return ds.GetGeoTransform()


def get_raster_dtype(filename: Filename) -> np.dtype:
    """Get the data type from a file.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.

    Returns
    -------
    np.dtype
        Data type.

    """
    ds = _get_gdal_ds(filename)
    return gdal_to_numpy_type(ds.GetRasterBand(1).DataType)


def get_raster_driver(filename: Filename) -> str:
    """Get the GDAL driver `ShortName` from a file.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.

    Returns
    -------
    str
        Driver name.

    """
    ds = _get_gdal_ds(filename)
    return ds.GetDriver().ShortName


def get_raster_bounds(
    filename: Optional[Filename] = None, ds: Optional[gdal.Dataset] = None
) -> Bbox:
    """Get the (left, bottom, right, top) bounds of the image."""
    if ds is None:
        if filename is None:
            msg = "Must provide either `filename` or `ds`"
            raise ValueError(msg)
        ds = _get_gdal_ds(filename)

    gt = ds.GetGeoTransform()
    xsize, ysize = ds.RasterXSize, ds.RasterYSize

    left, top = _apply_gt(gt=gt, x=0, y=0)
    right, bottom = _apply_gt(gt=gt, x=xsize, y=ysize)

    return Bbox(left, bottom, right, top)


def get_raster_metadata(filename: Filename, domain: str = ""):
    """Get metadata from a raster file.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.
    domain : str, optional
        Domain to get metadata for. Default is "" (all domains).

    Returns
    -------
    dict
        Dictionary of metadata.

    """
    ds = _get_gdal_ds(filename)
    return ds.GetMetadata(domain)


def set_raster_metadata(
    filename: Filename, metadata: Mapping[str, Any], domain: str = ""
):
    """Set metadata on a raster file.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.
    metadata : dict
        Dictionary of metadata to set.
    domain : str, optional
        Domain to set metadata for. Default is "" (all domains).

    """
    ds = gdal.Open(fspath(filename), gdal.GA_Update)
    # Ensure the keys/values are written as strings
    md_dict = {k: str(v) for k, v in metadata.items()}
    ds.SetMetadata(md_dict, domain)
    ds.FlushCache()
    ds = None


def get_raster_description(filename: Filename, band: int = 1):
    """Get description of a raster band.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.
    band : int, optional
        Band to get description for. Default is 1.

    """
    ds = _get_gdal_ds(filename)
    bnd = ds.GetRasterBand(band)
    return bnd.GetDescription()


def set_raster_description(filename: Filename, description: str, band: int = 1):
    """Set description on a raster band.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.
    description : str
        Description to set.
    band : int, optional
        Band to set description for. Default is 1.

    """
    ds = gdal.Open(fspath(filename), gdal.GA_Update)
    bnd = ds.GetRasterBand(band)
    bnd.SetDescription(description)
    bnd.FlushCache()
    ds = None


def get_raster_units(filename: Filename, band: int = 1) -> str | None:
    """Get units of a raster band.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.
    band : int
        Band to get units for.
        Default is 1.

    """
    ds = _get_gdal_ds(filename)
    bnd = ds.GetRasterBand(band)
    return bnd.GetUnitType() or None


def set_raster_units(filename: Filename, units: str, band: int | None = None) -> None:
    """Set units on a raster band.

    Parameters
    ----------
    filename : Filename
        Path to the file to load.
    units : str
        Units to set.
    band : int, optional
        Band to set units for. Default is None, which sets for all bands.

    """
    ds = gdal.Open(fspath(filename), gdal.GA_Update)
    if band is None:
        bands = range(1, ds.RasterCount + 1)
    for i in bands:
        bnd = ds.GetRasterBand(i)
        bnd.SetUnitType(units)
        bnd.FlushCache()


def rowcol_to_xy(
    row: int,
    col: int,
    ds: Optional[gdal.Dataset] = None,
    filename: Optional[Filename] = None,
) -> tuple[float, float]:
    """Convert indexes in the image space to georeferenced coordinates."""
    return _apply_gt(ds, filename, col, row)


def xy_to_rowcol(
    x: float,
    y: float,
    ds: Optional[gdal.Dataset] = None,
    filename: Optional[Filename] = None,
    do_round=True,
) -> tuple[int, int]:
    """Convert coordinates in the georeferenced space to a row and column index."""
    col, row = _apply_gt(ds, filename, x, y, inverse=True)
    # Need to convert to int, otherwise we get a float
    if do_round:
        # round up to the nearest pixel, instead of banker's rounding
        row = math.floor(row + 0.5)
        col = math.floor(col + 0.5)
    return int(row), int(col)


def _apply_gt(
    ds=None, filename=None, x=None, y=None, inverse=False, gt=None
) -> tuple[float, float]:
    """Read the (possibly inverse) geotransform, apply to the x/y coordinates."""
    if gt is None:
        if ds is None:
            ds = _get_gdal_ds(filename)
            gt = ds.GetGeoTransform()
            ds = None
        else:
            gt = ds.GetGeoTransform()

    if inverse:
        gt = gdal.InvGeoTransform(gt)
    # Reference: https://gdal.org/tutorials/geotransforms_tut.html
    x = gt[0] + x * gt[1] + y * gt[2]
    y = gt[3] + x * gt[4] + y * gt[5]
    return x, y


def write_arr(
    *,
    arr: Optional[ArrayLike],
    output_name: Filename,
    like_filename: Optional[Filename] = None,
    driver: Optional[str] = "GTiff",
    options: Optional[Sequence] = None,
    nbands: Optional[int] = None,
    shape: Optional[tuple[int, int]] = None,
    dtype: Optional[DTypeLike] = None,
    geotransform: Optional[Sequence[float]] = None,
    strides: Optional[dict[str, int]] = None,
    projection: Optional[Any] = None,
    nodata: Optional[float] = None,
    units: Optional[str] = None,
    description: Optional[str] = None,
):
    """Save an array to `output_name`.

    If `like_filename` if provided, copies the projection/nodata.
    Options can be overridden by passing `driver`/`nbands`/`dtype`.

    If arr is None, create an empty file with the same x/y shape as `like_filename`.

    Parameters
    ----------
    arr : ArrayLike, optional
        Array to save. If None, create an empty file.
    output_name : str or Path
        Path to save the file to.
    like_filename : str or Path, optional
        Path to a file to copy raster shape/metadata from.
    driver : str, optional
        GDAL driver to use. Default is "GTiff".
    options : list, optional
        list of options to pass to the driver. Default is DEFAULT_TIFF_OPTIONS.
    nbands : int, optional
        Number of bands to save. Default is 1.
    shape : tuple, optional
        (rows, cols) of desired output file.
        Overrides the shape of the output file, if using `like_filename`.
    dtype : DTypeLike, optional
        Data type to save. Default is `arr.dtype` or the datatype of like_filename.
    geotransform : list, optional
        Geotransform to save. Default is the geotransform of like_filename.
        See https://gdal.org/tutorials/geotransforms_tut.html .
    strides : dict, optional
        If using `like_filename`, used to change the pixel size of the output file.
        {"x": x strides, "y": y strides}
    projection : str or int, optional
        Projection to save. Default is the projection of like_filename.
        Possible values are anything parse-able by ``pyproj.CRS.from_user_input``
        (including EPSG ints, WKT strings, PROJ strings, etc.)
    nodata : float, optional
        Nodata value to save.
        Default is the nodata of band 1 of `like_filename` (if provided), or None.
    units : str, optional
        Units of the data. Default is None.
        Value is stored in the metadata as "units".
    description : str, optional
        Description of the raster bands stored in the metadata.

    """
    fi = FileInfo.from_user_inputs(
        arr=arr,
        output_name=output_name,
        like_filename=like_filename,
        driver=driver,
        options=options,
        nbands=nbands,
        shape=shape,
        dtype=dtype,
        geotransform=geotransform,
        strides=strides,
        projection=projection,
        nodata=nodata,
    )
    drv = gdal.GetDriverByName(fi.driver)
    ds_out = drv.Create(
        fspath(output_name),
        fi.xsize,
        fi.ysize,
        fi.nbands,
        fi.gdal_dtype,
        options=fi.options,
    )

    # Set the geo/proj information
    if fi.projection:
        # Make sure we're got a correct format for the projection
        # this still works if we're passed a WKT string
        proj = CRS.from_user_input(fi.projection).to_wkt()
        ds_out.SetProjection(proj)

    if fi.geotransform is not None:
        ds_out.SetGeoTransform(fi.geotransform)

    # Set the nodata/units/description for each band
    for i in range(fi.nbands):
        logger.debug(f"Setting nodata for band {i + 1}/{fi.nbands}")
        bnd = ds_out.GetRasterBand(i + 1)
        # Note: right now we're assuming the nodata/units/description
        if fi.nodata is not None:
            bnd.SetNoDataValue(fi.nodata)
        if units is not None:
            bnd.SetUnitType(units)
        if description is not None:
            bnd.SetDescription(description)

    # Write the actual data
    if arr is not None:
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        for i in range(fi.nbands):
            logger.debug(f"Writing band {i + 1}/{fi.nbands}")
            bnd = ds_out.GetRasterBand(i + 1)
            bnd.WriteArray(arr[i])

    ds_out.FlushCache()
    ds_out = None


def write_block(
    cur_block: NDArray,
    filename: Filename,
    row_start: int,
    col_start: int,
    band: int | None = None,
    dset: str | None = None,
):
    """Write out an ndarray to a subset of the pre-made `filename`.

    Parameters
    ----------
    cur_block : ArrayLike
        2D or 3D data array
    filename : Filename
        list of output files to save to, or (if cur_block is 2D) a single file.
    row_start : int
        Row index to start writing at.
    col_start : int
        Column index to start writing at.
    band : int, optional
        Raster band to write to within `filename`.
        If None, writes to band 1 (for 2D), or all bands if `cur_block.ndim = 3`.
    dset : str
        (For writing to HDF5/NetCDF files) The name of the string dataset
        withing `filename` to write to.

    Raises
    ------
    ValueError
        If length of `output_files` does not match length of `cur_block`.

    """
    if cur_block.ndim == 2 and band is None:
        # Make into 3D array shaped (1, rows, cols)
        cur_block = cur_block[np.newaxis, ...]
    # filename must be pre-made
    filename = Path(filename)
    if not filename.exists():
        msg = f"File {filename} does not exist"
        raise ValueError(msg)

    if filename.suffix in (".h5", ".hdf5", ".nc"):
        if dset is None:
            raise ValueError("Missing `dset` argument for writing to HDF5")
        _write_hdf5(cur_block, filename, row_start, col_start, dset)
    else:
        _write_gdal(cur_block, filename, row_start, col_start, band)


def _write_gdal(
    cur_block: NDArray,
    filename: Filename,
    row_start: int,
    col_start: int,
    band: int | None,
):
    ds = gdal.Open(fspath(filename), gdal.GA_Update)
    if band is not None:
        bnd = ds.GetRasterBand(band)
        bnd.WriteArray(cur_block, col_start, row_start)
        bnd = None
    else:
        for b_idx, cur_image in enumerate(cur_block, start=1):
            bnd = ds.GetRasterBand(b_idx)
            # only need offset for write:
            # https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.Band.WriteArray
            bnd.WriteArray(cur_image, col_start, row_start)
            bnd.FlushCache()
            bnd = None
    ds = None


def _write_hdf5(
    cur_block: NDArray,
    filename: Filename,
    row_start: int,
    col_start: int,
    dset: str,
):
    nrows, ncols = cur_block.shape[-2:]
    row_slice = slice(row_start, row_start + nrows)
    col_slice = slice(col_start, col_start + ncols)
    with h5py.File(filename, "a") as hf:
        ds = hf[dset]
        ds.write_direct(
            cur_block, source_sel=None, dest_sel=np.s_[row_slice, col_slice]
        )


@dataclass
class FileInfo:
    nbands: int
    ysize: int
    xsize: int
    dtype: DTypeLike
    gdal_dtype: int
    nodata: Optional[float]
    driver: str
    options: Optional[list]
    projection: Optional[str]
    geotransform: Optional[list[float]]

    @classmethod
    def from_user_inputs(
        cls,
        *,
        arr: Optional[ArrayLike],
        output_name: Filename,
        like_filename: Optional[Filename] = None,
        driver: Optional[str] = "GTiff",
        options: Optional[Sequence[Any]] = [],
        nbands: Optional[int] = None,
        shape: Optional[tuple[int, int]] = None,
        dtype: Optional[DTypeLike] = None,
        geotransform: Optional[Sequence[float]] = None,
        strides: Optional[dict[str, int]] = None,
        projection: Optional[Any] = None,
        nodata: Optional[float] = None,
    ) -> FileInfo:
        ds_like = _get_gdal_ds(like_filename) if like_filename is not None else None

        xsize = ysize = gdal_dtype = None
        if arr is not None:
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            ysize, xsize = arr.shape[-2:]
            gdal_dtype = numpy_to_gdal_type(arr.dtype)
        else:
            # If not passing an array to write, get shape/dtype from like_filename
            if shape is not None:
                ysize, xsize = shape
            else:
                if ds_like is None:
                    raise ValueError("Must provide shape if no `like_filename`")
                xsize, ysize = ds_like.RasterXSize, ds_like.RasterYSize
                # If using strides, adjust the output shape
                if strides is not None:
                    ysize, xsize = compute_out_shape(
                        (ysize, xsize), Strides(strides["y"], strides["x"])
                    )

            if dtype is not None:
                gdal_dtype = numpy_to_gdal_type(dtype)
            else:
                if ds_like is None:
                    raise ValueError("Must provide dtype if no `like_filename`")
                gdal_dtype = ds_like.GetRasterBand(1).DataType

        if any(v is None for v in (xsize, ysize, gdal_dtype)):
            msg = "Must specify either `arr` or `like_filename`"
            raise ValueError(msg)
        assert gdal_dtype is not None

        if nodata is None and ds_like is not None:
            b = ds_like.GetRasterBand(1)
            nodata = b.GetNoDataValue()

        if nbands is None:
            if arr is not None:
                nbands = arr.shape[0]
            elif ds_like is not None:
                nbands = ds_like.RasterCount
            else:
                nbands = 1

        if driver is None:
            if str(output_name).endswith(".tif"):
                driver = "GTiff"
            else:
                if not ds_like:
                    msg = "Must specify `driver` if `like_filename` is None"
                    raise ValueError(msg)
                driver = ds_like.GetDriver().ShortName
        if options is None and driver == "GTiff":
            options = list(DEFAULT_TIFF_OPTIONS)
        if not options:
            options = []

        # If not provided, attempt to get projection/geotransform from like_filename
        if projection is None and ds_like is not None:
            projection = ds_like.GetProjection()
        if geotransform is None and ds_like is not None:
            geotransform = ds_like.GetGeoTransform()
            # If we're using strides, adjust the geotransform
            if strides is not None:
                geotransform = list(geotransform)
                geotransform[1] *= strides["x"]
                geotransform[5] *= strides["y"]

        return cls(
            nbands=nbands,
            ysize=ysize,
            xsize=xsize,
            dtype=dtype,
            gdal_dtype=gdal_dtype,
            nodata=nodata,
            driver=driver,
            options=list(options),
            projection=projection,
            geotransform=list(geotransform) if geotransform else None,
        )


def get_raster_chunk_size(filename: Filename) -> list[int]:
    """Get size the raster's chunks on disk.

    This is called blockXsize, blockYsize by GDAL.
    """
    ds = _get_gdal_ds(filename)
    block_size = ds.GetRasterBand(1).GetBlockSize()
    for i in range(2, ds.RasterCount + 1):
        if block_size != ds.GetRasterBand(i).GetBlockSize():
            logger.warning(f"Warning: {filename} bands have different block shapes.")
            break
    return block_size
