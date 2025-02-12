from __future__ import annotations

import logging
import tempfile
from enum import IntEnum
from os import fspath
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from osgeo import gdal
from pyproj import CRS, Transformer
from shapely import from_wkt, geometry, ops, to_geojson

from dolphin import io
from dolphin._types import Bbox, PathOrStr

gdal.UseExceptions()

logger = logging.getLogger("dolphin")


class MaskConvention(IntEnum):
    """Enum for masking conventions to indicate the nodata value as 0 or 1.

    In numpy, 1 indicates nodata
    https://numpy.org/doc/stable/reference/maskedarray.generic.html#what-is-a-masked-array

    In SNAPHU (and isce2, and some water masks), the binary mask you pass has the
    opposite convention, where 0s mean the pixel is invalid (nodata).
    """

    ZERO_IS_NODATA = 0
    ONE_IS_NODATA = 1

    # Semantic aliases for the above:
    SNAPHU = 0
    NUMPY = 1

    def get_nodata_value(self) -> int:
        """Return the nodata value for this convention."""
        return int(not self.value)


def combine_mask_files(
    mask_files: Sequence[PathOrStr],
    output_file: PathOrStr,
    dtype: str = "uint8",
    output_convention: MaskConvention = MaskConvention.ZERO_IS_NODATA,
    input_conventions: Optional[Sequence[MaskConvention]] = None,
    combine_method: str = "any",
):
    """Combine multiple mask files into a single mask file.

    All `mask_files` must be the same size and projected on the same grid.

    Parameters
    ----------
    mask_files : list of Path or str
        list of mask files to combine.
    output_file : PathOrStr
        Path to the combined output file.
    dtype : str, optional
        Data type of output file. Default is uint8.
    output_convention : MaskConvention, optional
        Convention to use for output mask. Default is SNAPHU,
        where 0 indicates invalid pixels.
    input_conventions : list of MaskConvention, optional
        Convention to use for each input mask. Default is None,
        where it is assumed all masks use the "0 is invalid" convention.
    combine_method : str, optional, default = 'any', choices = ['any', 'all']
        Logical operation to use to combine masks. Default is 'any',
        which means the output is masked where *any* of the input masks indicated
        a masked pixel (the masked region grows larger).
        If 'all', the only pixels masked are those in which *all* input masks
        indicated a masked pixel (the masked region shrinks).

    Raises
    ------
    ValueError
        If `input_conventions` passed and is different length as `mask_files`
        If all mask_files are not the same shape

    """
    output_file = Path(output_file)
    xsize, ysize = io.get_raster_xysize(mask_files[0])

    if combine_method not in ["any", "all"]:
        msg = "combine_method must be 'any' or 'all'"
        raise ValueError(msg)

    if input_conventions is None:
        input_conventions = [MaskConvention.ZERO_IS_NODATA] * len(mask_files)
    elif isinstance(input_conventions, MaskConvention):
        input_conventions = [input_conventions] * len(mask_files)

    if len(input_conventions) != len(mask_files):
        msg = (
            f"input_conventions ({len(input_conventions)}) must have the same"
            f" length as mask_files ({len(mask_files)})"
        )
        raise ValueError(msg)

    # Uses the numpy convention (1 = invalid, 0 = valid) for combining logic
    # Loop through mask files and update the total mask
    if combine_method == "any":
        # "any" will use `logical_or` to grow the region starting empty region (as 0s)
        mask_total = np.zeros((ysize, xsize), dtype=bool)
    elif combine_method == "all":
        # "and" uses `logical_and` to shrink the full starting region (as 1s)
        mask_total = np.ones((ysize, xsize), dtype=bool)

    for input_convention, mask_file in zip(input_conventions, mask_files):
        # TODO: if we separate input missing data from mask 1/0, this changes
        mask = io.load_gdal(mask_file, masked=True).astype(bool)
        # Fill with "mask" value
        mask = mask.filled(bool(input_convention.value))
        if input_convention != MaskConvention.NUMPY:
            mask = ~mask

        if combine_method == "any":
            mask_total = np.logical_or(mask_total, mask)
        elif combine_method == "all":
            mask_total = np.logical_and(mask_total, mask)

    # Convert to output convention
    if output_convention == MaskConvention.SNAPHU:
        mask_total = ~mask_total

    io.write_arr(
        arr=mask_total.astype(dtype),
        output_name=output_file,
        like_filename=mask_files[0],
    )


def load_mask_as_numpy(mask_file: PathOrStr) -> np.ndarray:
    """Load `mask_file` and convert it to a NumPy boolean array.

    This function reads a mask file where 0 represents invalid data and 1 represents
    good data. It converts the mask to a boolean numpy array where True values
    indicate missing data (nodata) pixels, following the numpy masking convention.

    Parameters
    ----------
    mask_file : PathOrStr
        Path to the mask file. Can be a string or a Path-like object.

    Returns
    -------
    np.ndarray
        A boolean numpy array where True values indicate nodata (invalid) pixels
        and False values indicate valid data pixels.

    Notes
    -----
    The input mask file is expected to use 0 for invalid data and 1 for good data.
    The output mask follows the numpy masking convention where True indicates
    nodata and False indicates valid data.

    """
    # The mask file will by have 0s at invalid data, 1s at good
    nodata_mask = io.load_gdal(mask_file, masked=True).astype(bool).filled(False)
    # invert the mask so Trues are the missing data pixels
    nodata_mask = ~nodata_mask
    return nodata_mask


def create_bounds_mask(
    bounds: Bbox | tuple[float, float, float, float] | None,
    bounds_wkt: str | None,
    output_filename: PathOrStr,
    like_filename: PathOrStr,
    bounds_epsg: int = 4326,
    overwrite: bool = False,
) -> None:
    """Create a boolean raster mask where 1 is inside the given bounds and 0 is outside.

    Parameters
    ----------
    bounds : tuple, optional
        (min x, min y, max x, max y) of the area of interest
    bounds_wkt : tuple, optional
        Well known text (WKT) string describing Polygon of the area of interest.
        Alternative to bounds. Cannot pass both bounds and bounds_wkt.
    like_filename : Filename
        Reference file to copy the shape, extent, and projection.
    output_filename : Filename
        Output filename for the mask
    bounds_epsg : int, optional
        EPSG code of the coordinate system of the bounds.
        Default is 4326 (lat/lon coordinates for the bounds).
    overwrite : bool, optional
        Overwrite the output file if it already exists, by default False

    Raises
    ------
    ValueError
        If neither bounds nor bounds_wkt, or both bounds and bounds_wkt, is passed.

    """
    if bounds is None and bounds_wkt is None:
        raise ValueError("Must pass either `bounds` or `bounds_wkt`")
    if bounds is not None and bounds_wkt is not None:
        raise ValueError("Cannot pass both `bounds` and `bounds_wkt`")

    if Path(output_filename).exists():
        if not overwrite:
            logger.info(f"Skipping {output_filename} since it already exists.")
            return
        else:
            logger.info(f"Overwriting {output_filename} since overwrite=True.")
            Path(output_filename).unlink()

    # Create a polygon from the bounds or wkt
    bounds_poly = geometry.box(*bounds) if bounds is not None else from_wkt(bounds_wkt)

    # Geojson default is 4326, so we need it in that system
    # Transform bounds if necessary
    if bounds_epsg != 4326:
        # Pass the Pyproj transformer to `shapely.ops.transform`
        transformer = Transformer.from_crs(
            CRS.from_epsg(bounds_epsg), 4326, always_xy=True
        )
        bounds_poly_lonlat = ops.transform(transformer.transform, bounds_poly)
    else:
        bounds_poly_lonlat = bounds_poly

    logger.info(f"Creating mask for bounds {bounds_poly_lonlat}")

    # Create the output raster
    io.write_arr(
        arr=None,
        output_name=output_filename,
        dtype=bool,
        nbands=1,
        like_filename=like_filename,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_vector_file = Path(tmpdir) / "temp.geojson"
        with open(temp_vector_file, "w") as f:
            f.write(to_geojson(bounds_poly_lonlat))

        # Open the input vector file
        src_ds = gdal.OpenEx(fspath(temp_vector_file), gdal.OF_VECTOR)
        dst_ds = gdal.Open(fspath(output_filename), gdal.GA_Update)

        # Now burn in the union of all polygons
        gdal.Rasterize(dst_ds, src_ds, burnValues=[1])

    logger.info(f"Created {output_filename}")
