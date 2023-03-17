"""Module for creating the OPERA output product in NetCDF format."""
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pyproj
from numpy.typing import ArrayLike
from osgeo import gdal, osr

from dolphin._types import Filename
from dolphin.io import DEFAULT_HDF5_OPTIONS

BASE_GROUP = "/science/SENTINEL1"
DISP_GROUP = f"{BASE_GROUP}/DISP"
CORRECTIONS_GROUP = f"{BASE_GROUP}/corrections"
GLOBAL_ATTRS = dict(
    Conventions="CF-1.8",
    contact="operaops@jpl.nasa.gov",
    institution="NASA JPL",
    mission_name="OPERA",
    reference_document="TBD",
    title="OPERA L3_DISP_S1 Product",
)
GRID_MAPPING_DSET = "spatial_ref"


def _create_xy_dsets(
    group: h5py.Group, gt: List[float], shape: Tuple[int, int]
) -> Tuple[h5py.Dataset, h5py.Dataset]:
    """Create the x and y coordinate datasets."""
    ysize, xsize = shape
    # Parse the geotransform
    x_origin, x_res, _, y_origin, _, y_res = gt

    # Make the x/y arrays
    # Note that these are the center of the pixels, whereas the GeoTransform
    # is the upper left corner of the top left pixel.
    x = np.arange(x_origin + x_res / 2, x_origin + x_res * xsize, x_res)
    y = np.arange(y_origin + y_res / 2, y_origin + y_res * ysize, y_res)

    # Create the datasets
    x_ds = group.create_dataset("x_coordinates", data=x, dtype=float)
    y_ds = group.create_dataset("y_coordinates", data=y, dtype=float)

    for name, ds in zip(["x", "y"], [x_ds, y_ds]):
        ds.make_scale(name)
        ds.attrs["standard_name"] = f"projection_{name}_coordinate"
        ds.attrs["long_name"] = f"{name} coordinate of projection"
        ds.attrs["units"] = "m"

    return x_ds, y_ds


def _create_grid_mapping(group, crs: pyproj.CRS, gt: List[float]) -> h5py.Dataset:
    """Set up the grid mapping variable."""
    # https://github.com/corteva/rioxarray/blob/21284f67db536d9c104aa872ab0bbc261259e59e/rioxarray/rioxarray.py#L34
    dset = group.create_dataset(GRID_MAPPING_DSET, data=np.array(0), dtype=int)
    dset.attrs.update(crs.to_cf())
    # Also add the GeoTransform
    gt_string = " ".join([str(x) for x in gt])
    dset.attrs["GeoTransform"] = gt_string
    return dset


def _create_correction_dsets(
    corrections_group: h5py.Group, corrections: Dict[str, ArrayLike]
):
    """Create datasets for the tropospheric/ionospheric/other corrections."""
    troposphere = corrections.get("troposphere")
    if troposphere:
        troposphere_dset = corrections_group.create_dataset(
            "troposphere", data=troposphere, **DEFAULT_HDF5_OPTIONS
        )
        troposphere_dset.attrs["grid_mapping"] = "crs"

    ionosphere = corrections["ionosphere"]
    if ionosphere:
        # Write the ionosphere correction image
        ionosphere_dset = corrections_group.create_dataset(
            "ionosphere", data=ionosphere, **DEFAULT_HDF5_OPTIONS
        )
        ionosphere_dset.attrs["grid_mapping"] = "crs"


def create_output_product(
    filename: Filename,
    output_name="output.nc",
    corrections: Optional[Dict[str, ArrayLike]] = None,
):
    """Create the OPERA output product in NetCDF format.

    Parameters
    ----------
    filename : str
        The path to the input displacement image.
    output_name : str, optional
        The path to the output NetCDF file, by default "output.nc"
    corrections : Dict[str, ArrayLike], optional
        A dictionary of corrections to write to the output file, by default None
    """
    # Read the Geotiff file and its metadata
    displacement_ds = gdal.Open(filename)
    gt = displacement_ds.GetGeoTransform()
    crs = pyproj.CRS.from_wkt(displacement_ds.GetProjection())
    arr = displacement_ds.ReadAsArray()
    displacement_ds = None

    srs = osr.SpatialReference()
    srs.ImportFromWkt(displacement_ds.GetProjection())

    # Create the NetCDF file
    with h5py.File(output_name, "w") as f:
        f.attrs.update(GLOBAL_ATTRS)

        # Create the '/science/SENTINEL1/DISP/grids/displacement' group
        displacement_group = f.create_group(DISP_GROUP)

        # Set up the grid mapping variable
        _create_grid_mapping(displacement_group, crs, gt)

        # Set up the X/Y variables
        x_ds, y_ds = _create_xy_dsets(displacement_group, gt, arr.shape)

        # Write the displacement array
        displacement_dset = displacement_group.create_dataset(
            "displacement", data=arr, **DEFAULT_HDF5_OPTIONS
        )
        displacement_dset.attrs["grid_mapping"] = GRID_MAPPING_DSET
        # Attach the X/Y coordinates
        displacement_dset.dims[0].attach_scale(y_ds)
        displacement_dset.dims[1].attach_scale(x_ds)

        # Create the '/science/SENTINEL1/DISP/corrections' group
        corrections_group = f.require_group(CORRECTIONS_GROUP)
        if corrections:
            # Write the tropospheric/ionospheric correction images (if they exist)
            _create_correction_dsets(corrections_group, corrections)
