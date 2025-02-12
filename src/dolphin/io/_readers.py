from __future__ import annotations

import logging
import mmap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os import fspath
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Generator,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import h5py
import numpy as np
import rasterio as rio
from numpy.typing import ArrayLike
from opera_utils import get_dates, sort_files_by_date
from osgeo import gdal
from tqdm.auto import trange

from dolphin import io, utils
from dolphin._types import Filename
from dolphin.io._blocks import iter_blocks

from ._background import _DEFAULT_TIMEOUT, BackgroundReader
from ._paths import S3Path
from ._utils import _ensure_slices, _unpack_3d_slices

logger = logging.getLogger("dolphin")

__all__ = [
    "BinaryReader",
    "BinaryStackReader",
    "DatasetReader",
    "EagerLoader",
    "HDF5Reader",
    "HDF5StackReader",
    "RasterReader",
    "RasterStackReader",
    "StackReader",
    "VRTStack",
]


if TYPE_CHECKING:
    from dolphin._types import Index


@runtime_checkable
class DatasetReader(Protocol):
    """An array-like interface for reading input datasets.

    `DatasetReader` defines the abstract interface that types must conform to in order
    to be read by functions which iterate in blocks over the input data.
    Such objects must export NumPy-like `dtype`, `shape`, and `ndim` attributes,
    and must support NumPy-style slice-based indexing.

    Note that this protocol allows objects to be passed to `dask.array.from_array`
    which needs `.shape`, `.ndim`, `.dtype` and support numpy-style slicing.
    """

    dtype: np.dtype
    """numpy.dtype : Data-type of the array's elements."""

    shape: tuple[int, ...]
    """tuple of int : Tuple of array dimensions."""

    ndim: int
    """int : Number of array dimensions."""

    def __getitem__(self, key: tuple[Index, ...], /) -> ArrayLike:
        """Read a block of data."""
        ...


@runtime_checkable
class StackReader(DatasetReader, Protocol):
    """An array-like interface for reading a 3D stack of input datasets.

    `StackReader` defines the abstract interface that types must conform to in order
    to be valid inputs to be read in functions like [dolphin.ps.create_ps][].
    It is a specialization of [DatasetReader][] that requires a 3D shape.
    """

    ndim: int = 3
    """int : Number of array dimensions."""

    shape: tuple[int, int, int]
    """tuple of int : Tuple of array dimensions."""

    def __len__(self) -> int:
        """Int : Number of images in the stack."""
        return self.shape[0]


def _mask_array(arr: np.ndarray, nodata_value: float | None) -> np.ma.MaskedArray:
    """Mask an array based on a nodata value."""
    if np.isnan(nodata_value):
        return np.ma.masked_invalid(arr)
    return np.ma.masked_equal(arr, nodata_value)


@dataclass
class BinaryReader(DatasetReader):
    """A flat binary file for storing array data.

    See Also
    --------
    HDF5Dataset
    RasterReader

    Notes
    -----
    This class does not store an open file object. Instead, the file is opened on-demand
    for reading or writing and closed immediately after each read/write operation. This
    allows multiple spawned processes to write to the file in coordination (as long as a
    suitable mutex is used to guard file access.)

    """

    filename: Path
    """pathlib.Path : The file path."""

    shape: tuple[int, ...]
    """tuple of int : Tuple of array dimensions."""

    dtype: np.dtype
    """numpy.dtype : Data-type of the array's elements."""

    nodata: Optional[float] = None
    """Optional[float] : Value to use for nodata pixels."""

    def __post_init__(self):
        self.filename = Path(self.filename)
        if not self.filename.exists():
            msg = f"File {self.filename} does not exist."
            raise FileNotFoundError(msg)
        self.dtype = np.dtype(self.dtype)

    @property
    def ndim(self) -> int:  # type: ignore[override]
        """int : Number of array dimensions."""  # noqa: D403
        return len(self.shape)

    def __getitem__(self, key: tuple[Index, ...], /) -> np.ndarray:
        with self.filename.open("rb") as f:  # noqa: SIM117
            # Memory-map the entire file.
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                # In order to safely close the memory-map, there can't be any dangling
                # references to it, so we return a copy (not a view) of the requested
                # data and decref the array object.
                arr = np.frombuffer(mm, dtype=self.dtype).reshape(self.shape)
                data = arr[key].copy()
                del arr
        return _mask_array(data, self.nodata) if self.nodata is not None else data

    def __array__(self) -> np.ndarray:
        return self[:,]

    @classmethod
    def from_gdal(
        cls, filename: Filename, band: int = 1, nodata: Optional[float] = None
    ) -> BinaryReader:
        """Create a BinaryReader from a GDAL-readable file.

        Parameters
        ----------
        filename : Filename
            Path to the file to read.
        band : int, optional
            Band to read from the file, by default 1
        nodata : float, optional
            Value to use for nodata pixels, by default None
            If None passed, will search for a nodata value in the file.

        Returns
        -------
        BinaryReader
            The BinaryReader object.

        """
        with rio.open(filename) as src:
            dtype = src.dtypes[band - 1]
            shape = src.shape
            nodata = src.nodatavals[band - 1]
        return cls(
            Path(filename),
            shape=shape,
            dtype=dtype,
            nodata=nodata or nodata,
        )


@dataclass
class HDF5Reader(DatasetReader):
    """A Dataset in an HDF5 file.

    Attributes
    ----------
    filename : pathlib.Path | str
        Location of HDF5 file.
    dset_name : str
        Path to the dataset within the file.
    chunks : tuple[int, ...], optional
        Chunk shape of the dataset, or None if file is unchunked.
    keep_open : bool, optional (default False)
        If True, keep the HDF5 file handle open for faster reading.


    See Also
    --------
    BinaryReader
    RasterReader

    Notes
    -----
    If `keep_open=True`, this class does not store an open file object.
    Otherwise, the file is opened on-demand for reading or writing and closed
    immediately after each read/write operation.
    If passing the `HDF5Reader` to multiple spawned processes, it is recommended
    to set `keep_open=False` .

    """

    filename: Path
    """pathlib.Path : The file path."""

    dset_name: str
    """str : The path to the dataset within the file."""

    nodata: Optional[float] = None
    """Optional[float] : Value to use for nodata pixels.

    If None, looks for `_FillValue` or `missing_value` attributes on the dataset.
    """

    keep_open: bool = False
    """bool : If True, keep the HDF5 file handle open for faster reading."""

    keepdims: bool = True
    """bool : Maintain the dimension of the point array. If set to False, will
    skip `squeeze` on outputs with one dimension size of 1. Default is True."""

    def __post_init__(self):
        filename = Path(self.filename)

        hf = h5py.File(filename, "r")
        dset = hf[self.dset_name]
        self.shape = dset.shape
        self.dtype = dset.dtype
        self.chunks = dset.chunks
        if self.nodata is None:
            self.nodata = dset.attrs.get("_FillValue", None)
            if self.nodata is None:
                self.nodata = dset.attrs.get("missing_value", None)
        if self.keep_open:
            self._hf = hf
            self._dset = dset
        else:
            hf.close()

    @property
    def ndim(self) -> int:  # type: ignore[override]
        """Int : Number of array dimensions."""
        return len(self.shape)

    def __array__(self) -> np.ndarray:
        return self[:,]

    def __getitem__(self, key: tuple[Index, ...], /) -> np.ndarray:
        if self.keep_open:
            data = self._dset[key]
        else:
            with h5py.File(self.filename, "r") as f:
                data = f[self.dset_name][key]
        return _mask_array(data, self.nodata) if self.nodata is not None else data


@dataclass
class RasterReader(DatasetReader):
    """A single raster band of a GDAL-compatible dataset.

    See Also
    --------
    BinaryReader
    HDF5

    Notes
    -----
    If `keep_open=True`, this class does not store an open file object.
    Otherwise, the file is opened on-demand for reading or writing and closed
    immediately after each read/write operation.
    If passing the `RasterReader` to multiple spawned processes, it is recommended
    to set `keep_open=False` .

    """

    filename: Filename
    """Filename : The file path."""

    band: int
    """int : Band index (1-based)."""

    driver: str
    """str : Raster format driver name."""

    crs: rio.crs.CRS
    """rio.crs.CRS : The dataset's coordinate reference system."""

    transform: rio.transform.Affine
    """
    rasterio.transform.Affine : The dataset's georeferencing transformation matrix.

    This transform maps pixel row/column coordinates to coordinates in the dataset's
    coordinate reference system.
    """

    shape: tuple[int, int]
    dtype: np.dtype

    nodata: Optional[float] = None
    """Optional[float] : Value to use for nodata pixels."""

    keep_open: bool = False
    """bool : If True, keep the rasterio file handle open for faster reading."""

    chunks: Optional[tuple[int, int]] = None
    """Optional[tuple[int, int]] : Chunk shape of the dataset, or None if unchunked."""

    keepdims: bool = True
    """bool : Maintain the dimension of the point array. If set to False, will
    skip `squeeze` on outputs with one dimension size of 1. Default is True."""

    @classmethod
    def from_file(
        cls,
        filename: Filename,
        band: int = 1,
        nodata: Optional[float] = None,
        keepdims: bool = True,
        keep_open: bool = False,
        **options,
    ) -> RasterReader:
        with rio.open(filename, "r", **options) as src:
            shape = (src.height, src.width)
            dtype = np.dtype(src.dtypes[band - 1])
            driver = src.driver
            crs = src.crs
            nodata = nodata or src.nodatavals[band - 1]
            transform = src.transform
            chunks = src.block_shapes[band - 1]

            return cls(
                filename=filename,
                band=band,
                driver=driver,
                crs=crs,
                transform=transform,
                shape=shape,
                dtype=dtype,
                nodata=nodata,
                keepdims=keepdims,
                keep_open=keep_open,
                chunks=chunks,
            )

    def __post_init__(self):
        if self.keep_open:
            self._src = rio.open(self.filename, "r")

    @property
    def ndim(self) -> int:  # type: ignore[override]
        """Int : Number of array dimensions."""
        return 2

    def __array__(self) -> np.ndarray:
        return self[:, :]

    def __getitem__(self, key: tuple[Index, ...], /) -> np.ndarray:
        import rasterio.windows

        if key is ... or key == ():
            key = (slice(None), slice(None))

        if not isinstance(key, tuple):
            msg = "Index must be a tuple of slices or integers."
            raise TypeError(msg)

        r_slice, c_slice = _ensure_slices(*key[-2:])
        window = rasterio.windows.Window.from_slices(
            r_slice,
            c_slice,
            height=self.shape[0],
            width=self.shape[1],
        )
        if self.keep_open:
            out = self._src.read(self.band, window=window)

        with rio.open(self.filename) as src:
            out = src.read(self.band, window=window)
        out_masked = _mask_array(out, self.nodata) if self.nodata is not None else out
        # Note that Rasterio doesn't use the `step` of a slice, so we need to
        # manually slice the output array.
        r_step, c_step = r_slice.step or 1, c_slice.step or 1
        if self.keepdims:
            return out_masked[::r_step, ::c_step]
        else:
            return out_masked[::r_step, ::c_step].squeeze()


def _read_3d(
    key: tuple[Index, ...],
    readers: Sequence[DatasetReader],
    num_threads: int = 1,
    keepdims: bool = True,
):
    bands, r_slice, c_slice = _unpack_3d_slices(key)

    if isinstance(bands, slice):
        # convert the bands to -1-indexed list
        total_num_bands = len(readers)
        band_idxs = list(range(*bands.indices(total_num_bands)))
    elif isinstance(bands, int):
        band_idxs = [bands]
    else:
        msg = "Band index must be an integer or slice."
        raise TypeError(msg)

    # Get only the bands we need
    if num_threads == 1:
        out = np.stack([readers[i][r_slice, c_slice] for i in band_idxs], axis=0)
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = executor.map(lambda i: readers[i][r_slice, c_slice], band_idxs)
        out = np.stack(list(results), axis=0)

    return out if keepdims else np.squeeze(out)


@dataclass
class BaseStackReader(StackReader):
    """Base class for stack readers."""

    file_list: Sequence[Filename]
    readers: Sequence[DatasetReader]
    keepdims: bool = True
    num_threads: int = 1
    nodata: Optional[float] = None

    def __getitem__(self, key: tuple[Index, ...], /) -> np.ndarray:
        return _read_3d(
            key, self.readers, num_threads=self.num_threads, keepdims=self.keepdims
        )

    @property
    def shape_2d(self):
        return self.readers[0].shape

    @property
    def shape(self):
        return (len(self.file_list), *self.shape_2d)

    @property
    def dtype(self):
        return self.readers[0].dtype


@dataclass
class BinaryStackReader(BaseStackReader):
    @classmethod
    def from_file_list(
        cls, file_list: Sequence[Filename], shape_2d: tuple[int, int], dtype: np.dtype
    ) -> BinaryStackReader:
        """Create a BinaryStackReader from a list of files.

        Parameters
        ----------
        file_list : Sequence[Filename]
            List of paths to the files to read.
        shape_2d : tuple[int, int]
            Shape of each file.
        dtype : np.dtype
            Data type of each file.

        Returns
        -------
        BinaryStackReader
            The BinaryStackReader object.

        """
        readers = [
            BinaryReader(Path(f), shape=shape_2d, dtype=dtype) for f in file_list
        ]
        return cls(file_list=file_list, readers=readers, num_threads=1)

    @classmethod
    def from_gdal(
        cls,
        file_list: Sequence[Filename],
        band: int = 1,
        num_threads: int = 1,
        nodata: Optional[float] = None,
    ) -> BinaryStackReader:
        """Create a BinaryStackReader from a list of GDAL-readable files.

        Parameters
        ----------
        file_list : Sequence[Filename]
            List of paths to the files to read.
        band : int, optional
            Band to read from the file, by default 1
        num_threads : int, optional (default 1)
            Number of threads to use for reading.
        nodata : float, optional
            Manually set value to use for nodata pixels, by default None
            If None passed, will search for a nodata value in the file.

        Returns
        -------
        BinaryStackReader
            The BinaryStackReader object.

        """
        readers = []
        dtypes = set()
        shapes = set()
        for f in file_list:
            with rio.open(f) as src:
                dtypes.add(src.dtypes[band - 1])
                shapes.add(src.shape)
            if len(dtypes) > 1:
                msg = "All files must have the same data type."
                raise ValueError(msg)
            if len(shapes) > 1:
                msg = "All files must have the same shape."
                raise ValueError(msg)
            readers.append(BinaryReader.from_gdal(f, band=band))
        return cls(
            file_list=file_list,
            readers=readers,
            num_threads=num_threads,
            nodata=nodata,
        )


@dataclass
class HDF5StackReader(BaseStackReader):
    """A stack of datasets in an HDF5 file.

    See Also
    --------
    BinaryStackReader
    StackReader

    Notes
    -----
    If `keep_open=True`, this class stores an open file object.
    Otherwise, the file is opened on-demand for reading or writing and closed
    immediately after each read/write operation.
    If passing the `HDF5StackReader` to multiple spawned processes, it is recommended
    to set `keep_open=False`.

    """

    @classmethod
    def from_file_list(
        cls,
        file_list: Sequence[Filename],
        dset_names: str | Sequence[str],
        keep_open: bool = False,
        num_threads: int = 1,
        nodata: Optional[float] = None,
    ) -> HDF5StackReader:
        """Create a HDF5StackReader from a list of files.

        Parameters
        ----------
        file_list : Sequence[Filename]
            List of paths to the files to read.
        dset_names : str | Sequence[str]
            Name of the dataset to read from each file.
            If a single string, will be used for all files.
        keep_open : bool, optional (default False)
            If True, keep the HDF5 file handles open for faster reading.
        num_threads : int, optional (default 1)
            Number of threads to use for reading.
        nodata : float, optional
            Manually set value to use for nodata pixels, by default None
            If None passed, will search for a nodata value in the file.

        Returns
        -------
        HDF5StackReader
            The HDF5StackReader object.

        """
        if isinstance(dset_names, str):
            dset_names = [dset_names] * len(file_list)

        readers = [
            HDF5Reader(Path(f), dset_name=dn, keep_open=keep_open, nodata=nodata)
            for (f, dn) in zip(file_list, dset_names)
        ]
        # Check if nodata values were found in the files
        nds = {r.nodata for r in readers}
        if len(nds) == 1:
            nodata = nds.pop()

        return cls(file_list, readers, num_threads=num_threads, nodata=nodata)


@dataclass
class RasterStackReader(BaseStackReader):
    """A stack of datasets for any GDAL-readable rasters.

    See Also
    --------
    BinaryStackReader
    HDF5StackReader

    Notes
    -----
    If `keep_open=True`, this class stores an open file object.
    Otherwise, the file is opened on-demand for reading or writing and closed
    immediately after each read/write operation.

    """

    @classmethod
    def from_file_list(
        cls,
        file_list: Sequence[Filename],
        bands: int | Sequence[int] = 1,
        keepdims: bool = True,
        keep_open: bool = False,
        num_threads: int = 1,
        nodata: Optional[float] = None,
    ) -> RasterStackReader:
        """Create a RasterStackReader from a list of files.

        Parameters
        ----------
        file_list : Sequence[Filename]
            List of paths to the files to read.
        bands : int | Sequence[int]
            Band to read from each file.
            If a single int, will be used for all files.
            Default = 1.
        keepdims : bool
            Maintain the dimension of the point array. If set to False, will
            skip `squeeze` on outputs with one dimension size of 1.
            Default is False.
        keep_open : bool, optional (default False)
            If True, keep the rasterio file handles open for faster reading.
        num_threads : int, optional (default 1)
            Number of threads to use for reading.
        nodata : float, optional
            Manually set value to use for nodata pixels, by default None

        Returns
        -------
        RasterStackReader
            The RasterStackReader object.

        """
        if isinstance(bands, int):
            bands = [bands] * len(file_list)

        readers = [
            RasterReader.from_file(f, band=b, keep_open=keep_open, keepdims=keepdims)
            for (f, b) in zip(file_list, bands)
        ]
        # Check if nodata values were found in the files
        nds = {r.nodata for r in readers}
        if len(nds) == 1:
            nodata = nds.pop()
        return cls(
            file_list,
            readers,
            num_threads=num_threads,
            nodata=nodata,
            keepdims=keepdims,
        )


# Masked versions of each of the 2D/3D readers


class VRTStack(StackReader):
    """Class for creating a virtual stack of raster files.

    Attributes
    ----------
    file_list : list[Filename]
        Paths or GDAL-compatible strings (NETCDF:...) for paths to files.
    outfile : pathlib.Path, optional (default = Path("slc_stack.vrt"))
        Name of output file to write
    dates : list[list[DateOrDatetime]]
        list, where each entry is all dates matched from the corresponding file
        in `file_list`. This is used to sort the files by date.
        Each entry is a list because some files (compressed SLCs) may have
        multiple dates in the filename.
    use_abs_path : bool, optional (default = True)
        Write the filepaths of the SLCs in the VRT as "relative=0"
    subdataset : str, optional
        Subdataset to use from the files in `file_list`, if using NetCDF files.
    sort_files : bool, optional (default = True)
        Sort the files in `file_list`. Assumes that the naming convention
        will sort the files in increasing time order.
    file_date_fmt : str, optional (default = "%Y%m%d")
        Format string for parsing the dates from the filenames.
        Passed to [opera_utils.get_dates][].

    """

    def __init__(
        self,
        file_list: Sequence[Filename],
        outfile: Filename = "slc_stack.vrt",
        use_abs_path: bool = True,
        subdataset: Optional[str] = None,
        sort_files: bool = True,
        file_date_fmt: str = "%Y%m%d",
        write_file: bool = True,
        fail_on_overwrite: bool = False,
        skip_size_check: bool = False,
        num_threads: int = 1,
        read_masked: bool = False,
    ):
        if Path(outfile).exists() and write_file:
            if fail_on_overwrite:
                msg = (
                    f"Output file {outfile} already exists. "
                    "Please delete or specify a different output file. "
                    "To create from an existing VRT, use the `from_vrt_file` method."
                )
                raise FileExistsError(msg)
            else:
                logger.debug(f"Overwriting {outfile}")

        # files: list[Filename] = [Path(f) for f in file_list]
        self._use_abs_path = use_abs_path
        files: list[Filename | S3Path]
        if any(str(f).startswith("s3://") for f in file_list):
            files = [S3Path(str(f)) for f in file_list]
        elif use_abs_path:
            files = [utils._resolve_gdal_path(p) for p in file_list]
        else:
            files = list(file_list)
        # Extract the date/datetimes from the filenames
        dates = [get_dates(f, fmt=file_date_fmt) for f in file_list]
        if sort_files:
            files, dates = sort_files_by_date(files, file_date_fmt=file_date_fmt)

        # Save the attributes
        self.file_list = files
        self.dates = dates
        self.num_threads = num_threads
        self._read_masked = read_masked

        self.outfile = Path(outfile).resolve()
        # Assumes that all files use the same subdataset (if NetCDF)
        self.subdataset = subdataset

        if not skip_size_check:
            _assert_images_same_size(self._gdal_file_strings)

        # Use the first file in the stack to get size, transform info
        ds = gdal.Open(fspath(self._gdal_file_strings[0]))
        bnd1 = ds.GetRasterBand(1)
        self.xsize = ds.RasterXSize
        self.ysize = ds.RasterYSize
        self.nodatavals = []
        for i in range(1, ds.RasterCount + 1):
            bnd = ds.GetRasterBand(i)
            self.nodatavals.append(bnd.GetNoDataValue())
        self.nodata = self.nodatavals[0]
        # Should be CFloat32
        self.gdal_dtype = gdal.GetDataTypeName(bnd1.DataType)
        # Save these for setting at the end
        self.gt = ds.GetGeoTransform()
        self.proj = ds.GetProjection()
        self.srs = ds.GetSpatialRef()
        ds = bnd1 = None
        # Save the subset info

        self.xoff, self.yoff = 0, 0
        self.xsize_sub, self.ysize_sub = self.xsize, self.ysize

        if write_file:
            self._write()

    def _write(self):
        """Write out the VRT file pointing to the stack of SLCs, erroring if exists."""
        with open(self.outfile, "w") as fid:
            fid.write(
                f'<VRTDataset rasterXSize="{self.xsize_sub}"'
                f' rasterYSize="{self.ysize_sub}">\n'
            )

            for idx, filename in enumerate(self._gdal_file_strings, start=1):
                chunk_size = io.get_raster_chunk_size(filename)
                # chunks in a vrt have a min of 16, max of 2**14=16384
                # https://github.com/OSGeo/gdal/blob/2530defa1e0052827bc98696e7806037a6fec86e/frmts/vrt/vrtrasterband.cpp#L339
                if any(b < 16 for b in chunk_size) or any(
                    b > 16384 for b in chunk_size
                ):
                    chunk_str = ""
                else:
                    chunk_str = (
                        f'blockXSize="{chunk_size[0]}" blockYSize="{chunk_size[1]}"'
                    )
                outstr = f"""  <VRTRasterBand dataType="{self.gdal_dtype}" band="{idx}" {chunk_str}>
    <SimpleSource>
      <SourceFilename>{filename}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="{self.xoff}" yOff="{self.yoff}" xSize="{self.xsize_sub}" ySize="{self.ysize_sub}"/>
      <DstRect xOff="0" yOff="0" xSize="{self.xsize_sub}" ySize="{self.ysize_sub}"/>
    </SimpleSource>
  </VRTRasterBand>\n"""  # noqa: E501
                fid.write(outstr)

            fid.write("</VRTDataset>")

        # Set the georeferencing metadata
        ds = gdal.Open(fspath(self.outfile), gdal.GA_Update)
        ds.SetGeoTransform(self.gt)
        ds.SetProjection(self.proj)
        ds.SetSpatialRef(self.srs)
        if self.nodata is not None:
            for i in range(ds.RasterCount):
                # ds.GetRasterBand(i + 1).SetNoDataValue(self.nodatavals[i])
                # Force to be the same nodataval for all bands
                ds.GetRasterBand(i + 1).SetNoDataValue(self.nodata)

        ds = None

    @property
    def _gdal_file_strings(self) -> list[str]:
        """Get the GDAL-compatible paths to write to the VRT.

        If we're not using .h5 or .nc, this will just be the file_list as is.
        """
        out = []
        for f in self.file_list:
            if isinstance(f, S3Path):
                out.append(f.to_gdal())
            else:
                out.append(io.format_nc_filename(f, self.subdataset))
        return out

    def __fspath__(self):
        # Allows os.fspath() to work on the object, enabling rasterio.open()
        return fspath(self.outfile)

    @classmethod
    def from_vrt_file(cls, vrt_file, new_outfile=None, **kwargs):
        """Create a new VRTStack using an existing VRT file."""
        file_list, subdataset = _parse_vrt_file(vrt_file)
        if new_outfile is None:
            # Point to the same, if none provided
            new_outfile = vrt_file

        return cls(
            file_list,
            outfile=new_outfile,
            subdataset=subdataset,
            write_file=False,
            **kwargs,
        )

    @property
    def shape(self):
        """Get the 3D shape of the stack."""
        xsize, ysize = io.get_raster_xysize(self._gdal_file_strings[0])
        return (len(self.file_list), ysize, xsize)

    def __len__(self):
        return len(self.file_list)

    def __repr__(self):
        outname = fspath(self.outfile) if self.outfile else "(not written)"
        return f"VRTStack({len(self.file_list)} bands, outfile={outname})"

    def __eq__(self, other):
        if not isinstance(other, VRTStack):
            return False
        return (
            self._gdal_file_strings == other._gdal_file_strings
            and self.outfile == other.outfile
        )

    @property
    def ndim(self):
        return 3

    @property
    def dtype(self):
        return io.get_raster_dtype(self._gdal_file_strings[0])

    def __getitem__(self, index):
        if isinstance(index, int):
            if index < 0:
                index = len(self) + index
            return self.read_stack(band=index + 1)

        # TODO: raise an error if they try to skip like [::2, ::2]
        # or pass it to read_stack... but I dont think I need to support it.
        n, rows, cols = index
        if isinstance(rows, int):
            rows = slice(rows, rows + 1)
        if isinstance(cols, int):
            cols = slice(cols, cols + 1)
        if isinstance(n, int):
            if n < 0:
                n = len(self) + n
            return self.read_stack(band=n + 1, rows=rows, cols=cols)
        elif n is ...:
            n = slice(None)

        bands = list(range(1, 1 + len(self)))[n]
        if len(bands) == len(self):
            # This will use gdal's ds.ReadAsRaster, no iteration needed
            data = self.read_stack(band=None, rows=rows, cols=cols)
        else:
            # Get only the bands we need
            if self.num_threads == 1:
                data = np.stack(
                    [self.read_stack(band=i, rows=rows, cols=cols) for i in bands],
                    axis=0,
                )
            else:
                with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                    results = executor.map(
                        lambda i: self.read_stack(band=i, rows=rows, cols=cols), bands
                    )
                data = np.stack(list(results), axis=0)

        return data

    def read_stack(
        self,
        band: Optional[int] = None,
        subsample_factor: int = 1,
        rows: Optional[slice] = None,
        cols: Optional[slice] = None,
        masked: bool | None = None,
        keepdims: bool = True,
    ):
        """Read in the SLC stack."""
        if masked is None:
            masked = self._read_masked
        data = io.load_gdal(
            self.outfile,
            band=band,
            subsample_factor=subsample_factor,
            rows=rows,
            cols=cols,
            masked=masked,
        )
        # Check to get around gdal `ds.ReadAsArray()` squashing dimensions
        if len(self) == 1 and keepdims:
            # Add the front (1,) dimension which is missing for a single file
            data = data[None]
        return data


def _parse_vrt_file(vrt_file):
    """Extract the filenames, and possible subdatasets, from a .vrt file.

    Assumes, if using HDFS/NetCDF files, that the subdataset is the same.

    Note that we are parsing the XML, not using `GetFilelist`, because the
    function does not seem to work when using HDF5 files. E.g.

        <SourceFilename ="1">NETCDF:20220111.nc:SLC/VV</SourceFilename>

    This would not get added to the result of `GetFilelist`

    Parameters
    ----------
    vrt_file : Filename
        Path to the VRT file to read.

    Returns
    -------
    filepaths
        List of filepaths to the SLCs
    sds
        Subdataset name, if using NetCDF/HDF5 files

    """
    file_strings = []
    with open(vrt_file) as f:
        for line in f:
            if "<SourceFilename" not in line:
                continue
            # Get the middle part of < >filename</ >
            fn = line.split(">")[1].strip().split("<")[0]
            file_strings.append(fn)

    sds = ""
    filepaths = []
    for name in file_strings:
        if name.upper().startswith("HDF5:") or name.upper().startswith("NETCDF:"):
            prefix, filename, subdataset = name.split(":")
            # Clean up subdataset
            sds = subdataset.replace('"', "").replace("'", "").lstrip("/")
            # Remove quoting if it was present
            filepaths.append(filename.replace('"', "").replace("'", ""))
        else:
            filepaths.append(name)

    return filepaths, sds


def _assert_images_same_size(files):
    """Ensure all files are the same size."""
    with ThreadPoolExecutor(5) as executor:
        sizes = list(executor.map(io.get_raster_xysize, files))
    if len(set(sizes)) > 1:
        msg = f"Not all files have the same raster (x, y) size:\n{set(sizes)}"
        raise ValueError(msg)


class EagerLoader(BackgroundReader):
    """Class to pre-fetch data chunks in a background thread."""

    def __init__(
        self,
        reader: DatasetReader,
        block_shape: tuple[int, int],
        overlaps: tuple[int, int] = (0, 0),
        skip_empty: bool = True,
        nodata_value: Optional[float] = None,
        nodata_mask: Optional[ArrayLike] = None,
        queue_size: int = 1,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        super().__init__(nq=queue_size, timeout=timeout, name="EagerLoader")
        self.reader = reader
        # Set up the generator of ((row_start, row_end), (col_start, col_end))
        # convert the slice generator to a list so we have the size
        nrows, ncols = self.reader.shape[-2:]
        self.slices = list(
            iter_blocks(
                arr_shape=(nrows, ncols),
                block_shape=block_shape,
                overlaps=overlaps,
            )
        )
        if nodata_value is None:
            nodata_value = getattr(reader, "nodata", None)
        self._queue_size = queue_size
        self._skip_empty = skip_empty
        self._nodata_mask = nodata_mask
        self._block_shape = block_shape
        self._nodata = nodata_value
        if self._nodata is None:
            self._nodata = np.nan

    def read(self, rows: slice, cols: slice) -> tuple[np.ndarray, tuple[slice, slice]]:
        logger.debug(f"EagerLoader reading {rows}, {cols}")
        cur_block = self.reader[..., rows, cols]
        return cur_block, (rows, cols)

    def iter_blocks(
        self, **tqdm_kwargs
    ) -> Generator[tuple[np.ndarray, tuple[slice, slice]], None, None]:
        # Queue up all slices to the work queue
        queued_slices = []
        for rows, cols in self.slices:
            # Skip queueing a read if all nodata
            if self._skip_empty and self._nodata_mask is not None:
                logger.debug("Checking nodata mask")
                if self._nodata_mask[rows, cols].all():
                    logger.debug("Skipping!")
                    continue
            self.queue_read(rows, cols)
            queued_slices.append((rows, cols))

        logger.info(f"Processing {self._block_shape} sized blocks...")
        for _ in trange(len(queued_slices), **tqdm_kwargs):
            cur_block, (rows, cols) = self.get_data()
            logger.debug(f"got data for {rows, cols}: {cur_block.shape}")

            # Otherwise look at the actual block we loaded
            if self._skip_empty:
                if isinstance(cur_block, np.ma.MaskedArray) and cur_block.mask.all():
                    continue
                if np.isnan(self._nodata):
                    block_is_nodata = np.isnan(cur_block)
                else:
                    block_is_nodata = cur_block == self._nodata
                if np.all(block_is_nodata):
                    logger.debug(
                        f"Skipping block {rows}, {cols} since it was all nodata"
                    )
                    continue
            yield cur_block, (rows, cols)

        self.notify_finished()
