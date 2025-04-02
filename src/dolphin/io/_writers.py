from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

import numpy as np
import rasterio
import rasterio.errors
from numpy.typing import ArrayLike, DTypeLike
from rasterio.windows import Window
from typing_extensions import Self

from dolphin._types import Filename

from ._background import BackgroundWriter
from ._utils import _unpack_3d_slices, round_mantissa

__all__ = [
    "BackgroundBlockWriter",
    "BackgroundRasterWriter",
    "BackgroundStackWriter",
    "DatasetStackWriter",
    "DatasetWriter",
    "RasterWriter",
]

if TYPE_CHECKING:
    from dolphin._types import Index


class BackgroundBlockWriter(BackgroundWriter):
    """Class to write data to multiple files in the background using `gdal` bindings."""

    def __init__(
        self,
        *,
        max_queue: int = 0,
        debug: bool = False,
        keep_bits: int | None = None,
        **kwargs,
    ):
        super().__init__(nq=max_queue, name="Writer")
        self.keep_bits = keep_bits
        if debug:
            #  background thread. Just synchronously write data
            self.notify_finished()
            self.queue_write = self.write  # type: ignore[assignment]

    def write(
        self,
        data: ArrayLike,
        filename: Filename,
        row_start: int,
        col_start: int,
        band: int | None = None,
    ):
        """Write out an ndarray to a subset of the pre-made `filename`.

        Parameters
        ----------
        data : ArrayLike
            2D or 3D data array to save.
        filename : Filename
            list of output files to save to, or (if cur_block is 2D) a single file.
        row_start : int
            Row index to start writing at.
        col_start : int
            Column index to start writing at.
        band : int, optional
            Band index in the file to write. Defaults to None, which uses first band,
            or whole raster for 3D data.

        Raises
        ------
        ValueError
            If length of `output_files` does not match length of `cur_block`.

        """
        from dolphin.io import write_block

        if np.issubdtype(data.dtype, np.floating) and self.keep_bits is not None:
            round_mantissa(data, keep_bits=self.keep_bits)

        write_block(data, filename, row_start, col_start, band=band)


@runtime_checkable
class DatasetWriter(Protocol):
    """An array-like interface for writing output datasets.

    `DatasetWriter` defines the abstract interface that types must conform to in order
    to be used by functions which write outputs in blocks.
    Such objects must export NumPy-like `dtype`, `shape`, and `ndim` attributes,
    and must support NumPy-style slice-based indexing for setting data..
    """

    dtype: np.dtype
    """numpy.dtype : Data-type of the array's elements."""

    shape: tuple[int, ...]
    """tuple of int : Tuple of array dimensions."""

    ndim: int
    """int : Number of array dimensions."""

    def __setitem__(self, key: tuple[Index, ...], value: np.ndarray, /) -> None:
        """Write a block of data."""
        ...


@runtime_checkable
class DatasetStackWriter(Protocol):
    """An array-like interface for writing output datasets.

    `DatasetWriter` defines the abstract interface that types must conform to in order
    to be used by functions which write outputs in blocks.
    Such objects must export NumPy-like `dtype`, `shape`, and `ndim` attributes,
    and must support NumPy-style slice-based indexing for setting data..
    """

    dtype: np.dtype
    """numpy.dtype : Data-type of the array's elements."""

    shape: tuple[int, ...]
    """tuple of int : Tuple of array dimensions."""

    ndim: int = 3
    """int : Number of array dimensions."""

    def __setitem__(self, key: tuple[Index, ...], value: np.ndarray, /) -> None:
        """Write a block of data."""
        ...


RasterT = TypeVar("RasterT", bound="RasterWriter")


@dataclass
class RasterWriter(DatasetWriter, AbstractContextManager["RasterWriter"]):
    """A single raster band in a GDAL-compatible dataset containing one or more bands.

    `Raster` provides a convenient interface for using SNAPHU to unwrap ground-projected
    interferograms in raster formats supported by the Geospatial Data Abstraction
    Library (GDAL). It acts as a thin wrapper around a Rasterio dataset and a band
    index, providing NumPy-like access to the underlying raster data.

    Data access is performed lazily -- the raster contents are not stored in memory
    unless/until they are explicitly accessed by an indexing operation.

    `Raster` objects must be closed after use in order to ensure that any data written
    to them is flushed to disk and any associated file objects are closed. The `Raster`
    class implements Python's context manager protocol, which can be used to reliably
    ensure that the raster is closed upon exiting the context manager.
    """

    filename: Filename
    """str or Path : Path to the file to write."""
    band: int = 1
    """int : Band index in the file to write."""
    keep_bits: int | None = None
    """int : For floating point rasters, the number of mantissa bits to keep."""

    def __post_init__(self) -> None:
        # Open the dataset.
        self.dataset = rasterio.open(self.filename, mode="r+")

        # Check that `band` is a valid band index in the dataset.
        nbands = self.dataset.count
        if not (1 <= self.band <= nbands):
            errmsg = (
                f"band index {self.band} out of range: dataset contains {nbands} raster"
                " band(s)"
            )
            raise IndexError(errmsg)

        self.ndim = 2

    @classmethod
    def create(
        cls,
        fp: Filename,
        width: int | None = None,
        height: int | None = None,
        dtype: DTypeLike | None = None,
        driver: str | None = None,
        crs: str | Mapping[str, str] | rasterio.crs.CRS | None = None,
        transform: rasterio.transform.Affine | None = None,
        *,
        like_filename: Filename | None = None,
        keep_bits: int | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a new single-band raster dataset.

        If another raster is passed via the `like` argument, the new dataset will
        inherit the shape, data-type, driver, coordinate reference system (CRS), and
        geotransform of the reference raster. Driver-specific dataset creation options
        such as chunk size and compression flags may also be inherited.

        All other arguments take precedence over `like` and may be used to override
        attributes of the reference raster when creating the new raster.

        Parameters
        ----------
        fp : str or path-like
            File system path or URL of the local or remote dataset.
        width, height : int or None, optional
            The numbers of columns and rows of the raster dataset. Required if `like` is
            not specified. Otherwise, if None, the new dataset is created with the same
            width/height as `like`. Defaults to None.
        dtype : data-type or None, optional
            Data-type of the raster dataset's elements. Must be convertible to a
            `numpy.dtype` object and must correspond to a valid GDAL datatype. Required
            if `like` is not specified. Otherwise, if None, the new dataset is created
            with the same data-type as `like`. Defaults to None.
        driver : str or None, optional
            Raster format driver name. If None, the method will attempt to infer the
            driver from the file extension. Defaults to None.
        crs : str, dict, rasterio.crs.CRS, or None; optional
            The coordinate reference system. If None, the CRS of `like` will be used, if
            available, otherwise the raster will not be georeferenced. Defaults to None.
        transform : rasterio.transform.Affine or None, optional
            Affine transformation mapping the pixel space to geographic space. If None,
            the geotransform of `like` will be used, if available, otherwise the default
            transform will be used. Defaults to None.
        like_filename : Raster or None, optional
            An optional reference raster. If not None, the new raster will be created
            with the same metadata (shape, data-type, driver, CRS/geotransform, etc) as
            the reference raster. All other arguments will override the corresponding
            attribute of the reference raster. Defaults to None.
        keep_bits : int, optional
            Number of bits to preserve in mantissa. Defaults to None.
            Lower numbers will truncate the mantissa more and enable more compression.
        **kwargs : dict, optional
            Additional driver-specific creation options passed to `rasterio.open`.

        """
        if like_filename is not None:
            with rasterio.open(like_filename) as dataset:
                kwargs = dataset.profile | kwargs

        if width is not None:
            kwargs["width"] = width
        if height is not None:
            kwargs["height"] = height
        if dtype is not None:
            kwargs["dtype"] = np.dtype(dtype)
        if driver is not None:
            kwargs["driver"] = driver
        if crs is not None:
            kwargs["crs"] = crs
        if transform is not None:
            kwargs["transform"] = transform

        # Always create a single-band dataset, even if `like` was part of a multi-band
        # dataset.
        kwargs["count"] = 1

        # Create the new single-band dataset.
        with rasterio.open(fp, mode="w+", **kwargs):
            pass

        return cls(fp, band=1, keep_bits=keep_bits)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.dataset.dtypes[self.band - 1])

    @property
    def height(self) -> int:
        """int : The number of rows in the raster."""  # noqa: D403
        return self.dataset.height  # type: ignore[no-any-return]

    @property
    def width(self) -> int:
        """int : The number of columns in the raster."""  # noqa: D403
        return self.dataset.width  # type: ignore[no-any-return]

    @property
    def shape(self):
        return self.height, self.width

    @property
    def closed(self) -> bool:
        """bool : True if the dataset is closed."""  # noqa: D403
        return self.dataset.closed  # type: ignore[no-any-return]

    def close(self) -> None:
        """Close the underlying dataset.

        Has no effect if the dataset is already closed.
        """
        if not self.closed:
            self.dataset.close()

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore[no-untyped-def]
        self.close()

    def _window_from_slices(self, key: slice | tuple[slice, ...]) -> Window:
        if isinstance(key, slice):
            row_slice = key
            col_slice = slice(None)
        else:
            row_slice, col_slice = key

        return Window.from_slices(
            row_slice, col_slice, height=self.height, width=self.width
        )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return f"{clsname}(dataset={self.dataset!r}, band={self.band!r})"

    def __setitem__(self, key: tuple[Index, ...], value: np.ndarray, /) -> None:
        if np.issubdtype(value.dtype, np.floating) and self.keep_bits is not None:
            round_mantissa(value, keep_bits=self.keep_bits)
        with rasterio.open(
            self.filename,
            "r+",
        ) as dataset:
            if len(key) == 2:
                rows, cols = key
            elif len(key) == 3:
                _, rows, cols = _unpack_3d_slices(key)
            else:
                raise ValueError(
                    f"Invalid key for {self.__class__!r}.__setitem__: {key!r}"
                )
            try:
                window = Window.from_slices(
                    rows,
                    cols,
                    height=dataset.height,
                    width=dataset.width,
                )
            except rasterio.errors.WindowError as e:
                raise ValueError(f"Error creating window: {key = }, {value = }") from e
            return dataset.write(value, self.band, window=window)


class BackgroundRasterWriter(BackgroundWriter, DatasetWriter):
    """Class to write data to files in a background thread."""

    def __init__(
        self,
        filename: Filename,
        *,
        max_queue: int = 0,
        debug: bool = False,
        keep_bits: int | None = None,
        **kwargs,
    ):
        super().__init__(nq=max_queue, name="Writer")
        if debug:
            #  background thread. Just synchronously write data
            self.notify_finished()
            self.queue_write = self.write  # type: ignore[assignment]

        if Path(filename).exists():
            self._raster = RasterWriter(filename, keep_bits=keep_bits)
        else:
            self._raster = RasterWriter.create(filename, keep_bits=keep_bits, **kwargs)
        self.filename = filename
        self.ndim = 2

    def write(self, key: tuple[Index, ...], value: np.ndarray):
        """Write out an ndarray to a subset of the pre-made `filename`.

        Parameters
        ----------
        key : tuple[Index,...]
            The key of the data to write.

        value : np.ndarray
            The block of data to write.

        """
        self._raster[key] = value

    def __setitem__(self, key: tuple[Index, ...], value: np.ndarray, /) -> None:
        self.queue_write(key, value)

    def close(self):
        """Close the underlying dataset and stop the background thread."""
        self._raster.close()
        self.notify_finished()

    @property
    def closed(self) -> bool:
        """bool : True if the dataset is closed."""  # noqa: D403
        return self._raster.closed

    @property
    def shape(self):
        return self._raster.shape

    @property
    def dtype(self) -> np.dtype:
        return self._raster.dtype

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # type: ignore[no-untyped-def]
        self.close()


class BackgroundStackWriter(BackgroundWriter, DatasetStackWriter):
    """Class to write 3D data to multiple files in a background thread.

    Will create/overwrite the files in `file_list` if they exist.
    """

    def __init__(
        self,
        file_list: Sequence[Filename],
        *,
        like_filename: Filename | None = None,
        max_queue: int = 0,
        debug: bool = False,
        keep_bits: int | None = None,
        **file_creation_kwargs,
    ):
        from dolphin.io import write_arr

        super().__init__(nq=max_queue, name="GdalStackWriter")
        if debug:
            # Stop background thread. Just synchronously write data
            self.notify_finished()
            self.queue_write = self.write  # type: ignore[assignment]

        for fn in file_list:
            write_arr(
                arr=None,
                output_name=fn,
                like_filename=like_filename,
                **file_creation_kwargs,
            )

        self.file_list = file_list
        self.keep_bits = keep_bits

        with rasterio.open(self.file_list[0]) as src:
            self.shape = (len(self.file_list), *src.shape)
            self.dtype = src.dtypes[0]

    def write(
        self, data: ArrayLike, row_start: int, col_start: int, band: int | None = None
    ):
        """Write out an ndarray to a subset of the pre-made `filename`.

        Parameters
        ----------
        data : ArrayLike
            3D data array to save.
        row_start : int
            Row index to start writing at.
        col_start : int
            Column index to start writing at.
        band : int, optional
            Band index in the file to write. Defaults to None, which uses first band,
            or whole raster for 3D data.

        Raises
        ------
        ValueError
            If length of `output_files` does not match length of `cur_block`.

        """
        from dolphin.io import write_block

        _do_round = (
            np.issubdtype(data.dtype, np.floating) and self.keep_bits is not None
        )
        if data.ndim == 2:
            data = data[None, ...]
        if data.shape[0] != len(self.file_list):
            raise ValueError(f"{data.shape = }, but {len(self.file_list) = }")
        for fn, layer in zip(self.file_list, data):
            if _do_round:
                assert self.keep_bits is not None
                round_mantissa(layer, keep_bits=self.keep_bits)
            write_block(layer, fn, row_start, col_start, band=band)

    def __setitem__(self, key, value):
        # Unpack the slices
        band, rows, cols = _unpack_3d_slices(key)
        # Check we asked to write to all the files
        if band not in (slice(None), slice(None, None, None), ...):
            self.notify_finished()
            raise NotImplementedError("Can only write to all files at once.")
        self.queue_write(value, rows.start, cols.start)

    @property
    def closed(self) -> bool:
        """bool : True if the dataset is closed."""  # noqa: D403
        return self._thread.is_alive() is False

    def close(self) -> None:
        """Close the underlying dataset and stop the background thread."""
        self.notify_finished()
