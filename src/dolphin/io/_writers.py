from __future__ import annotations

from contextlib import AbstractContextManager
from typing import (
    TYPE_CHECKING,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import numpy as np
import rasterio
from numpy.typing import ArrayLike, DTypeLike

from dolphin._types import Filename

from ._background import BackgroundWriter

__all__ = [
    "DatasetWriter",
    "RasterWriter",
    "Writer",
]

if TYPE_CHECKING:
    from dolphin._types import Index


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


RasterT = TypeVar("RasterT", bound="RasterWriter")


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

    @classmethod
    def create(
        cls: type[RasterT],
        fp: Filename,
        width: int | None = None,
        height: int | None = None,
        dtype: DTypeLike | None = None,
        driver: str | None = None,
        crs: str | Mapping[str, str] | rasterio.crs.CRS | None = None,
        transform: rasterio.transform.Affine | None = None,
        *,
        like: Raster | None = None,
        **kwargs: Any,
    ) -> RasterT:
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
        like : Raster or None, optional
            An optional reference raster. If not None, the new raster will be created
            with the same metadata (shape, data-type, driver, CRS/geotransform, etc) as
            the reference raster. All other arguments will override the corresponding
            attribute of the reference raster. Defaults to None.
        **kwargs : dict, optional
            Additional driver-specific creation options passed to `rasterio.open`.
        """
        if like is not None:
            kwargs = like.dataset.profile | kwargs

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
        dataset = rasterio.open(fp, mode="w+", **kwargs)

        # XXX We need this gross hack in order to bypass calling `__init__` (which only
        # supports opening existing datasets).
        raster = cls.__new__(cls)
        raster._dataset = dataset
        raster._band = 1

        return raster

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
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dataset(self) -> rasterio.io.DatasetWriter:
        """The underlying `rasterio` dataset."""
        return self._dataset

    @property
    def band(self) -> int:
        """int : Band index (1-based)."""  # noqa: D403
        return self._band

    @property
    def mode(self) -> str:
        """str : File access mode."""  # noqa: D403
        return self.dataset.mode  # type: ignore[no-any-return]

    @property
    def driver(self) -> str:
        """str : Raster format driver name."""  # noqa: D403
        return self.dataset.driver  # type: ignore[no-any-return]

    @property
    def crs(self) -> rasterio.crs.CRS:
        """rasterio.crs.CRS : The dataset's coordinate reference system."""
        return self.dataset.crs

    @property
    def transform(self) -> rasterio.transform.Affine:
        """rasterio.transform.Affine : dataset's georeferencing transformation matrix.

        This transform maps pixel row/column coordinates to coordinates in the dataset's
        coordinate reference system.
        """
        return self.dataset.transform

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

    def __array__(self) -> np.ndarray:
        return self.dataset.read(self.band)

    def _window_from_slices(
        self, key: slice | tuple[slice, ...]
    ) -> rasterio.windows.Window:
        if isinstance(key, slice):
            row_slice = key
            col_slice = slice(None)
        else:
            row_slice, col_slice = key

        return rasterio.windows.Window.from_slices(
            row_slice, col_slice, height=self.height, width=self.width
        )

    def __getitem__(self, key: slice | tuple[slice, ...], /) -> np.ndarray:
        window = self._window_from_slices(key)
        return self.dataset.read(self.band, window=window)

    def __setitem__(self, key: slice | tuple[slice, ...], value: np.ndarray, /) -> None:
        window = self._window_from_slices(key)
        self.dataset.write(value, self.band, window=window)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return f"{clsname}(dataset={self.dataset!r}, band={self.band!r})"


class RasterWriter(BackgroundWriter, RasterWriter):
    """Class to write data to files in a background thread."""

    def __init__(self, max_queue: int = 0, debug: bool = False, **kwargs):
        if debug is False:
            super().__init__(nq=max_queue, name="Writer", **kwargs)
        else:
            # Don't start a background thread. Just synchronously write data
            self.queue_write = self.write  # type: ignore[assignment]

    def write(
        self, data: ArrayLike, filename: Filename, row_start: int, col_start: int
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

        Raises
        ------
        ValueError
            If length of `output_files` does not match length of `cur_block`.
        """
        write_block(data, filename, row_start, col_start)

    def __setitem__(self, key, value):
        self.queue_write(value, key)


class Writer(BackgroundWriter):
    """Class to write data to files in a background thread."""

    def __init__(self, max_queue: int = 0, debug: bool = False, **kwargs):
        if debug is False:
            super().__init__(nq=max_queue, name="Writer", **kwargs)
        else:
            # Don't start a background thread. Just synchronously write data
            self.queue_write = self.write  # type: ignore[assignment]

    def write(
        self, data: ArrayLike, filename: Filename, row_start: int, col_start: int
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

        Raises
        ------
        ValueError
            If length of `output_files` does not match length of `cur_block`.
        """
        from dolphin.io import write_block

        write_block(data, filename, row_start, col_start)

    def __setitem__(self, key, value):
        self.queue_write(value, key)
