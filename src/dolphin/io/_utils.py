from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dolphin._types import Index

__all__ = [
    "get_gtiff_options",
    "repack_raster",
    "repack_rasters",
    "round_mantissa",
]


def _ensure_slices(rows: Index, cols: Index) -> tuple[slice, slice]:
    def _parse(key: Index):
        if isinstance(key, int):
            return slice(key, key + 1)
        elif key is ...:
            return slice(None)
        else:
            return key

    return _parse(rows), _parse(cols)


def _unpack_3d_slices(key: tuple[Index, ...]) -> tuple[Index, slice, slice]:
    # Check that it's a tuple of slices
    if not isinstance(key, tuple):
        msg = "Index must be a tuple of slices."
        raise TypeError(msg)
    if len(key) not in (1, 3):
        msg = "Index must be a tuple of 1 or 3 slices."
        raise TypeError(msg)
    # If only the band is passed (e.g. stack[0]), convert to (0, :, :)
    if len(key) == 1:
        key = (key[0], slice(None), slice(None))
    # unpack the slices
    bands, rows, cols = key
    # convert the rows/cols to slices
    r_slice, c_slice = _ensure_slices(rows, cols)
    return bands, r_slice, c_slice


def get_gtiff_options(
    max_error: float | None = None,
    compression_type: str = "lzw",
    chunk_size: int = 256,
    predictor: int | None = None,
    zlevel: int | None = 1,
) -> dict[str, str]:
    """Generate GTiff creation options for GDAL translate.

    Parameters
    ----------
    max_error : float
        Maximum compression error.
    compression_type : str, optional
        Compression type to use (default is "lzw").
    chunk_size : int, optional
        Size of the chunks for blockxsize and blockysize (default is 256).
    predictor : int or None, optional
        Predictor type to use (default is 3). Use None to omit the predictor.
    zlevel : int or None, optional
        Compression level for the 'deflate' and 'zstd' compression types (default is 1).
        Use None to omit the zlevel.

    Returns
    -------
    dict[str, str] | list[str]
        List of GTiff creation options formatted for GDAL (if `gdal_format=True`)
        Otherwise, a dict mapping option to value for rasterio.

    """
    options = {
        "bigtiff": "yes",
        "tiled": "yes",
        "blockxsize": str(chunk_size),
        "blockysize": str(chunk_size),
        "compress": compression_type,
    }
    if zlevel is not None:
        options["zlevel"] = str(zlevel)
    if predictor is not None:
        options["predictor"] = str(predictor)
    if compression_type.lower().startswith("lerc") and max_error is not None:
        options["max_z_error"] = str(max_error)

    return options


def _format_for_gdal(options: dict[str, str]) -> list[str]:
    """Output creation options as a list of -co strings for GDAL.

    Parameters
    ----------
    options : dict[str, str]
        Dict of creation options usable in Rasterio.

    Returns
    -------
    list[str]
        List -co options for GDAL.

    """
    return [f"{k.upper()}={v}" for k, v in options.items()]


def repack_raster(
    raster_path: Path,
    output_dir: Path | None = None,
    keep_bits: int | None = None,
    block_shape: int | tuple[int, int] = (1024, 1024),
    **output_options,
) -> Path:
    """Repack a single raster file with GDAL Translate using provided options.

    Parameters
    ----------
    raster_path : Path
        Path to the input raster file.
    output_dir : Path, optional
        Directory to save the repacked rasters or None for in-place repacking.
    keep_bits : int, optional
        Number of bits to preserve in mantissa. Defaults to None.
        Lower numbers will truncate the mantissa more and enable more compression.
    block_shape: int | tuple[int, int]
        Size of blocks to read in at one time.
    **output_options
        Keyword args passed to `get_gtiff_options`

    Returns
    -------
    output_path : Path
        Path to newly created file.
        If `output_dir` is None, this will be the same filename as `raster_path`

    """
    import rasterio as rio
    from rasterio.windows import Window

    from ._blocks import iter_blocks

    if isinstance(block_shape, int):
        block_shape = (block_shape, block_shape)

    if output_dir is None:
        output_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            suffix=raster_path.suffix, dir=output_dir, delete=False
        )
        output_path = Path(output_file.name)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / raster_path.name

    options = get_gtiff_options(**output_options)

    with rio.open(raster_path) as src:
        profile = src.profile
        profile.update(**options)
        # Work in blocks on the input raster
        blocks = iter_blocks(
            arr_shape=(src.height, src.width),
            block_shape=block_shape,
        )

        with rio.open(output_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                for row_slice, col_slice in blocks:
                    window = Window.from_slices(rows=row_slice, cols=col_slice)
                    data = src.read(i, window=window)

                    if keep_bits is not None:
                        round_mantissa(data, keep_bits)

                    dst.write(data, i, window=window)

    if output_dir is None:
        # Overwrite the original
        shutil.move(output_path, raster_path)
        output_path = raster_path

    return output_path


def repack_rasters(
    raster_files: list[Path],
    output_dir: Path | None = None,
    num_threads: int = 4,
    keep_bits: int | None = None,
    block_shape: int | tuple[int, int] = (1024, 1024),
    **output_options,
):
    """Recreate and compress a list of raster files.

    Useful for rasters which were created in block and lost
    the full effect of compression.

    Parameters
    ----------
    raster_files : List[Path]
        List of paths to the input raster files.
    output_dir : Path, optional
        Directory to save the processed rasters or None for in-place processing.
    num_threads : int, optional
        Number of threads to use (default is 4).
    keep_bits : int, optional
        Number of bits to preserve in mantissa. Defaults to None.
        Lower numbers will truncate the mantissa more and enable more compression.
    block_shape: int | tuple[int, int]
        Size of blocks to read in at one time.
    **output_options
        Creation options to pass to `get_gtiff_options`

    Returns
    -------
    output_path : Path
        Path to newly created file.
        If `output_dir` is None, this will be the same as `raster_paths`

    """
    from tqdm.contrib.concurrent import thread_map

    thread_map(
        lambda raster: repack_raster(
            raster,
            output_dir,
            keep_bits=keep_bits,
            block_shape=block_shape,
            **output_options,
        ),
        raster_files,
        max_workers=num_threads,
        desc="Processing Rasters",
    )


def round_mantissa(z: np.ndarray, keep_bits: int = 10) -> None:
    """Zero out mantissa bits of elements of array in place.

    Drops a specified number of bits from the floating point mantissa,
    leaving an array more amenable to compression.

    Parameters
    ----------
    z : numpy.ndarray
        Real or complex array whose mantissas are to be zeroed out
    keep_bits : int
        Number of bits to preserve in mantissa.
        Lower numbers will truncate the mantissa more and enable
        more compression.
        Default is 10.

    References
    ----------
    https://numcodecs.readthedocs.io/en/v0.12.1/_modules/numcodecs/bitround.html

    """
    max_bits = {
        "float16": 10,
        "float32": 23,
        "float64": 52,
    }
    # recurse for complex data
    if np.iscomplexobj(z):
        round_mantissa(z.real, keep_bits)
        round_mantissa(z.imag, keep_bits)
        return

    if not z.dtype.kind == "f" or z.dtype.itemsize > 8:
        raise TypeError("Only float arrays (16-64bit) can be bit-rounded")

    bits = max_bits[str(z.dtype)]
    # cast float to int type of same width (preserve endianness)
    a_int_dtype = np.dtype(z.dtype.str.replace("f", "i"))
    all_set = np.array(-1, dtype=a_int_dtype)
    if keep_bits == bits:
        return z
    if keep_bits > bits:
        raise ValueError("keep_bits too large for given dtype")
    b = z.view(a_int_dtype)
    maskbits = bits - keep_bits
    mask = (all_set >> maskbits) << maskbits
    half_quantum1 = (1 << (maskbits - 1)) - 1
    b += ((b >> maskbits) & 1) + half_quantum1
    b &= mask
