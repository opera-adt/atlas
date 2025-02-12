from __future__ import annotations

import argparse
import logging
from enum import Enum
from os import fspath
from pathlib import Path
from typing import Sequence

from osgeo import gdal
from tqdm.contrib.concurrent import thread_map

from dolphin._types import PathOrStr

gdal.UseExceptions()

logger = logging.getLogger("dolphin")

DEFAULT_LEVELS = [4, 8, 16, 32, 64]


class Resampling(Enum):
    """GDAL resampling algorithm."""

    NEAREST = "nearest"
    AVERAGE = "average"


class ImageType(Enum):
    """Types of images produced by dolphin."""

    UNWRAPPED = "unwrapped"
    INTERFEROGRAM = "interferogram"
    CORRELATION = "correlation"
    CONNCOMP = "conncomp"
    PS = "ps"


IMAGE_TYPE_TO_RESAMPLING = {
    ImageType.UNWRAPPED: Resampling.AVERAGE,
    ImageType.INTERFEROGRAM: Resampling.AVERAGE,
    ImageType.CORRELATION: Resampling.AVERAGE,
    ImageType.CONNCOMP: Resampling.NEAREST,
    # No max in resampling, yet, which would be best
    # https://github.com/OSGeo/gdal/issues/3683
    ImageType.PS: Resampling.AVERAGE,
}


def create_image_overviews(
    file_path: Path | str,
    levels: Sequence[int] = DEFAULT_LEVELS,
    image_type: ImageType | None = None,
    resampling: Resampling | None = None,
    external: bool = False,
    compression: str = "LZW",
    num_gdal_threads: int = 1,
):
    """Add GDAL compressed overviews to an existing file.

    Parameters
    ----------
    file_path : Path
        Path to the file to process.
    levels : Sequence[int]
        List of overview levels to add.
        Default = [4, 8, 16, 32, 64]
    image_type : ImageType, optional
        If provided, looks up the default resampling algorithm
        most appropriate for this type of raster.
    resampling : str or Resampling
        GDAL resampling algorithm for overviews. Required
        if not specifying `image_type`.
    external : bool, default = False
        Use external overviews (.ovr files).
    compression: str, default = "DEFLATE"
        Compression algorithm to use for overviews.
        See https://gdal.org/programs/gdaladdo.html for options.
    num_gdal_threads: int, default = 1
        Number of threads to use for overview creation.
        For more details, see https://gdal.org/programs/gdaladdo.html and
        https://gdal.org/en/stable/user/configoptions.html#config-GDAL_NUM_THREADS

    """
    if image_type is None and resampling is None:
        raise ValueError("Must provide `image_type` or `resampling`")
    if image_type is not None:
        resampling = IMAGE_TYPE_TO_RESAMPLING[ImageType(image_type)]
    else:
        resampling = Resampling(resampling)

    flags = gdal.GA_Update if not external else gdal.GA_ReadOnly
    ds = gdal.Open(fspath(file_path), flags)
    if ds.GetRasterBand(1).GetOverviewCount() > 0:
        logger.debug("%s already has overviews. Skipping.", file_path)
        return

    gdal.SetConfigOption("COMPRESS_OVERVIEW", compression)
    gdal.SetConfigOption("GDAL_NUM_THREADS", str(num_gdal_threads))
    ds.BuildOverviews(resampling.value, levels)


def create_overviews(
    file_paths: Sequence[PathOrStr],
    levels: Sequence[int] = DEFAULT_LEVELS,
    image_type: ImageType | None = None,
    resampling: Resampling = Resampling.AVERAGE,
    max_workers: int = 5,
) -> None:
    """Process many files to add GDAL overviews and compression.

    Parameters
    ----------
    file_paths : Sequence[PathOrStr]
        Sequence of file paths to process.
    levels : Sequence[int]
        Sequence of overview levels to add.
        Default = [4, 8, 16, 32, 64]
    image_type : ImageType, optional
        If provided, looks up the default resampling algorithm
    resampling : str or Resampling
        GDAL resampling algorithm for overviews. Required
        if not specifying `image_type`.
    max_workers : int, default = 5
        Number of parallel threads to run.

    """
    thread_map(
        lambda file_path: create_image_overviews(
            Path(file_path),
            levels=list(levels),
            image_type=image_type,
            resampling=resampling,
        ),
        file_paths,
        max_workers=max_workers,
    )


def run():
    """Add compressed GDAL overviews to files."""
    parser = argparse.ArgumentParser(
        description="Add compressed GDAL overviews to files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("file_paths", nargs="+", type=str, help="Path to files")
    parser.add_argument(
        "--levels",
        "-l",
        nargs="*",
        default=[4, 8, 16, 32, 64],
        type=int,
        help="Overview levels to add.",
    )
    parser.add_argument(
        "--resampling",
        "-r",
        default=Resampling("nearest"),
        choices=[r.value for r in Resampling],
        type=Resampling,
        help="Resampling algorithm to use when building overviews",
    )
    parser.add_argument(
        "--max-workers",
        "-n",
        default=5,
        type=int,
        help="Number of parallel files to process",
    )

    args = parser.parse_args()

    # Convert resampling argument from string to Resampling Enum
    resampling_enum = Resampling(args.resampling)

    create_overviews(
        file_paths=args.file_paths,
        levels=args.levels,
        resampling=resampling_enum,
        max_workers=args.max_workers,
    )
