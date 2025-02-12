"""Stitch burst interferograms (optional) and unwrap them."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dolphin import stitching
from dolphin._log import log_runtime
from dolphin._overviews import ImageType, create_image_overviews, create_overviews
from dolphin._types import Bbox
from dolphin.interferogram import estimate_interferometric_correlations
from dolphin.io._utils import repack_raster

from .config import OutputOptions

logger = logging.getLogger("dolphin")


@dataclass
class StitchedOutputs:
    """Output rasters from stitching step."""

    ifg_paths: list[Path]
    """List of Paths to the stitched interferograms."""
    interferometric_corr_paths: list[Path]
    """List of Paths to interferometric correlation files created."""
    temp_coh_file: Path
    """Path to temporal correlation file created."""
    ps_file: Path
    """Path to ps mask file created."""
    amp_dispersion_file: Path
    """Path to amplitude dispersion file created."""
    shp_count_file: Path
    """Path to SHP count file created."""
    similarity_file: Path
    """Path to cosine similarity file created."""


@log_runtime
def run(
    ifg_file_list: Sequence[Path],
    temp_coh_file_list: Sequence[Path],
    ps_file_list: Sequence[Path],
    amp_dispersion_list: Sequence[Path],
    shp_count_file_list: Sequence[Path],
    similarity_file_list: Sequence[Path],
    stitched_ifg_dir: Path,
    output_options: OutputOptions,
    file_date_fmt: str = "%Y%m%d",
    corr_window_size: tuple[int, int] = (11, 11),
    num_workers: int = 3,
) -> StitchedOutputs:
    """Stitch together spatial subsets from phase linking.

    For Sentinel-1 processing, these can be burst-wise interferograms.

    Parameters
    ----------
    ifg_file_list : Sequence[Path]
        Sequence of interferograms files to stitch.
    temp_coh_file_list : Sequence[Path]
        Sequence of paths to the temporal coherence files.
    ps_file_list : Sequence[Path]
        Sequence of paths to the (looked) ps mask files.
    amp_dispersion_list : Sequence[Path]
        Sequence of paths to the (looked) amplitude dispersion files.
    shp_count_file_list : Sequence[Path]
        Sequence of paths to the SHP counts files.
    similarity_file_list : Sequence[Path]
        Sequence of paths to the spatial phase cosine similarity files.
    stitched_ifg_dir : Path
        Location to store the output stitched ifgs and correlations
    output_options : OutputOptions
        [`UnwrapWorkflow`][dolphin.workflows.config.OutputOptions] object
        for with parameters for the input/output options
    file_date_fmt : str
        Format of dates contained in filenames.
        default = "%Y%m%d"
    corr_window_size : tuple[int, int]
        Size of moving window (rows, cols) to use for estimating correlation.
        Default = (11, 11)
    num_workers : int
        Number of threads to use for stitching in parallel.
        Default = 3

    Returns
    -------
    stitched_ifg_paths : list[Path]
        list of Paths to the stitched interferograms.
    interferometric_corr_paths : list[Path]
        list of Paths to interferometric correlation files created.
    stitched_temp_coh_file : Path
        Path to temporal correlation file created.
    stitched_ps_file : Path
        Path to ps mask file created.
    stitched_amp_disp_file : Path
        Path to amplitude dispersion file created.
    stitched_shp_count_file : Path
        Path to SHP count file created.

    """
    stitched_ifg_dir.mkdir(exist_ok=True, parents=True)
    # Also preps for snaphu, which needs binary format with no nans
    logger.info("Stitching interferograms by date.")
    out_bounds = Bbox(*output_options.bounds) if output_options.bounds else None
    date_to_ifg_path = stitching.merge_by_date(
        image_file_list=ifg_file_list,
        file_date_fmt=file_date_fmt,
        output_dir=stitched_ifg_dir,
        output_suffix=".int.tif",
        driver="GTiff",
        out_bounds=out_bounds,
        out_bounds_epsg=output_options.bounds_epsg,
        num_workers=num_workers,
    )
    stitched_ifg_paths = list(date_to_ifg_path.values())

    # Estimate the interferometric correlation from the stitched interferogram
    interferometric_corr_paths = estimate_interferometric_correlations(
        stitched_ifg_paths,
        window_size=corr_window_size,
        num_workers=num_workers,
    )

    # Stitch the correlation files
    stitched_temp_coh_file = stitched_ifg_dir / "temporal_coherence.tif"
    if not stitched_temp_coh_file.exists():
        stitching.merge_images(
            temp_coh_file_list,
            outfile=stitched_temp_coh_file,
            driver="GTiff",
            resample_alg="nearest",
            out_bounds=out_bounds,
            out_bounds_epsg=output_options.bounds_epsg,
        )
        repack_raster(stitched_temp_coh_file, keep_bits=10)

    # Stitch the looked PS files
    stitched_ps_file = stitched_ifg_dir / "ps_mask_looked.tif"
    if not stitched_ps_file.exists():
        stitching.merge_images(
            ps_file_list,
            outfile=stitched_ps_file,
            out_nodata=255,
            driver="GTiff",
            resample_alg="nearest",
            out_bounds=out_bounds,
            out_bounds_epsg=output_options.bounds_epsg,
        )

    # Stitch the amp dispersion files
    stitched_amp_disp_file = stitched_ifg_dir / "amp_dispersion_looked.tif"
    if not stitched_amp_disp_file.exists():
        stitching.merge_images(
            amp_dispersion_list,
            outfile=stitched_amp_disp_file,
            driver="GTiff",
            resample_alg="nearest",
            out_bounds=out_bounds,
            out_bounds_epsg=output_options.bounds_epsg,
        )
        repack_raster(stitched_temp_coh_file, keep_bits=10)

    stitched_shp_count_file = stitched_ifg_dir / "shp_counts.tif"
    if not stitched_shp_count_file.exists():
        stitching.merge_images(
            shp_count_file_list,
            outfile=stitched_shp_count_file,
            driver="GTiff",
            resample_alg="nearest",
            out_bounds=out_bounds,
            out_bounds_epsg=output_options.bounds_epsg,
        )

    stitched_similarity_file = stitched_ifg_dir / "similarity.tif"
    if not stitched_similarity_file.exists():
        stitching.merge_images(
            similarity_file_list,
            outfile=stitched_similarity_file,
            driver="GTiff",
            resample_alg="nearest",
            out_bounds=out_bounds,
            out_bounds_epsg=output_options.bounds_epsg,
        )

    if output_options.add_overviews:
        logger.info("Creating overviews for stitched images")
        create_overviews(stitched_ifg_paths, image_type=ImageType.INTERFEROGRAM)
        create_overviews(interferometric_corr_paths, image_type=ImageType.CORRELATION)
        create_image_overviews(stitched_ps_file, image_type=ImageType.PS)
        create_image_overviews(stitched_temp_coh_file, image_type=ImageType.CORRELATION)
        create_image_overviews(stitched_amp_disp_file, image_type=ImageType.CORRELATION)
        create_image_overviews(stitched_shp_count_file, image_type=ImageType.PS)
        create_image_overviews(
            stitched_similarity_file, image_type=ImageType.CORRELATION
        )

    return StitchedOutputs(
        stitched_ifg_paths,
        interferometric_corr_paths,
        stitched_temp_coh_file,
        stitched_ps_file,
        stitched_amp_disp_file,
        stitched_shp_count_file,
        stitched_similarity_file,
    )
