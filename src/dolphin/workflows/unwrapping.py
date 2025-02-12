"""Stitch burst interferograms (optional) and unwrap them."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence

from dolphin import io, stitching, unwrap
from dolphin._log import log_runtime
from dolphin._overviews import ImageType, create_overviews
from dolphin._types import PathOrStr

from .config import UnwrapOptions

logger = logging.getLogger("dolphin")


@log_runtime
def run(
    ifg_file_list: Sequence[Path],
    cor_file_list: Sequence[Path],
    nlooks: float,
    unwrap_options: UnwrapOptions,
    temporal_coherence_filename: PathOrStr | None = None,
    similarity_filename: PathOrStr | None = None,
    mask_file: PathOrStr | None = None,
    add_overviews: bool = True,
) -> tuple[list[Path], list[Path]]:
    """Run the displacement workflow on a stack of SLCs.

    Parameters
    ----------
    ifg_file_list : Sequence[Path]
        Sequence interferograms files to unwrap.
    cor_file_list : Sequence[Path]
        Sequence interferometric correlation files, one per file in `ifg_file_list`
    nlooks : float
        Effective number of looks used to form the input correlation data.
    unwrap_options : UnwrapOptions
        [`UnwrapOptions`][dolphin.workflows.config.UnwrapOptions] config object
        with parameters for running unwrapping jobs.
    temporal_coherence_filename : Filename, optional
        Path to temporal coherence file from phase linking.
    similarity_filename : Filename, optional
        Path to phase cosine similarity file from phase linking.
    mask_file : PathOrStr, optional
        Path to boolean mask indicating nodata areas.
        1 indicates valid data, 0 indicates missing data.
    add_overviews : bool, default = True
        If True, creates overviews of the unwrapped phase and connected component
        labels.

    Returns
    -------
    unwrapped_paths : list[Path]
        list of Paths to unwrapped interferograms created.
    conncomp_paths : list[Path]
        list of Paths to connected component files created.

    """
    t0 = time.perf_counter()
    if len(ifg_file_list) != len(cor_file_list):
        msg = f"{len(ifg_file_list) = } != {len(cor_file_list) = }"
        raise ValueError(msg)

    output_path = unwrap_options._directory
    output_path.mkdir(exist_ok=True, parents=True)
    if mask_file is not None:
        output_mask = _get_matching_mask(
            mask_file=mask_file,
            output_dir=output_path,
            match_file=ifg_file_list[0],
        )
    else:
        output_mask = None

    logger.info(f"Unwrapping {len(ifg_file_list)} interferograms")

    # Make a scratch directory for unwrapping
    unwrap_scratchdir = unwrap_options._directory / "scratch"
    unwrap_scratchdir.mkdir(exist_ok=True, parents=True)

    unwrapped_paths, conncomp_paths = unwrap.run(
        ifg_filenames=ifg_file_list,
        cor_filenames=cor_file_list,
        output_path=output_path,
        unwrap_options=unwrap_options,
        nlooks=nlooks,
        temporal_coherence_filename=temporal_coherence_filename,
        similarity_filename=similarity_filename,
        mask_filename=output_mask,
        scratchdir=unwrap_scratchdir,
    )

    if add_overviews:
        logger.info("Creating overviews for unwrapped images")
        create_overviews(unwrapped_paths, image_type=ImageType.UNWRAPPED)
        create_overviews(conncomp_paths, image_type=ImageType.CONNCOMP)

    # Dump the used options for JSON parsing
    logger.info(
        "unwrapping complete",
        extra={
            "elapsed": time.perf_counter() - t0,
            "unwrap_options": unwrap_options.model_dump(mode="json"),
        },
    )

    return (unwrapped_paths, conncomp_paths)


def _get_matching_mask(
    mask_file: PathOrStr, output_dir: Path, match_file: PathOrStr
) -> Path:
    """Create a mask with the same size/projection as `match_file`."""
    # Check that the input mask is the same size as the ifgs:
    if io.get_raster_xysize(mask_file) == match_file:
        logger.info(f"Using {mask_file} to mask during unwrapping")
        output_mask = Path(mask_file)
    else:
        logger.info(f"Warping {mask_file} to match size of interferograms")
        output_mask = output_dir / "warped_mask.tif"
        if output_mask.exists():
            logger.info(f"Mask already exists at {output_mask}")
        else:
            stitching.warp_to_match(
                input_file=mask_file,
                match_file=match_file,
                output_file=output_mask,
            )
    return output_mask
