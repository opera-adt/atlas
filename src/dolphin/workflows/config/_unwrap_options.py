from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
)

from ._enums import UnwrapMethod

logger = logging.getLogger("dolphin")

__all__ = [
    "PreprocessOptions",
    "SnaphuOptions",
    "SpurtOptions",
    "TophuOptions",
    "UnwrapOptions",
]


def to_tuple(v: int | tuple[int, int] | None) -> tuple[int, int]:
    """Convert the input value to a tuple of two integers.

    Parameters
    ----------
    v : Union[int, Tuple[int, int], None]
        The value to be converted.

    Returns
    -------
    Tuple[int, int]
        Tuple containing two integers.

    Examples
    --------
    >>> to_tuple(3)
    (3, 3)
    >>> to_tuple((2, 4))
    (2, 4)
    >>> to_tuple(None)
    (1, 1)

    """
    if v is None:
        return (1, 1)
    if isinstance(v, int):
        return (v, v)
    return v


class PreprocessOptions(BaseModel, extra="forbid"):
    alpha: float = Field(
        0.5,
        description="Adaptive phase (Goldstein) filter exponent parameter.",
        ge=0.0,
        le=1.0,
    )
    max_radius: int = Field(
        51,
        ge=0.0,
        description="(for interpolation) Maximum radius to find scatterers.",
    )
    interpolation_cor_threshold: float = Field(
        0.3,
        description=(
            "Threshold on the correlation raster to use for interpolation. Pixels with"
            " less than this value are replaced by a weighted combination of"
            " neighboring pixels."
        ),
        ge=0.0,
        le=1.0,
    )
    interpolation_similarity_threshold: float = Field(
        0.3,
        description=(
            "Threshold on the correlation raster to use for interpolation. Pixels with"
            " less than this value are replaced by a weighted combination of"
            " neighboring pixels."
        ),
        ge=0.0,
        le=1.0,
    )


class SnaphuOptions(BaseModel, extra="forbid"):
    ntiles: tuple[int, int] = Field(
        (1, 1),
        description=(
            "Number of tiles to split the inputs into using SNAPHU's internal tiling."
        ),
    )
    tile_overlap: tuple[int, int] = Field(
        (0, 0),
        description=(
            "Amount of tile overlap (in pixels) along the (row, col) directions."
        ),
    )
    n_parallel_tiles: int = Field(
        1,
        description="Number of tiles to unwrap in parallel for each interferogram.",
    )
    init_method: Literal["mcf", "mst"] = Field(
        "mcf",
        description="Initialization method for SNAPHU.",
    )
    cost: Literal["defo", "smooth"] = Field(
        "smooth",
        description="Statistical cost mode method for SNAPHU.",
    )
    single_tile_reoptimize: bool = Field(
        False,
        description=(
            "If True, after unwrapping with multiple tiles, an additional"
            " post-processing unwrapping step is performed to re-optimize the unwrapped"
            " phase using a single tile"
        ),
    )

    _to_tuple = field_validator("ntiles", "tile_overlap", mode="before")(to_tuple)


class TophuOptions(BaseModel, extra="forbid"):
    ntiles: tuple[int, int] = Field(
        (1, 1),
        description="Number of tiles to split the inputs into",
    )
    downsample_factor: tuple[int, int] = Field(
        (1, 1),
        description="Extra multilook factor to use for the coarse unwrap.",
    )
    init_method: Literal["mcf", "mst"] = Field(
        "mcf",
        description="Initialization method for SNAPHU.",
    )
    cost: Literal["defo", "smooth"] = Field(
        "smooth",
        description="Statistical cost mode method for SNAPHU.",
    )

    _to_tuple = field_validator("ntiles", "downsample_factor", mode="before")(to_tuple)


class SpurtGeneralSettings(BaseModel):
    use_tiles: bool = Field(default=True, description="Tile up data spatially.")


class SpurtTilerSettings(BaseModel):
    """Class for holding tile generation settings."""

    max_tiles: int = Field(64, description="Maximum number of tiles allowed.", ge=1)
    target_points_for_generation: int = Field(
        120_000,
        description="Number of points used for determining tiles based on density.",
        gt=0,
    )
    target_points_per_tile: int = Field(
        900_000, description="Target points per tile when generating tiles.", gt=0
    )
    dilation_factor: float = Field(
        0.05,
        description=(
            "Dilation factor of non-overlapping tiles. 0.05 would lead to 5 percent"
            " dilation of the tile."
        ),
        ge=0.0,
    )


class SpurtSolverSettings(BaseModel):
    t_worker_count: int = Field(
        default=1,
        alias="t_worker_count",
        description=(
            "Number of workers for temporal unwrapping in parallel. Set value to <=0 to"
            " let workflow use default workers (ncpus - 1)."
        ),
    )
    s_worker_count: int = Field(
        default=1,
        description=(
            "Number of workers for spatial unwrapping in parallel. Set value to <=0"
            " to let workflow use (ncpus - 1)."
        ),
    )
    links_per_batch: int = Field(
        default=150_000,
        description=(
            "Temporal unwrapping operations over spatial links are performed in batches"
            " and each batch is solved in parallel."
        ),
        gt=0,
    )
    t_cost_type: Literal["constant", "distance", "centroid"] = Field(
        default="constant",
        description="Temporal unwrapping costs.",
    )
    t_cost_scale: float = Field(
        default=100.0,
        description="Scale factor used to compute edge costs for temporal unwrapping.",
        gt=0.0,
    )
    s_cost_type: Literal["constant", "distance", "centroid"] = Field(
        default="constant", description="Spatial unwrapping costs.", alias="s_cost_type"
    )
    s_cost_scale: float = Field(
        default=100.0,
        description="Scale factor used to compute edge costs for spatial unwrapping.",
        gt=0.0,
    )
    num_parallel_tiles: int = Field(
        default=1,
        description="Number of tiles to process in parallel. Set to 0 for all tiles.",
        ge=0.0,
    )


class SpurtMergerSettings(BaseModel):
    min_overlap_points: int = Field(
        default=25,
        description="Minimum number of overlap pixels to be considered valid.",
    )
    method: Literal["dirichlet"] = Field(
        default="dirichlet", description="Currently, only 'dirichlet' is supported."
    )
    bulk_method: Literal["integer", "L2"] = Field(
        default="L2", description="Method used to estimate bulk offset between tiles."
    )
    num_parallel_ifgs: int = Field(
        default=3,
        ge=0,
        description=(
            "Number of interferograms to merge in one batch. Use zero to merge all"
            " interferograms in a single batch."
        ),
    )


class SpurtOptions(BaseModel, extra="forbid"):
    """Options for running 3D unwrapping on a set of interferograms.

    Uses [`spurt`](https://github.com/isce-framework/spurt) to run the
    temporal/spatial unwrapping. Options are passed through to `spurt`
    library.

    """

    temporal_coherence_threshold: float = Field(
        0.7,
        description="Temporal coherence to pick pixels used on an irregular grid.",
        ge=0.0,
        lt=1.0,
    )
    similarity_threshold: float = Field(
        0.5,
        description=(
            "Similarity to pick pixels used on an irregular grid. Any pixel with"
            " similarity above `similarity_threshold` *or* above the temporal coherence"
            " threshold is chosen."
        ),
        ge=0.0,
        lt=1.0,
    )
    run_ambiguity_interpolation: bool = Field(
        True,
        description=(
            "After running spurt, interpolate the values that were masked during"
            " unwrapping (which are otherwise left as nan)."
        ),
    )
    # TODO: do we want to allow a "AND" or "OR" option, so users can pick if they want
    # `good_sim & good_temp_coh`, or `good_sim | good_temp_coh`
    general_settings: SpurtGeneralSettings = Field(default_factory=SpurtGeneralSettings)
    tiler_settings: SpurtTilerSettings = Field(default_factory=SpurtTilerSettings)
    solver_settings: SpurtSolverSettings = Field(default_factory=SpurtSolverSettings)
    merger_settings: SpurtMergerSettings = Field(default_factory=SpurtMergerSettings)


class UnwrapOptions(BaseModel, extra="forbid"):
    """Options for unwrapping after wrapped phase estimation."""

    run_unwrap: bool = Field(
        True,
        description=(
            "Whether to run the unwrapping step after wrapped phase estimation."
        ),
    )
    run_goldstein: bool = Field(
        False,
        description="Whether to run Goldstein filtering step on wrapped interferogram.",
    )
    run_interpolation: bool = Field(
        False,
        description="Whether to run interpolation step on wrapped interferogram.",
    )
    _directory: Path = PrivateAttr(Path("unwrapped"))
    unwrap_method: UnwrapMethod = UnwrapMethod.SNAPHU
    n_parallel_jobs: int = Field(
        1, description="Number of interferograms to unwrap in parallel."
    )
    zero_where_masked: bool = Field(
        False,
        description=(
            "Set wrapped phase/correlation to 0 where mask is 0 before unwrapping. "
        ),
    )
    preprocess_options: PreprocessOptions = Field(default_factory=PreprocessOptions)
    snaphu_options: SnaphuOptions = Field(default_factory=SnaphuOptions)
    tophu_options: TophuOptions = Field(default_factory=TophuOptions)
    spurt_options: SpurtOptions = Field(default_factory=SpurtOptions)
