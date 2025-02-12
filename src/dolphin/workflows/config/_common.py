from __future__ import annotations

import glob
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from dolphin import __version__ as _dolphin_version
from dolphin._types import Bbox
from dolphin.io import DEFAULT_HDF5_OPTIONS, DEFAULT_TIFF_OPTIONS
from dolphin.stack import CompressedSlcPlan

from ._enums import ShpMethod
from ._yaml_model import YamlModel

logger = logging.getLogger("dolphin")

__all__ = [
    "HalfWindow",
    "InputOptions",
    "InterferogramNetwork",
    "OutputOptions",
    "PhaseLinkingOptions",
    "PsOptions",
    "TimeseriesOptions",
    "WorkerSettings",
]


class PsOptions(BaseModel, extra="forbid"):
    """Options for the PS pixel selection portion of the workflow."""

    _directory: Path = PrivateAttr(Path("PS"))
    _output_file: Path = PrivateAttr(Path("PS/ps_pixels.tif"))
    _amp_dispersion_file: Path = PrivateAttr(Path("PS/amp_dispersion.tif"))
    _amp_mean_file: Path = PrivateAttr(Path("PS/amp_mean.tif"))

    amp_dispersion_threshold: float = Field(
        0.25,
        description="Amplitude dispersion threshold to consider a pixel a PS.",
        gt=0.0,
    )


class HalfWindow(BaseModel, extra="forbid"):
    """Class to hold half-window size for multi-looking during phase linking."""

    x: int = Field(11, description="Half window size (in pixels) for x direction", gt=0)
    y: int = Field(5, description="Half window size (in pixels) for y direction", gt=0)

    def to_looks(self):
        """Convert (x, y) half-window size to (row, column) looks."""
        return 2 * self.y + 1, 2 * self.x + 1

    @classmethod
    def from_looks(cls, row_looks: int, col_looks: int):
        """Create a half-window from looks."""
        return cls(x=col_looks // 2, y=row_looks // 2)


class PhaseLinkingOptions(BaseModel, extra="forbid"):
    """Configurable options for wrapped phase estimation."""

    _directory: Path = PrivateAttr(Path("linked_phase"))
    ministack_size: int = Field(
        15, description="Size of the ministack for sequential estimator.", gt=1
    )
    max_num_compressed: int = Field(
        100,
        description=(
            "Maximum number of compressed images to use in sequential estimator."
            " If there are more ministacks than this, the earliest CCSLCs will be"
            " left out of the later stacks. "
        ),
        gt=0,
    )
    output_reference_idx: int = Field(
        0,
        description=(
            "Index of input SLC to use for making phase linked interferograms after"
            " EVD/EMI."
        ),
    )
    half_window: HalfWindow = HalfWindow()
    use_evd: bool = Field(
        False, description="Use EVD on the coherence instead of using the EMI algorithm"
    )

    beta: float = Field(
        0.00,
        description=(
            "Beta regularization parameter for correlation matrix inversion. 0 is no"
            " regularization."
        ),
        ge=0.0,
        le=1.0,
    )
    zero_correlation_threshold: float = Field(
        0.00,
        description=(
            "Snap correlation values in the coherence matrix below this value to 0."
        ),
        ge=0.0,
        le=1.0,
    )
    shp_method: ShpMethod = ShpMethod.GLRT
    shp_alpha: float = Field(
        0.001,
        description=(
            "Significance level (probability of false alarm) for SHP tests. Lower"
            " numbers include more pixels within the multilook window during covariance"
            " estimation."
        ),
        gt=0.0,
        lt=1.0,
    )
    mask_input_ps: bool = Field(
        False,
        description=(
            "If True, pixels labeled as PS will get set to NaN during phase linking to"
            " avoid summing their phase. Default of False means that the SHP algorithm"
            " will decide if a pixel should be included, regardless of its PS label."
        ),
    )
    baseline_lag: Optional[int] = Field(
        None,
        gt=0,
        description=(
            "StBAS parameter to include only nearest-N interferograms for"
            "phase linking. A `baseline_lag` of `n` will only include the closest"
            "`n` interferograms. `baseline_line` must be positive."
        ),
    )
    compressed_slc_plan: CompressedSlcPlan = CompressedSlcPlan.ALWAYS_FIRST

    @field_validator("compressed_slc_plan", mode="before")
    @classmethod
    def _replace_none(cls, v):
        if v is None:
            return CompressedSlcPlan.ALWAYS_FIRST
        return v


class InterferogramNetwork(BaseModel, extra="forbid"):
    """Options to determine the type of network for interferogram formation.

    If no parameters passed, uses single-reference network with `reference_idx=0`.
    """

    _directory: Path = PrivateAttr(Path("interferograms"))

    reference_idx: Optional[int] = Field(
        None,
        description=(
            "For single-reference network: Index of the reference image in the network"
        ),
    )
    max_bandwidth: Optional[int] = Field(
        None,
        description="Max `n` to form the nearest-`n` interferograms by index.",
        ge=1,
    )
    max_temporal_baseline: Optional[int] = Field(
        None,
        description="Maximum temporal baseline of interferograms.",
        ge=0,
    )
    indexes: Optional[list[tuple[int, int]]] = Field(
        None,
        description=(
            "For manual-index network: list of (ref_idx, sec_idx) defining the"
            " interferograms to form."
        ),
    )

    @model_validator(mode="after")
    def _check_zero_parameters(self) -> InterferogramNetwork:
        ref_idx = self.reference_idx
        max_bw = self.max_bandwidth
        max_tb = self.max_temporal_baseline
        indexes = self.indexes
        # Check if more than one has been set:
        if ref_idx is None and max_bw is None and max_tb is None and indexes is None:
            logger.debug(
                "No network configuration options were set. Using single-reference."
            )
            self.reference_idx = 0
        return self


class TimeseriesOptions(BaseModel, extra="forbid"):
    """Options for inversion/time series fitting."""

    _directory: Path = PrivateAttr(Path("timeseries"))
    _velocity_file: Path = PrivateAttr(Path("timeseries/velocity.tif"))
    run_inversion: bool = Field(
        True,
        description=(
            "Whether to run the inversion step after unwrapping, if more than "
            " a single-reference network is used."
        ),
    )
    method: Literal["L1", "L2"] = Field(
        "L1", description="Norm to use during timeseries inversion."
    )
    reference_point: Optional[tuple[int, int]] = Field(
        None,
        description=(
            "Reference point (row, col) used if performing a time series inversion. "
            "If not provided, a point will be selected from a consistent connected "
            "component with low amplitude dispersion."
        ),
    )

    run_velocity: bool = Field(
        True,
        description="Run the velocity estimation from the phase time series.",
    )
    correlation_threshold: float = Field(
        0.2,
        description="Pixels with correlation below this value will be masked out.",
        ge=0.0,
        le=1.0,
    )
    block_shape: tuple[int, int] = Field(
        (256, 256),
        description=(
            "Size (rows, columns) of blocks of data to load at a time. 3D dimsion is"
            " number of interferograms (during inversion) and number of SLC dates"
            " (during velocity fitting)"
        ),
    )
    num_parallel_blocks: int = Field(
        4, description="Number of parallel blocks to process at once."
    )


class WorkerSettings(BaseModel, extra="forbid"):
    """Settings for controlling CPU/GPU settings and parallelism."""

    gpu_enabled: bool = Field(
        False,
        description="Whether to use GPU for processing (if available)",
    )
    threads_per_worker: int = Field(
        1,
        ge=1,
        description=(
            "Number of threads to use per worker. This sets the OMP_NUM_THREADS"
            " environment variable in each python process."
        ),
    )
    n_parallel_bursts: int = Field(
        default=1,
        ge=1,
        description=(
            "If processing separate spatial bursts, number of bursts to run in parallel"
            " for wrapped-phase-estimation."
        ),
    )
    block_shape: tuple[int, int] = Field(
        (512, 512),
        description="Size (rows, columns) of blocks of data to load at a time.",
    )


class InputOptions(BaseModel, extra="forbid"):
    """Options specifying input datasets for workflow."""

    subdataset: Optional[str] = Field(
        None,
        description="If passing HDF5/NetCDF files, subdataset to use from CSLC files. ",
    )
    cslc_date_fmt: str = Field(
        "%Y%m%d",
        description="Format of dates contained in CSLC filenames",
    )
    wavelength: Optional[float] = Field(
        None,
        description=(
            "Radar wavelength (in meters) of the transmitted data. used to convert the"
            " units in the rasters in `timeseries/` to from radians to meters. If None"
            " and sensor is not recognized, outputs remain in radians."
        ),
    )


class OutputOptions(BaseModel, extra="forbid"):
    """Options for the output size/format/compressions."""

    output_resolution: Optional[dict[str, int]] = Field(
        # {"x": 20, "y": 20},
        # TODO: how to get a blank "x" and "y" in the schema printed instead of nothing?
        None,
        description="Output (x, y) resolution (in units of input data)",
    )
    strides: dict[str, int] = Field(
        {"x": 1, "y": 1},
        description=(
            "Alternative to specifying output resolution: Specify the (x, y) strides"
            " (decimation factor) to perform while processing input. For example,"
            " strides of [4, 2] would turn an input resolution of [5, 10] into an"
            " output resolution of [20, 20]."
        ),
        validate_default=True,
    )
    bounds: Optional[tuple[float, float, float, float]] = Field(
        None,
        description=(
            "Area of interest: [left, bottom, right, top] coordinates. "
            "e.g. `bbox=[-150.2,65.0,-150.1,65.5]`"
        ),
    )
    bounds_wkt: Optional[str] = Field(
        None,
        description=(
            "Area of interest as a simple Polygon in well-known-text (WKT) format."
            " Can pass a string, or a `.wkt` filename containing the Polygon text."
        ),
    )
    bounds_epsg: int = Field(
        4326,
        description=(
            "EPSG code for the `bounds` or `bounds_wkt` coordinates, if specified."
        ),
    )

    hdf5_creation_options: dict = Field(
        DEFAULT_HDF5_OPTIONS,
        description="Options for `create_dataset` with h5py.",
    )
    gtiff_creation_options: list[str] = Field(
        list(DEFAULT_TIFF_OPTIONS),
        description="GDAL creation options for GeoTIFF files",
    )
    add_overviews: bool = Field(
        True,
        description=(
            "Whether to add overviews to the output GeoTIFF files. This will "
            "increase file size, but can be useful for visualizing the data with "
            "web mapping tools. See https://gdal.org/programs/gdaladdo.html for more."
        ),
    )
    overview_levels: list[int] = Field(
        [4, 8, 16, 32, 64],
        description="List of overview levels to create (if `add_overviews=True`).",
    )
    # Note: we use NaiveDatetime, since other datetime parsing results in Naive
    # (no TzInfo) datetimes, which can't be compared to datetimes with timezones
    extra_reference_date: Optional[datetime] = Field(
        None,
        description=(
            "Specify an extra reference datetime in UTC. Adding this lets you"
            " to create and unwrap two single reference networks; the later resets at"
            " the given date (e.g. for a large earthquake event). If passing strings,"
            " formats accepted are YYYY-MM-DD[T]HH:MM[:SS[.ffffff]][Z or [±]HH[:]MM],"
            " or YYYY-MM-DD"
        ),
    )

    # validators
    @field_validator("bounds_wkt", mode="after")
    @classmethod
    def _read_wkt_file(cls, bounds_wkt: str):
        if bounds_wkt and bounds_wkt.endswith(".wkt"):
            return Path(bounds_wkt).read_text()
        return bounds_wkt

    @field_validator("bounds", mode="after")
    @classmethod
    def _check_and_convert_bounds(cls, bounds, info):
        bounds_wkt = info.data.get("bounds_wkt")
        if bounds is not None and bounds_wkt is not None:
            msg = "Cannot specify both bounds and bounds_wkt."
            raise ValueError(msg)
        if bounds:
            return Bbox(*bounds)
        return bounds

    @field_validator("strides")
    @classmethod
    def _check_strides_against_res(cls, strides, info):
        """Compute the output resolution from the strides."""
        resolution = info.data.get("output_resolution")
        if strides is not None and resolution is not None:
            msg = "Cannot specify both strides and output_resolution."
            raise ValueError(msg)
        elif strides is None and resolution is None:
            msg = "Must specify either strides or output_resolution."
            raise ValueError(msg)

        # Check that the dict has the correct keys
        if strides is not None:
            if set(strides.keys()) != {"x", "y"}:
                msg = "Strides must be a dict with keys 'x' and 'y'"
                raise ValueError(msg)
            # and that the strides are integers
            if not all(isinstance(v, int) for v in strides.values()):
                msg = "Strides must be integers"
                raise ValueError(msg)
        if resolution is not None:
            if set(resolution.keys()) != {"x", "y"}:
                msg = "Resolution must be a dict with keys 'x' and 'y'"
                raise ValueError(msg)
            # and that the resolution is valid, > 0. Can be int or float
            if any(v <= 0 for v in resolution.values()):
                msg = "Resolutions must be > 0"
                raise ValueError(msg)
            # TODO: compute strides from resolution
            msg = "output_resolution not yet implemented. Use `strides`."
            raise NotImplementedError(msg)
        return strides

    @field_validator("extra_reference_date", mode="after")
    @classmethod
    def _strip_timezone(cls, extra_reference_date):
        if extra_reference_date is None:
            return None
        return extra_reference_date.replace(tzinfo=None)


class WorkflowBase(YamlModel):
    """Base of multiple workflow configuration models."""

    # Paths to input/output files
    input_options: InputOptions = Field(default_factory=InputOptions)

    mask_file: Optional[Path] = Field(
        None,
        description=(
            "Mask file used to ignore low correlation/bad data (e.g water mask)."
            " Convention is 0 for no data/invalid, and 1 for good data. Dtype must be"
            " uint8."
        ),
    )
    work_directory: Path = Field(
        Path(),
        description="Name of sub-directory to use for writing output files",
        validate_default=True,
    )
    keep_paths_relative: bool = Field(
        False,
        description=(
            "Don't resolve filepaths that are given as relative to be absolute."
        ),
    )

    # General workflow metadata
    worker_settings: WorkerSettings = Field(default_factory=WorkerSettings)
    log_file: Optional[Path] = Field(
        default=None,
        description=(
            "Path to output log file (in addition to logging to `stderr`)."
            " Default logs to `dolphin.log` within `work_directory`"
        ),
    )
    creation_time_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Time the config file was created",
    )

    model_config = ConfigDict(extra="allow")
    _dolphin_version: str = PrivateAttr(_dolphin_version)
    # internal helpers
    # Stores the list of directories to be created by the workflow
    _directory_list: list[Path] = PrivateAttr(default_factory=list)

    def model_post_init(self, context: Any, /) -> None:
        """After validation, set up properties for use during workflow run."""
        super().model_post_init(context)
        # Ensure outputs from workflow steps are within work directory.
        if not self.keep_paths_relative:
            # Save all directories as absolute paths
            self.work_directory = self.work_directory.resolve(strict=False)

    def create_dir_tree(self) -> None:
        """Create the directory tree for the workflow."""
        for d in self._directory_list:
            logger.debug(f"Creating directory: {d}")
            d.mkdir(parents=True, exist_ok=True)


def _read_file_list_or_glob(cls, value):  # noqa: ARG001:
    """Check if the input file list is a glob pattern or a text file.

    If it's a text file, read the lines and return a list of Path objects.
    If it's a string representing a glob pattern, return a list of Path objects.

    Parameters
    ----------
    cls
        pydantic model class
    value : str | Path | list[str] | list[Path]
        Value passed to pydantic model: Input file list.

    """
    if value is None:
        return []

    # Check if they've passed a glob pattern
    if (
        isinstance(value, (list, tuple))
        and (len(value) == 1)
        and glob.has_magic(str(value[0]))
    ):
        value = glob.glob(str(value[0]))
    elif isinstance(value, (str, Path)):
        v_path = Path(value)

        # Check if it's a newline-delimited list of input files
        if glob.has_magic(str(value)):
            value = glob.glob(str(value))
        elif v_path.exists() and v_path.is_file():
            filenames = [Path(f) for f in v_path.read_text().splitlines()]

            # If given as relative paths, make them relative to the text file
            parent = v_path.parent
            return [parent / f if not f.is_absolute() else f for f in filenames]
        else:
            msg = f"Input file list {v_path} does not exist or is not a file."
            raise ValueError(msg)

    return list(value)
