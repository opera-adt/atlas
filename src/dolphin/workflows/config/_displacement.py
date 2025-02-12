from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any, Optional

from opera_utils import get_burst_id, get_dates, sort_files_by_date
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from dolphin import constants
from dolphin._types import TropoModel, TropoType

from ._common import (
    InputOptions,
    InterferogramNetwork,
    OutputOptions,
    PhaseLinkingOptions,
    PsOptions,
    TimeseriesOptions,
    WorkflowBase,
    _read_file_list_or_glob,
)
from ._unwrap_options import UnwrapMethod, UnwrapOptions

__all__ = [
    "CorrectionOptions",
    "DisplacementWorkflow",
]

logger = logging.getLogger("dolphin")


# Add a class for troposphere, ionosphere corrections, with geometry files and DEM
class CorrectionOptions(BaseModel, extra="forbid"):
    """Configuration for the auxillary phase corrections."""

    _atm_directory: Path = Path("atmosphere")
    _iono_date_fmt: list[str] = ["%j0.%y", "%Y%j0000"]

    troposphere_files: list[Path] = Field(
        default_factory=list,
        description=(
            "List of weather-model files (one per date) for tropospheric corrections"
        ),
    )

    tropo_date_fmt: str = Field(
        "%Y%m%d",
        description="Format of dates contained in weather-model filenames",
    )

    tropo_package: Annotated[str, StringConstraints(to_lower=True)] = Field(
        "pyaps",
        description="Package for tropospheric correction. Choices: pyaps, raider",
    )

    tropo_model: TropoModel = Field(
        TropoModel.ECMWF, description="source of the atmospheric model."
    )

    tropo_delay_type: TropoType = Field(
        TropoType.COMB,
        description=(
            "Tropospheric delay type to calculate, comb contains both wet "
            "and dry delays."
        ),
    )

    ionosphere_files: list[Path] = Field(
        default_factory=list,
        description=(
            "List of GNSS-derived TEC maps for ionospheric corrections (one per date)."
            " Source is https://cddis.nasa.gov/archive/gnss/products/ionex/"
        ),
    )

    geometry_files: list[Path] = Field(
        default_factory=list,
        description=(
            "Line-of-sight geometry files for each burst/SLC stack area, for use in"
            " correction computations."
        ),
    )
    dem_file: Optional[Path] = Field(
        None,
        description="DEM file for tropospheric/ topographic phase corrections.",
    )

    @field_validator(
        "troposphere_files", "ionosphere_files", "geometry_files", mode="before"
    )
    @classmethod
    def _to_empty_list(cls, v):
        return v if v is not None else []


class DisplacementWorkflow(WorkflowBase):
    """Configuration for the workflow."""

    # Paths to input/output files
    input_options: InputOptions = Field(default_factory=InputOptions)
    cslc_file_list: list[Path] = Field(
        default_factory=list,
        description=(
            "list of CSLC files, or newline-delimited file "
            "containing list of CSLC files."
        ),
    )
    output_options: OutputOptions = Field(default_factory=OutputOptions)

    # Options for each step in the workflow
    ps_options: PsOptions = Field(default_factory=PsOptions)
    amplitude_dispersion_files: list[Path] = Field(
        default_factory=list,
        description=(
            "Paths to existing Amplitude Dispersion file (1 per SLC region) for PS"
            " update calculation. If none provided, computed using the input SLC stack."
        ),
    )
    amplitude_mean_files: list[Path] = Field(
        default_factory=list,
        description=(
            "Paths to an existing Amplitude Mean files (1 per SLC region) for PS update"
            " calculation. If none provided, computed using the input SLC stack."
        ),
    )
    layover_shadow_mask_files: list[Path] = Field(
        default_factory=list,
        description=(
            "Paths to layover/shadow binary masks, where 0 indicates a pixel in"
            " layover/shadow, 1 is a good pixel. If none provided, no masking is"
            " performed for layover/shadow."
        ),
    )

    phase_linking: PhaseLinkingOptions = Field(default_factory=PhaseLinkingOptions)
    interferogram_network: InterferogramNetwork = Field(
        default_factory=InterferogramNetwork
    )
    unwrap_options: UnwrapOptions = Field(default_factory=UnwrapOptions)
    timeseries_options: TimeseriesOptions = Field(default_factory=TimeseriesOptions)
    correction_options: CorrectionOptions = Field(default_factory=CorrectionOptions)

    # internal helpers
    # Stores the list of directories to be created by the workflow
    model_config = ConfigDict(
        extra="allow", json_schema_extra={"required": ["cslc_file_list"]}
    )

    # validators
    # reuse the _read_file_list_or_glob
    _check_cslc_file_glob = field_validator("cslc_file_list", mode="before")(
        _read_file_list_or_glob
    )

    @model_validator(mode="after")
    def _check_input_files_exist(self) -> DisplacementWorkflow:
        file_list = self.cslc_file_list
        if not file_list:
            msg = "Must specify list of input SLC files."
            raise ValueError(msg)

        input_options = self.input_options
        date_fmt = input_options.cslc_date_fmt
        # Filter out files that don't have dates in the filename
        files_matching_date = [Path(f) for f in file_list if get_dates(f, fmt=date_fmt)]
        if len(files_matching_date) < len(file_list):
            msg = (
                f"Found {len(files_matching_date)} files with dates like {date_fmt} in"
                f" the filename out of {len(file_list)} files."
            )
            raise ValueError(msg)

        ext = file_list[0].suffix
        # If they're HDF5/NetCDF files, we need to check that the subdataset exists
        if ext in [".h5", ".nc"]:
            subdataset = input_options.subdataset
            if subdataset is None:
                msg = "Must provide subdataset name for input NetCDF/HDF5 files."
                raise ValueError(msg)

        # Coerce the file_list to a sorted list of Path objects
        self.cslc_file_list = [
            Path(f) for f in sort_files_by_date(file_list, file_date_fmt=date_fmt)[0]
        ]

        return self

    def model_post_init(self, context: Any, /) -> None:
        """After validation, set up properties for use during workflow run."""
        super().model_post_init(context)

        if self.input_options.wavelength is None and self.cslc_file_list:
            # Try to infer the wavelength from filenames
            try:
                get_burst_id(self.cslc_file_list[-1])
                # The Burst ID was recognized for OPERA-S1 SLCs: use S1 wavelength
                self.input_options.wavelength = constants.SENTINEL_1_WAVELENGTH
            except ValueError:
                pass

        # Ensure outputs from workflow steps are within work directory.
        if not self.keep_paths_relative:
            # Resolve all CSLC paths:
            self.cslc_file_list = [p.resolve(strict=False) for p in self.cslc_file_list]

        work_dir = self.work_directory
        # For each workflow step that has an output folder, move it inside
        # the work directory (if it's not already inside).
        # They may already be inside if we're loading from a json/yaml file.
        for step in [
            "ps_options",
            "phase_linking",
            "interferogram_network",
            "unwrap_options",
            "timeseries_options",
        ]:
            opts = getattr(self, step)
            if isinstance(opts, dict):
                # If this occurs, we are printing the schema.
                # Using newer pydantic `model_construct`, this would be a dict,
                # instead of an object.
                # We don't care about the subsequent logic here
                return

            if opts._directory.parent != work_dir:
                opts._directory = work_dir / opts._directory
            if not self.keep_paths_relative:
                opts._directory = opts._directory.resolve(strict=False)

        # Track the directories that need to be created at start of workflow
        self._directory_list = [
            work_dir,
            self.ps_options._directory,
            self.phase_linking._directory,
            self.interferogram_network._directory,
            self.unwrap_options._directory,
            self.timeseries_options._directory,
        ]
        # Add the output PS files we'll create to the `PS` directory, making
        # sure they're inside the work directory
        ps_opts = self.ps_options
        ps_opts._amp_dispersion_file = work_dir / ps_opts._amp_dispersion_file
        ps_opts._amp_mean_file = work_dir / ps_opts._amp_mean_file
        ps_opts._output_file = work_dir / ps_opts._output_file

        self.timeseries_options._velocity_file = (
            work_dir / self.timeseries_options._velocity_file
        )

        # Modify interferogram options if using spurt for 3d unwrapping,
        # which only does nearest-3 interferograms
        if self.unwrap_options.unwrap_method == UnwrapMethod.SPURT:
            logger.info(
                "Using spurt: will form single reference interferograms, later convert"
                " to nearest-3"
            )
            self.interferogram_network.reference_idx = 0
            # Force all other network options to None
            for attr in ["max_bandwidth", "max_temporal_baseline", "indexes"]:
                setattr(self.interferogram_network, attr, None)
