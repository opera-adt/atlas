from __future__ import annotations

# from dolphin._dates import get_dates, sort_files_by_date
import logging
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field, field_validator

from ._common import (
    InputOptions,
    OutputOptions,
    WorkflowBase,
    _read_file_list_or_glob,
)
from ._unwrap_options import UnwrapOptions

__all__ = [
    "UnwrappingWorkflow",
]

logger = logging.getLogger("dolphin")


class UnwrappingWorkflow(WorkflowBase):
    """Configuration for the unwrapping stage of the workflow."""

    # Paths to input/output files
    ifg_file_list: list[Path] = Field(
        default_factory=list,
        description=(
            "list of CSLC files, or newline-delimited file "
            "containing list of CSLC files."
        ),
    )
    input_options: InputOptions = Field(default_factory=InputOptions)
    output_options: OutputOptions = Field(default_factory=OutputOptions)

    unwrap_options: UnwrapOptions = Field(default_factory=UnwrapOptions)

    # internal helpers
    # Stores the list of directories to be created by the workflow
    model_config = ConfigDict(
        extra="allow", json_schema_extra={"required": ["ifg_file_list"]}
    )

    # validators
    # reuse the _read_file_list_or_glob
    _check_cslc_file_glob = field_validator("ifg_file_list", mode="before")(
        _read_file_list_or_glob
    )

    def model_post_init(self, context: Any, /) -> None:
        """After validation, set up properties for use during workflow run."""
        super().model_post_init(context)
        # Ensure outputs from workflow steps are within work directory.
        if not self.keep_paths_relative:
            # Resolve all CSLC paths:
            self.ifg_file_list = [p.resolve(strict=False) for p in self.ifg_file_list]

        work_dir = self.work_directory
        # move output dir inside the work directory (if it's not already inside).
        # They may already be inside if we're loading from a json/yaml file.
        opts = self.unwrap_options
        if opts._directory.parent != work_dir:
            opts._directory = work_dir / opts._directory
        if not self.keep_paths_relative:
            opts._directory = opts._directory.resolve(strict=False)

        # Track the directories that need to be created at start of workflow
        self._directory_list = [
            work_dir,
            self.unwrap_options._directory,
        ]
