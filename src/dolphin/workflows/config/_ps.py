from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field, field_validator

from ._common import (
    InputOptions,
    OutputOptions,
    PsOptions,
    WorkflowBase,
    _read_file_list_or_glob,
)

__all__ = [
    "PsWorkflow",
]

logger = logging.getLogger("dolphin")


class PsWorkflow(WorkflowBase):
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

    # Options for each step in the workflow
    ps_options: PsOptions = Field(default_factory=PsOptions)
    output_options: OutputOptions = Field(default_factory=OutputOptions)
    layover_shadow_mask_files: list[Path] = Field(
        default_factory=list,
        description=(
            "Paths to layover/shadow binary masks, where 0 indicates a pixel in"
            " layover/shadow, 1 is a good pixel. If none provided, no masking is"
            " performed for layover/shadow."
        ),
    )

    # internal helpers
    model_config = ConfigDict(
        extra="allow", json_schema_extra={"required": ["cslc_file_list"]}
    )

    # validators
    # reuse the _read_file_list_or_glob
    _check_cslc_file_glob = field_validator("cslc_file_list", mode="before")(
        _read_file_list_or_glob
    )

    def model_post_init(self, context: Any, /) -> None:
        """After validation, set up properties for use during workflow run."""
        super().model_post_init(context)
        # Ensure outputs from workflow steps are within work directory.
        if not self.keep_paths_relative:
            # Resolve all CSLC paths:
            self.cslc_file_list = [p.resolve(strict=False) for p in self.cslc_file_list]

        work_dir = self.work_directory
        # move the folders inside the work directory
        ps_opts = self.ps_options
        if ps_opts._directory.parent != work_dir:
            ps_opts._directory = work_dir / ps_opts._directory
        if not self.keep_paths_relative:
            ps_opts._directory = ps_opts._directory.resolve(strict=False)

        # Track the directories that need to be created at start of workflow
        self._directory_list = [
            work_dir,
            self.ps_options._directory,
        ]
        # Add the output PS files we'll create to the `PS` directory, making
        # sure they're inside the work directory
        ps_opts._amp_dispersion_file = work_dir / ps_opts._amp_dispersion_file
        ps_opts._amp_mean_file = work_dir / ps_opts._amp_mean_file
        ps_opts._output_file = work_dir / ps_opts._output_file
