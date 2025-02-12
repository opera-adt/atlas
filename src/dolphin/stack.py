#!/usr/bin/env python
from __future__ import annotations

import json
import logging
import warnings
from datetime import date, datetime
from enum import Enum
from os import fspath
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from opera_utils import get_dates
from pydantic import BaseModel, Field, field_validator, model_validator

from dolphin._types import Filename
from dolphin.io import DEFAULT_DATETIME_FORMAT
from dolphin.utils import format_dates

DateOrDatetime = datetime | date
logger = logging.getLogger("dolphin")


class CompressedSlcPlan(str, Enum):
    """Plan for creating Compressed SLCs during phase linking."""

    ALWAYS_FIRST = "always_first"
    FIRST_PER_MINISTACK = "first_per_ministack"
    LAST_PER_MINISTACK = "last_per_ministack"


class BaseStack(BaseModel):
    """Base class for mini- and full stack classes."""

    file_list: list[Filename] = Field(
        ...,
        description="List of SLC filenames in the ministack.",
    )
    dates: list[Sequence[datetime]] = Field(
        ...,
        description=(
            "List of date sequences, one for each SLC in the ministack. "
            "Each item is a list/tuple of datetime.date or datetime.datetime objects, "
            "as returned by [opera_utils.get_dates][]."
        ),
    )
    is_compressed: list[bool] = Field(
        ...,
        description=(
            "List of booleans indicating whether each SLC is compressed or real."
        ),
    )
    file_date_fmt: str = Field(
        DEFAULT_DATETIME_FORMAT,
        description="Format string for the dates/datetimes in the ministack filenames.",
    )
    output_folder: Path = Field(
        Path(),
        description="Folder/location where ministack will write outputs to.",
    )

    model_config = {
        # For the `Filename, so it can handle the `GeneralPath` protocol`
        # https://github.com/pydantic/pydantic/discussions/5767
        "arbitrary_types_allowed": True
    }

    @field_validator("dates", mode="before")
    @classmethod
    def _check_if_not_tuples(cls, v):
        if isinstance(v[0], (date, datetime)):
            v = [[d] for d in v]
        return v

    @model_validator(mode="after")
    def _check_lengths(self):
        if len(self.file_list) == 0:
            msg = "Cannot create empty ministack"
            raise ValueError(msg)
        elif len(self.file_list) == 1:
            warnings.warn("Creating ministack with only one SLC", stacklevel=2)
        if len(self.file_list) != len(self.is_compressed):
            lengths = f"{len(self.file_list)} and {len(self.is_compressed)}"
            msg = f"file_list and is_compressed must be the same length: Got {lengths}"
            raise ValueError(msg)
        if len(self.dates) != len(self.file_list):
            lengths = f"{len(self.dates)} and {len(self.file_list)}"
            msg = f"dates and file_list must be the same length. Got {lengths}"
            raise ValueError(msg)
        return self

    @property
    def first_real_slc_idx(self) -> int:
        """Index of the first real SLC in the ministack."""
        try:
            return np.where(~np.array(self.is_compressed))[0][0]
        except IndexError as e:
            msg = "No real SLCs in ministack"
            raise ValueError(msg) from e

    @property
    def real_slc_date_range(self) -> tuple[DateOrDatetime, DateOrDatetime]:
        """Date range of the real SLCs in the ministack."""
        return (self.dates[self.first_real_slc_idx][0], self.dates[-1][-1])

    @property
    def real_slc_date_range_str(self) -> str:
        """Date range of the real SLCs in the ministack."""
        return format_dates(*self.real_slc_date_range, fmt=self.file_date_fmt)

    @property
    def compressed_slc_file_list(self) -> list[Filename]:
        """List of compressed SLCs for this ministack."""
        return [f for f, is_comp in zip(self.file_list, self.is_compressed) if is_comp]

    @property
    def real_slc_file_list(self) -> list[Filename]:
        """List of real SLCs for this ministack."""
        return [
            f for f, is_comp in zip(self.file_list, self.is_compressed) if not is_comp
        ]

    def get_date_str_list(self) -> list[str]:
        """Get a formatted string for each date/date tuple in the ministack."""
        # Should either be like YYYYMMDD, or YYYYMMDD_YYYYMMDD_YYYYMMDD
        return [format_dates(*d, fmt=self.file_date_fmt) for d in self.dates]

    def __rich_repr__(self):
        yield "file_list", self.file_list
        yield "dates", self.dates
        yield "is_compressed", self.is_compressed
        yield "file_date_fmt", self.file_date_fmt
        yield "output_folder", self.output_folder


class CompressedSlcInfo(BaseModel):
    """Class for holding attributes about one compressed SLC."""

    reference_date: datetime = Field(
        ...,
        description=(
            "Reference date for understanding output interferograms. Note that this may"
            " be different from `start_date` (the first real SLC which  was used in the"
            " compression)."
        ),
    )
    start_date: datetime = Field(
        ..., description="Datetime of the first real SLC used in the compression."
    )
    end_date: datetime = Field(
        ..., description="Datetime of the last real SLC used in the compression."
    )

    real_slc_file_list: Optional[list[Filename]] = Field(
        None,
        description="List of real SLC filenames in the ministack.",
    )
    real_slc_dates: Optional[list[datetime]] = Field(
        None,
        description=(
            "List of date sequences, one for each SLC in the ministack. "
            "Each item is a list/tuple of datetime.date or datetime.datetime objects."
        ),
    )
    compressed_slc_file_list: Optional[list[Filename]] = Field(
        None,
        description="List of compressed SLC filenames in the ministack.",
    )
    file_date_fmt: str = Field(
        DEFAULT_DATETIME_FORMAT,
        description="Format string for the dates/datetimes in the ministack filenames.",
    )
    filename_template: str = Field(
        "compressed_{date_str}.tif",
        description="Template for creating filenames from the CCSLC date triplet.",
    )
    output_folder: Path = Field(
        Path(),
        description="Folder/location where ministack will write outputs to.",
    )

    model_config = {
        # For the `Filename, so it can handle the `GeneralPath` protocol`
        # https://github.com/pydantic/pydantic/discussions/5767
        "arbitrary_types_allowed": True
    }

    @field_validator("real_slc_dates", mode="before")
    @classmethod
    def _untuple_dates(cls, v):
        """Make the dates not be tuples/lists of datetimes."""
        if v is None:
            return v
        out = []
        for item in v:
            if hasattr(item, "__iter__"):
                # Make sure they didn't pass more than 1 date, implying
                # a compressed SLC
                # assert len(item) == 1
                if isinstance(item, str):
                    out.append(item)
                elif len(item) > 1:
                    msg = f"Cannot pass multiple dates for a compressed SLC. Got {item}"
                    raise ValueError(msg)
                else:
                    out.append(item[0])
            else:
                out.append(item)
        return out

    @model_validator(mode="after")
    def _check_lengths(self):
        if self.real_slc_dates is None or self.real_slc_file_list is None:
            return self
        rlen = len(self.real_slc_file_list)
        clen = len(self.real_slc_dates)
        if rlen != clen:
            lengths = f"{rlen} and {clen}"
            msg = (
                "real_slc_file_list and real_slc_dates must be the same length. "
                f"Got {lengths}"
            )
            raise ValueError(msg)
        return self

    @property
    def dates(self) -> tuple[DateOrDatetime, DateOrDatetime, DateOrDatetime]:
        """Alias for the (reference, start, end) date triplet."""
        return (self.reference_date, self.start_date, self.end_date)

    @property
    def real_date_range(self) -> tuple[DateOrDatetime, DateOrDatetime]:
        """Date range of the real SLCs in the ministack."""
        return self.start_date, self.end_date

    @property
    def filename(self) -> str:
        """Create filename using a template with '{date_str}` in the name."""
        date_str = format_dates(*self.dates, fmt=self.file_date_fmt)
        return self.filename_template.format(date_str=date_str)

    @property
    def path(self) -> Path:
        """The path of the compressed SLC for this ministack."""
        return self.output_folder / self.filename

    def write_metadata(
        self, domain: str = "DOLPHIN", output_file: Optional[Filename] = None
    ):
        """Write the metadata to the compressed SLC file.

        Parameters
        ----------
        domain : str, optional
            Domain to write the metadata to, by default "DOLPHIN".
        output_file : Optional[Filename], optional
            Path to the file to write the metadata to, by default None.
            If None, will use `self.path`.

        """
        from dolphin.io import set_raster_metadata

        out = self.path if output_file is None else Path(output_file)
        if not out.exists():
            msg = f"Must create {out} before writing metadata"
            raise FileNotFoundError(msg)

        set_raster_metadata(
            out,
            metadata=self.model_dump(mode="json"),
            domain=domain,
        )

    @classmethod
    def from_filename(
        cls, filename: Filename, date_fmt: str = DEFAULT_DATETIME_FORMAT
    ) -> CompressedSlcInfo:
        """Parse just the dates from a compressed SLC filename."""
        try:
            ref, start, end = get_dates(filename, fmt=date_fmt)[0:3]
        except IndexError as e:
            msg = f"{filename} does not have 3 dates like {date_fmt}"
            raise ValueError(msg) from e
        output_folder = Path(filename).parent
        fname = Path(filename).name

        # Use whichever filename was passed in by replacing the dates we parsed
        _date_str = format_dates(ref, start, end, fmt=date_fmt)
        filename_template = fname.replace(_date_str, "{date_str}")
        return cls(
            reference_date=ref,
            start_date=start,
            end_date=end,
            output_folder=output_folder,
            file_date_fmt=date_fmt,
            filename_template=filename_template,
        )

    @classmethod
    def from_file_metadata(cls, filename: Filename) -> CompressedSlcInfo:
        """Try to parse the CCSLC metadata from `filename`."""
        from dolphin.io import get_raster_metadata

        if not Path(filename).exists():
            raise FileNotFoundError(filename)

        domains = ["DOLPHIN", ""]
        for domain in domains:
            gdal_md = get_raster_metadata(filename, domain=domain)
            if not gdal_md:
                continue
            else:
                break
        else:
            msg = f"Could not find metadata in {filename}"
            raise ValueError(msg)
        # GDAL can write it weirdly and mess up the JSON
        cleaned = {}
        for k, v in gdal_md.items():
            try:
                # Swap the single quotes for double quotes to parse lists
                cleaned[k] = json.loads(v.replace("'", '"'))
            except json.JSONDecodeError:
                cleaned[k] = v
        # Parse the date/file lists from the metadata
        out = cls.model_validate(cleaned)
        # Overwrite the `output_folder` part- we may have moved it since
        # writing the metadata
        out.output_folder = Path(filename).parent
        return out

    def __fspath__(self):
        return fspath(self.path)


class MiniStackInfo(BaseStack):
    """Class for holding attributes about one mini-stack of SLCs.

    Used for planning the processing of a batch of SLCs.
    """

    output_reference_idx: int = Field(
        0,
        description="Index of the SLC to use as reference during phase linking",
    )
    compressed_reference_idx: int = Field(
        0,
        description=(
            "Index of the SLC to use as during compressed SLC creation. May be"
            " different than `output_reference_idx`."
        ),
    )

    def __rich_repr__(self):
        yield from super().__rich_repr__()
        yield "output_reference_idx", self.output_reference_idx
        yield "compressed_reference_idx", self.compressed_reference_idx

    @property
    def output_reference_date(self):
        """Date of the reference phase of the stack."""
        # Note this works for either a length-1 tuple (real SLC), or for
        # the compressed SLC formate (ref, start, end)
        return self.dates[self.output_reference_idx][0]

    @property
    def compressed_reference_date(self):
        """Date of the reference phase of the stack."""
        # Note this works for either a length-1 tuple (real SLC), or for
        # the compressed SLC formate (ref, start, end)
        return self.dates[self.compressed_reference_idx][0]

    def get_compressed_slc_info(self) -> CompressedSlcInfo:
        """Get the compressed SLC which will come from this ministack.

        Excludes the existing compressed SLCs during the compression.
        """
        real_slc_files: list[Filename] = []
        real_slc_dates: list[Sequence[DateOrDatetime]] = []
        comp_slc_files: list[Filename] = []
        for f, d, is_comp in zip(self.file_list, self.dates, self.is_compressed):
            if is_comp:
                comp_slc_files.append(f)
            else:
                real_slc_files.append(f)
                real_slc_dates.append(d)

        return CompressedSlcInfo(
            reference_date=self.compressed_reference_date,
            start_date=real_slc_dates[0][0],
            end_date=real_slc_dates[-1][0],
            real_slc_file_list=real_slc_files,
            real_slc_dates=real_slc_dates,
            compressed_slc_file_list=comp_slc_files,
            file_date_fmt=self.file_date_fmt,
            output_folder=self.output_folder,
        )


class MiniStackPlanner(BaseStack):
    """Class for planning the processing of batches of SLCs."""

    max_num_compressed: int = 5
    output_reference_idx: int = Field(
        0,
        description="Index of the SLC to use as reference during phase linking",
    )
    compressed_slc_plan: CompressedSlcPlan = CompressedSlcPlan.ALWAYS_FIRST

    def plan(
        self, ministack_size: int, compressed_idx: int | None = None
    ) -> list[MiniStackInfo]:
        """Create a list of ministacks to be processed."""
        if ministack_size < 2:
            msg = "Cannot create ministacks with size < 2"
            raise ValueError(msg)

        # For now, only allow `compressed_idx` when doing a single batch.
        # The logic is more complicated for multiple `compressed_idx`s, and
        # it's unclear who would need that
        if compressed_idx is not None and ministack_size < len(self.file_list):
            raise ValueError(
                "Cannot set `compressed_idx` when creating multiple ministacks."
            )

        output_ministacks: list[MiniStackInfo] = []

        # Start of with any compressed SLCs that are passed in
        compressed_slc_infos: list[CompressedSlcInfo] = []
        for f in self.compressed_slc_file_list:
            # TODO: will we ever actually need to read the old metadata here?
            compressed_slc_infos.append(CompressedSlcInfo.from_filename(f))

        # Solve each ministack using current chunk (and the previous compressed SLCs)
        ministack_starts = range(
            self.first_real_slc_idx, len(self.file_list), ministack_size
        )

        for full_stack_idx in ministack_starts:
            cur_slice = slice(full_stack_idx, full_stack_idx + ministack_size)
            cur_files = list(self.file_list[cur_slice]).copy()
            cur_dates = list(self.dates[cur_slice]).copy()

            # Read compressed*.tif files and if they do not exist use the compressed*.h5
            comp_slc_files = [c.path for c in compressed_slc_infos]
            # Add the existing compressed SLC files to the start, but
            # limit the num comp slcs `max_num_compressed`
            cur_comp_slc_files = comp_slc_files[-self.max_num_compressed :]
            combined_files = cur_comp_slc_files + cur_files

            combined_dates = [
                c.dates for c in compressed_slc_infos[-self.max_num_compressed :]
            ] + cur_dates

            num_ccslc = len(cur_comp_slc_files)
            combined_is_compressed = num_ccslc * [True] + list(
                self.is_compressed[cur_slice]
            )

            # Make the current ministack output folder using the start/end dates
            new_date_str = format_dates(
                cur_dates[0][0], cur_dates[-1][-1], fmt=self.file_date_fmt
            )
            cur_output_folder = self.output_folder / new_date_str

            if compressed_idx is not None:
                compressed_reference_idx = compressed_idx
            elif self.compressed_slc_plan == CompressedSlcPlan.ALWAYS_FIRST:
                # Simplest operational version: CompSLCs have same base phase,
                # but different "residual" added on
                # We use the `output_reference_idx`, 0 by default, but this index
                # may be passed in if we are manually specifying an output
                compressed_reference_idx = self.output_reference_idx
            elif self.compressed_slc_plan == CompressedSlcPlan.FIRST_PER_MINISTACK:
                # Like Ansari, 2017 paper: each ministack is "self contained"
                compressed_reference_idx = num_ccslc
                # Ansari, 2017 also had output_reference_idx = num_ccslcs, and
                # used the "Datum Adjustment" step to get outputs relative to day 0
                # Here, we'll use 0 (or manually specifed)
            elif self.compressed_slc_plan == CompressedSlcPlan.LAST_PER_MINISTACK:
                # Alternative that allows sequential interferograms across ministacks
                compressed_reference_idx = -1

            if self.compressed_slc_plan == CompressedSlcPlan.LAST_PER_MINISTACK:
                # For this, we'll always use the most recent compressed SLC as output
                # reference so that we can connected stacks, but minimize the temporal
                # baseline of interferograms we form
                output_reference_idx = num_ccslc - 1
            else:
                output_reference_idx = self.output_reference_idx

            cur_ministack = MiniStackInfo(
                file_list=combined_files,
                dates=combined_dates,
                is_compressed=combined_is_compressed,
                output_reference_idx=output_reference_idx,
                compressed_reference_idx=compressed_reference_idx,
                output_folder=cur_output_folder,
            )

            output_ministacks.append(cur_ministack)
            cur_comp_slc = cur_ministack.get_compressed_slc_info()
            compressed_slc_infos.append(cur_comp_slc)

        return output_ministacks

    def __rich_repr__(self):
        yield from super().__rich_repr__()
        yield "max_num_compressed", self.max_num_compressed
        yield "output_reference_idx", self.output_reference_idx
        yield "compressed_slc_plan", self.compressed_slc_plan
