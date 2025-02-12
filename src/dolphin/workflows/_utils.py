from __future__ import annotations

import contextlib
import datetime
import logging
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from opera_utils import group_by_date

from dolphin._types import Filename

from .config import DisplacementWorkflow

logger = logging.getLogger("dolphin")


def _create_burst_cfg(
    cfg: DisplacementWorkflow,
    burst_id: str,
    grouped_slc_files: dict[str, list[Path]],
    grouped_amp_mean_files: dict[str, list[Path]],
    grouped_amp_dispersion_files: dict[str, list[Path]],
    grouped_layover_shadow_mask_files: dict[str, list[Path]],
) -> DisplacementWorkflow:
    cfg_temp_dict = cfg.model_dump(exclude={"cslc_file_list"})

    # Just update the inputs and the work directory
    top_level_work = cfg_temp_dict["work_directory"]
    cfg_temp_dict.update({"work_directory": top_level_work / burst_id})
    cfg_temp_dict["cslc_file_list"] = grouped_slc_files[burst_id]
    cfg_temp_dict["amplitude_mean_files"] = grouped_amp_mean_files[burst_id]
    cfg_temp_dict["amplitude_dispersion_files"] = grouped_amp_dispersion_files[burst_id]
    cfg_temp_dict["layover_shadow_mask_files"] = grouped_layover_shadow_mask_files[
        burst_id
    ]
    return DisplacementWorkflow(**cfg_temp_dict)


def _remove_dir_if_empty(d: Path) -> None:
    with contextlib.suppress(OSError):
        d.rmdir()


def parse_ionosphere_files(
    ionosphere_files: Sequence[Filename],
    iono_date_fmts: Sequence[str] = ["%j0.%y", "%Y%j0000"],
) -> Mapping[tuple[datetime.datetime], list[Path]]:
    """Parse ionosphere files and group them by date.

    Parameters
    ----------
    ionosphere_files : Sequence[Union[str, Path]]
        List of ionosphere file paths.
    iono_date_fmts: Sequence[str]
        Format of dates within ionosphere file names to search for.
        Default is ["%j0.%y", "%Y%j0000"], which matches the old name
        'jplg2970.16i', and the new name format
        'JPL0OPSFIN_20232540000_01D_02H_GIM.INX' (respectively)


    Returns
    -------
    grouped_iono_files : Mapping[Tuple[datetime], List[Path]]
        Dictionary mapping dates to lists of files.

    """
    grouped_iono_files: Mapping[tuple[datetime.datetime], list[Path]] = defaultdict(
        list
    )
    for fmt in iono_date_fmts:
        group_iono = group_by_date(ionosphere_files, file_date_fmt=fmt)
        if () in group_iono:  # files which didn't match the date format
            group_iono.pop(())
        grouped_iono_files = {**grouped_iono_files, **group_iono}

    return grouped_iono_files
