"""Module for creating interferograms and networks of interferograms."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from datetime import date, datetime
from os import fspath
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike
from opera_utils import get_dates
from osgeo import gdal
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from tqdm.contrib.concurrent import thread_map

from dolphin import io, utils
from dolphin._types import Filename, T
from dolphin.filtering import gaussian_filter_nan

gdal.UseExceptions()

logger = logging.getLogger("dolphin")
DateOrDatetime = Union[datetime, date]

DEFAULT_SUFFIX = ".int.vrt"


class VRTInterferogram(BaseModel, extra="allow"):
    """Create an interferogram using a VRTDerivedRasterBand.

    Attributes
    ----------
    ref_slc : Union[Path, str]
        Path to reference SLC file
    sec_slc : Union[Path, str]
        Path to secondary SLC file
    path : Optional[Path], optional
        Path to output interferogram. Defaults to Path('<date1>_<date2>.int.vrt'),
        placed in the same directory as `ref_slc`.
    outdir : Optional[Path], optional
        Directory to place output interferogram. Defaults to the same directory as
        `ref_slc`. If only `outdir` is specified, the output interferogram will
        be named '<date1>_<date2>.int.vrt', where the dates are parsed from the
        inputs. If `path` is specified, this is ignored.
    subdataset : Optional[str], optional
        Subdataset to use for the input files (if passing file paths
        to NETCDF/HDF5 files).
        Defaults to None.
    date_format : str, optional
        Date format to use when parsing dates from the input files.
        Defaults to '%Y%m%d'.
    pixel_function : str, optional
        GDAL Pixel function to use, choices={'cmul', 'mul'}.
        Defaults to 'cmul', which performs `ref_slc * sec_slc.conj()`.
        See https://gdal.org/drivers/raster/vrt.html#default-pixel-functions
    dates : tuple[datetime.date, datetime.date]
        Date of the interferogram (parsed from the input files).

    """

    subdataset: Optional[str] = Field(
        None,
        description="Subdataset to use for the input files. Defaults to None.",
    )
    ref_slc: Union[Path, str] = Field(..., description="Path to reference SLC file")
    sec_slc: Union[Path, str] = Field(..., description="Path to secondary SLC file")
    verify_slcs: bool = Field(
        True, description="Raise an error if `ref_slc` or `sec_slc` aren't readable."
    )
    outdir: Optional[Path] = Field(
        None,
        description=(
            "Directory to place output interferogram. Defaults to the same"
            " directory as `ref_slc`. If only `outdir` is specified, the output"
            f" interferogram will be named '<date1>_<date2>{DEFAULT_SUFFIX}', where the"
            " dates are parsed from the inputs. If `path` is specified, this is ignored"
        ),
        validate_default=True,
    )
    path: Optional[Path] = Field(
        None,
        description=(
            f"Path to output interferogram. Defaults to <date1>_<date2>{DEFAULT_SUFFIX}"
            ", where the dates are parsed from the input files, placed in the same "
            "directory as `ref_slc`."
        ),
        validate_default=True,
    )
    date_format: str = Field(
        io.DEFAULT_DATETIME_FORMAT,
        description="datetime format used to parse SLC filenames",
    )
    ref_date: Optional[DateOrDatetime] = Field(
        None,
        description=(
            "Reference date of the interferogram. If not specified,"
            "will be parsed from `ref_slc` using `date_format`."
        ),
    )
    sec_date: Optional[DateOrDatetime] = Field(
        None,
        description=(
            "Secondary date of the interferogram. If not specified,"
            "will be parsed from `sec_slc` using `date_format`."
        ),
    )
    resolve_paths: bool = Field(
        True, description="Resolve paths of `ref_slc`/`sec_slc` when saving the VRT"
    )
    use_relative: bool = Field(
        False, description='If true, use `relativeToVRT="1" in the VRT'
    )
    write: bool = Field(True, description="Write the VRT file to disk")

    pixel_function: Literal["cmul", "mul"] = "cmul"
    _template = """\
<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
    <VRTRasterBand dataType="CFloat32" band="1" subClass="VRTDerivedRasterBand">
        <PixelFunctionType>{pixel_function}</PixelFunctionType>
        <SimpleSource>
            <SourceFilename relativeToVRT="{rel}">{ref_slc}</SourceFilename>
        </SimpleSource>
        <SimpleSource>
            <SourceFilename relativeToVRT="{rel}">{sec_slc}</SourceFilename>
        </SimpleSource>
    </VRTRasterBand>
</VRTDataset>
    """

    @field_validator("ref_slc", "sec_slc")
    @classmethod
    def _check_gdal_string(cls, v: Union[Path, str], info: ValidationInfo):
        subdataset = info.data.get("subdataset")
        # If we're using a subdataset, create a the GDAL-readable string
        return io.format_nc_filename(v, subdataset)

    @field_validator("outdir")
    @classmethod
    def _check_output_dir(cls, v, info: ValidationInfo):
        if v is not None:
            return Path(v)
        # If outdir is not set, use the directory of the reference SLC
        ref_slc = str(info.data.get("ref_slc"))
        return utils._get_path_from_gdal_str(ref_slc).parent

    @model_validator(mode="after")
    def _parse_dates(self) -> VRTInterferogram:
        # Get the dates from the input files if not provided
        if self.ref_date is None:
            d = get_dates(self.ref_slc, fmt=self.date_format)
            if not d:
                msg = f"No dates found in '{self.ref_slc}' like {self.date_format}"
                raise ValueError(msg)
            self.ref_date = d[0]
        if self.sec_date is None:
            d = get_dates(self.sec_slc, fmt=self.date_format)
            if not d:
                msg = f"No dates found in '{self.sec_slc}' like {self.date_format}"
                raise ValueError(msg)
            self.sec_date = d[0]

        return self

    @model_validator(mode="after")
    def _resolve_files(self) -> VRTInterferogram:
        """Check that the inputs are the same size and geotransform."""
        if not self.ref_slc or not self.sec_slc:
            # Skip validation if files are not set
            return self
        if self.resolve_paths:
            self.ref_slc = utils._resolve_gdal_path(self.ref_slc)  # type: ignore[assignment]
            self.sec_slc = utils._resolve_gdal_path(self.sec_slc)  # type: ignore[assignment]
        return self

    @model_validator(mode="after")
    def _validate_files(self) -> VRTInterferogram:
        # Only run this check if we care to validate the readability
        if not self.verify_slcs:
            return self

        ds1 = gdal.Open(fspath(self.ref_slc))
        ds2 = gdal.Open(fspath(self.sec_slc))
        xsize, ysize = ds1.RasterXSize, ds1.RasterYSize
        xsize2, ysize2 = ds2.RasterXSize, ds2.RasterYSize
        if xsize != xsize2 or ysize != ysize2:
            msg = f"Input files {self.ref_slc} and {self.sec_slc} are not the same size"
            raise ValueError(msg)
        gt1 = ds1.GetGeoTransform()
        gt2 = ds2.GetGeoTransform()
        if gt1 != gt2:
            msg = f"{self.ref_slc} and {self.sec_slc} have different GeoTransforms"
            raise ValueError(msg)

        return self

    @model_validator(mode="after")
    def _form_path(self) -> VRTInterferogram:
        """Create the filename (if not provided) from the provided SLCs."""
        if self.path is not None:
            return self

        if self.outdir is None:
            # If outdir is not set, use the directory of the reference SLC
            self.outdir = utils._get_path_from_gdal_str(self.ref_slc).parent
        assert self.ref_date is not None
        assert self.sec_date is not None
        date_str = utils.format_date_pair(
            self.ref_date, self.sec_date, fmt=self.date_format
        )
        path = self.outdir / (date_str + DEFAULT_SUFFIX)
        self.path = path
        return self

    @model_validator(mode="after")
    def _write_vrt(self) -> VRTInterferogram:
        """Write out the VRT if requested."""
        if not self.write:
            return self
        assert self.path is not None
        if Path(self.path).exists():
            logger.info(f"Removing {self.path}")
            self.path.unlink()

        xsize, ysize = io.get_raster_xysize(self.ref_slc)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            f.write(
                self._template.format(
                    xsize=xsize,
                    ysize=ysize,
                    ref_slc=self.ref_slc,
                    sec_slc=self.sec_slc,
                    pixel_function=self.pixel_function,
                    rel=self.use_relative,
                )
            )
        io.copy_projection(self.ref_slc, self.path)
        return self

    def load(self):
        """Load the interferogram as a numpy array."""
        return io.load_gdal(self.path)

    @property
    def shape(self):  # noqa: D102
        xsize, ysize = io.get_raster_xysize(self.path)
        return (ysize, xsize)

    @classmethod
    def from_vrt_file(cls, path: Filename) -> VRTInterferogram:
        """Load a VRTInterferogram from an existing VRT file.

        Parameters
        ----------
        path : Filename
            Path to VRT file.

        Returns
        -------
        VRTInterferogram
            VRTInterferogram object.

        """
        from dolphin.io._readers import _parse_vrt_file

        # Use the parsing function
        (ref_slc, sec_slc), subdataset = _parse_vrt_file(path)
        if subdataset is not None:
            ref_slc = io.format_nc_filename(ref_slc, subdataset)
            sec_slc = io.format_nc_filename(sec_slc, subdataset)

        return cls.model_construct(
            ref_slc=ref_slc,
            sec_slc=sec_slc,
            path=Path(path).resolve(),
            subdataset=subdataset,
            date_format="%Y%m%d",
        )


# Alias for the pairs of filenames in ifgs
IfgPairT = tuple[Filename, Filename]


@dataclass
class Network:
    """A network of interferograms from a list of SLCs.

    Attributes
    ----------
    slc_list : list[Filename]
        list of SLCs to use to form interferograms.
    ifg_list : list[tuple[Filename, Filename]]
        list of `VRTInterferogram`s created from the SLCs.
    max_bandwidth : int | None, optional
        Maximum number of SLCs to include in an interferogram, by index distance.
        Defaults to None.
    max_temporal_baseline : Optional[float], default = None
        Maximum temporal baseline to include in an interferogram, in days.
    include_annual : bool, default = False
        Attempt to include annual pairs, in addition to the other type of network
        requested. Will skip if no pairs exist within 365 +/- `buffer_days`
    annual_buffer_days : float, default = 30
        Search buffer for finding annual pairs, if `include_annual=True`
    date_format : str, optional
        Date format to use when parsing dates from the input files (only
        used if setting `max_temporal_baseline`).
        defaults to '%Y%m%d'.
    dates: Sequence[datetime | date], optional
        Alternative to `date_format`: manually specify the date/datetime of each item in
        `slc_list` instead of parsing the name.
        Only used for `max_temporal_baseline` networks.
    reference_idx : int | None, optional
        Index of the SLC to use as the reference for all interferograms.
        Defaults to None.
    indexes : Sequence[tuple[int, int]], optional
        Manually list (ref_idx, sec_idx) pairs to use for interferograms.
    subdataset : Optional[str], default = None
        If passing NetCDF files in `slc_list, the subdataset of the image data
        within the file.
        Can also pass a sequence of one subdataset per entry in `slc_list`
    verify_slcs : bool, default = True
        Raise an error if any SLCs aren't GDAL-readable.
    write : bool
        Whether to write the VRT files to disk. Defaults to True.

    """

    slc_list: Sequence[Filename]
    outdir: Optional[Filename] = None
    max_bandwidth: Optional[int] = None
    max_temporal_baseline: Optional[float] = None
    include_annual: bool = False
    annual_buffer_days: float = 30
    date_format: str = io.DEFAULT_DATETIME_FORMAT
    dates: Optional[Sequence[DateOrDatetime]] = None
    reference_idx: Optional[int] = None
    indexes: Optional[Sequence[tuple[int, int]]] = None
    subdataset: Optional[Union[str, Sequence[str]]] = None
    verify_slcs: bool = True
    resolve_paths: bool = True
    use_relative: bool = False
    write: bool = True

    def __post_init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        if self.subdataset is None or isinstance(self.subdataset, str):
            self._slc_to_subdataset = {slc: self.subdataset for slc in self.slc_list}
        else:
            # We're passing a sequence
            assert len(self.subdataset) == len(self.slc_list)
            self._slc_to_subdataset = dict(zip(self.slc_list, self.subdataset))

        if self.outdir is None:
            self.outdir = Path(self.slc_list[0]).parent

        # Set the dates to be used for each ifg
        if self.dates is None:
            # Use the first one we find in the name
            self.dates = [get_dates(f, fmt=self.date_format)[0] for f in self.slc_list]
        if len(self.dates) != len(self.slc_list):
            msg = f"{len(self.dates) = }, but {len(self.slc_list) = }"
            raise ValueError(msg)
        self._slc_to_date = dict(zip(self.slc_list, self.dates))

        # Run the appropriate network creation based on the options we passed
        self.slc_file_pairs = self._make_ifg_pairs()

        # Create each VRT file
        self.ifg_list: list[VRTInterferogram] = self._create_vrt_ifgs()

    def _make_ifg_pairs(self) -> list[IfgPairT]:
        """Form interferogram pairs from a list of SLC files sorted by date."""
        assert self.dates is not None
        ifgs: list[IfgPairT] = []
        # For each type of possible network, add the ifgs to the list
        if self.indexes is not None:
            # Give the option to select exactly which interferograms to create
            ifgs.extend(
                [
                    (self.slc_list[ref_idx], self.slc_list[sec_idx])
                    for ref_idx, sec_idx in self.indexes
                ]
            )
        if self.max_bandwidth is not None:
            ifgs.extend(Network.limit_by_bandwidth(self.slc_list, self.max_bandwidth))
        if self.max_temporal_baseline is not None:
            ifgs.extend(
                Network.limit_by_temporal_baseline(
                    self.slc_list,
                    dates=self.dates,
                    max_temporal_baseline=self.max_temporal_baseline,
                )
            )
        if self.reference_idx is not None:
            ifgs.extend(
                Network._single_reference_network(self.slc_list, self.reference_idx)
            )

        if self.include_annual:
            # Add in the annual pairs, then re-sort
            annual_ifgs = Network.find_annuals(
                self.slc_list, self.dates, buffer_days=self.annual_buffer_days
            )
            ifgs.extend(annual_ifgs)

        if not ifgs:
            msg = "No valid ifg list generation method specified"
            raise ValueError(msg)

        # Sort and dedupe them
        return sorted(set(ifgs))

    def _create_vrt_ifgs(self) -> list[VRTInterferogram]:
        """Write out a VRTInterferogram for each ifg."""
        ifg_list: list[VRTInterferogram] = []
        date_pairs = self._get_ifg_date_pairs()
        for idx, (ref, sec) in enumerate(self._gdal_file_strings):
            ref_date, sec_date = date_pairs[idx]

            v = VRTInterferogram(
                ref_slc=ref,
                sec_slc=sec,
                date_format=self.date_format,
                outdir=self.outdir,
                ref_date=ref_date,
                sec_date=sec_date,
                verify_slcs=self.verify_slcs,
                resolve_paths=self.resolve_paths,
                use_relative=self.use_relative,
                write=self.write,
            )
            ifg_list.append(v)
        return ifg_list

    @property
    def _gdal_file_strings(self):
        # format each file in each pair
        out = []
        for slc1, slc2 in self.slc_file_pairs:
            sd1 = self._slc_to_subdataset[slc1]
            sd2 = self._slc_to_subdataset[slc2]
            out.append(
                [io.format_nc_filename(slc1, sd1), io.format_nc_filename(slc2, sd2)]
            )
        return out

    def _get_ifg_date_pairs(self) -> list[tuple[DateOrDatetime, DateOrDatetime]]:
        date_pairs = []
        for slc1, slc2 in self.slc_file_pairs:
            d1 = self._slc_to_date[slc1]
            d2 = self._slc_to_date[slc2]
            date_pairs.append((d1, d2))
        return date_pairs

    def __repr__(self):
        return (
            f"Network(ifg_list={self.ifg_list}, slc_list={self.slc_list},"
            f" max_bandwidth={self.max_bandwidth},"
            f" max_temporal_baseline={self.max_temporal_baseline},"
            f" reference_idx={self.reference_idx})"
        )

    def __str__(self):
        return (
            f"Network of {len(self.ifg_list)} interferograms, "
            f"max_bandwidth={self.max_bandwidth}, "
            f"max_temporal_baseline={self.max_temporal_baseline}, "
            f"reference_idx={self.reference_idx}"
        )

    @staticmethod
    def _single_reference_network(
        slc_list: Sequence[Filename], reference_idx=0
    ) -> list[IfgPairT]:
        """Form a list of single-reference interferograms."""
        if len(slc_list) < 2:
            msg = "Need at least two dates to make an interferogram list"
            raise ValueError(msg)
        ref = slc_list[reference_idx]
        return [tuple(sorted([ref, date])) for date in slc_list if date != ref]

    @staticmethod
    def limit_by_bandwidth(
        slc_list: Iterable[T], max_bandwidth: int
    ) -> list[tuple[T, T]]:
        """Form a list of the "nearest-`max_bandwidth`" ifgs.

        Parameters
        ----------
        slc_list : Iterable[Filename]
            list of dates of SLCs
        max_bandwidth : int
            Largest allowed span of ifgs, by index distance, to include.
            max_bandwidth=1 will only include nearest-neighbor ifgs.

        Returns
        -------
        list
            Pairs of (date1, date2) ifgs

        """
        slc_to_idx = {s: idx for idx, s in enumerate(slc_list)}
        return [
            (a, b)
            for (a, b) in Network.all_pairs(slc_list)
            if slc_to_idx[b] - slc_to_idx[a] <= max_bandwidth
        ]

    @staticmethod
    def limit_by_temporal_baseline(
        slc_list: Iterable[T],
        dates: Sequence[DateOrDatetime],
        max_temporal_baseline: Optional[float] = None,
    ) -> list[tuple[T, T]]:
        """Form a list of the ifgs limited to a maximum temporal baseline.

        Parameters
        ----------
        slc_list : Iterable[Filename]
            Iterable of input SLC files
        dates: Sequence[DateOrDatetime]
            Dates or Datetimes corresponding to `slc_list`
        max_temporal_baseline : float, optional
            Largest allowed span of ifgs, by index distance, to include.
            max_bandwidth=1 will only include nearest-neighbor ifgs.

        Returns
        -------
        list
            Pairs of (date1, date2) ifgs

        Raises
        ------
        ValueError
            If any of the input files have more than one date.

        """
        ifg_strs = Network.all_pairs(slc_list)
        ifg_dates = Network.all_pairs(dates)
        baselines = [Network.temporal_baseline(ifg) for ifg in ifg_dates]
        return [
            ifg for ifg, b in zip(ifg_strs, baselines) if b <= max_temporal_baseline
        ]

    @staticmethod
    def find_annuals(
        slc_list: Iterable[T],
        dates: Sequence[DateOrDatetime],
        buffer_days: float = 30,
    ) -> list[tuple[T, T]]:
        """Pick out interferograms which are closest to 1 year in span.

        We only want to pick 1 ifg per date, closest to a year, but we will skip
        a date if it doesn't have an ifg of baseline 365 +/- `buffer_days`.
        """
        # keep track how far into ifg_list the last sar date was
        date_to_date_pair: dict[
            DateOrDatetime, tuple[DateOrDatetime, DateOrDatetime]
        ] = {}
        date_to_file: dict[DateOrDatetime, tuple[T, T]] = {}
        slc_pairs = Network.all_pairs(slc_list)
        date_pairs = Network.all_pairs(dates)
        for ifg, date_pair in zip(slc_pairs, date_pairs):
            early = date_pair[0]
            baseline_days = Network.temporal_baseline(date_pair)
            if abs(baseline_days - 365) > buffer_days:
                continue
            dp = date_to_date_pair.get(early)
            # Use this ifg as the annual if none exist, or if it's closer to 365
            if dp is None or abs(baseline_days - 365) < Network.temporal_baseline(dp):
                date_to_file[early] = ifg
        return sorted(date_to_file.values())

    @staticmethod
    def all_pairs(slc_list: Iterable[T]) -> list[tuple[T, T]]:
        """Create the list of all possible ifg pairs from slc_list."""
        return list(itertools.combinations(slc_list, r=2))

    @staticmethod
    def temporal_baseline(ifg_pair: tuple[DateOrDatetime, DateOrDatetime]):
        """Compute the temporal baseline (in decimal days) between the date pair."""
        return (ifg_pair[1] - ifg_pair[0]).total_seconds() / 86400

    def __len__(self):
        return len(self.ifg_list)

    def __getitem__(self, idx):
        return self.ifg_list[idx]

    def __iter__(self):
        return iter(self.ifg_list)

    def __contains__(self, item):
        return item in self.ifg_list

    def __eq__(self, other):
        return self.ifg_list == other.ifg_list


def estimate_correlation_from_phase(
    ifg: Union[VRTInterferogram, ArrayLike], window_size: int | tuple[int, int]
) -> np.ndarray:
    """Estimate correlation from only an interferogram (no SLCs/magnitudes).

    This is a simple correlation estimator that takes the (complex) average
    in a moving window in an interferogram. Used to get some estimate of interferometric
    correlation on the result of phase-linking interferograms.

    Parameters
    ----------
    ifg : Union[VRTInterferogram, ArrayLike]
        Interferogram to estimate correlation from.
        If a VRTInterferogram, will load and take the phase.
        If `ifg` is complex, will normalize to unit magnitude before estimating.
    window_size : int | tuple[int, int]
        Size of window to use for correlation estimation.
        If int, will use a symmetrical Gaussian sigma.

    Returns
    -------
    np.ndarray
        Correlation array

    """
    if isinstance(ifg, VRTInterferogram):
        ifg = ifg.load()
    nan_mask = np.isnan(ifg)
    zero_mask = ifg == 0
    if not np.iscomplexobj(ifg):
        # If they passed phase, convert to complex
        inp = np.exp(1j * np.nan_to_num(ifg))
    else:
        # If they passed complex, normalize to unit magnitude
        inp = np.exp(1j * np.nan_to_num(np.angle(ifg)))
    # Now set the 0s to nans (so that either 0 input, or nan input gets ignored)
    inp[inp == 0] = np.nan

    # The gaussian window becomes negligible at about "window_size/3":
    sigma: float | tuple[float, float]
    if isinstance(window_size, (float, int)):
        sigma = window_size / 3
    else:
        sigma = (window_size[0] / 3), (window_size[1] / 3)
    # Note: clipping is from possible partial windows producing correlation
    # above 1
    cor = np.clip(np.abs(gaussian_filter_nan(inp, sigma, mode="nearest")), 0, 1)
    # Return the input nans to nan
    cor[nan_mask] = np.nan
    # If the input was 0, the correlation is 0
    cor[zero_mask] = 0
    return cor


def estimate_interferometric_correlations(
    ifg_filenames: Sequence[Filename],
    window_size: tuple[int, int],
    out_driver: str = "GTiff",
    out_suffix: str = ".cor.tif",
    options: Sequence[str] = io.DEFAULT_TIFF_OPTIONS,
    keep_bits: int = 10,
    num_workers: int = 3,
) -> list[Path]:
    """Estimate correlations for a sequence of interferograms.

    Will use the same filename base as inputs with a new suffix.
    Calls `estimate_correlation_from_phase` on each of `ifg_filenames`.

    Parameters
    ----------
    ifg_filenames : Sequence[Filename]
        Paths to complex interferogram files.
    window_size : tuple[int, int] | int
        sigma, or (row sigma, column sigma) to use for Gaussian filtering.
    out_driver : str, optional
        Name of output GDAL driver, by default "GTiff".
    out_suffix : str, optional
        File suffix to use for correlation files, by default ".cor.tif"
    options : list[str], optional
        GDAL Creation options for the output array.
    keep_bits : int, optional
        Number of bits to preserve in mantissa. Defaults to None.
        Lower numbers will truncate the mantissa more and enable more compression.
    num_workers : int
        Number of threads to use for stitching in parallel.
        Default = 3

    Returns
    -------
    list[Path]
        Paths to newly written correlation files.

    """
    path_tuples: list[tuple[Path, Path]] = []
    output_paths: list[Path] = []
    for fn in ifg_filenames:
        ifg_path = Path(fn)
        cor_path = ifg_path.with_suffix(out_suffix)
        output_paths.append(cor_path)
        if cor_path.exists():
            logger.info(f"Skipping existing interferometric correlation for {ifg_path}")
            continue
        path_tuples.append((ifg_path, cor_path))

    def process_ifg(args):
        ifg_path, cor_path = args
        logger.debug(f"Estimating correlation for {ifg_path}, writing to {cor_path}")
        ifg = io.load_gdal(ifg_path)
        cor = estimate_correlation_from_phase(ifg, window_size=window_size)
        if keep_bits:
            io.round_mantissa(cor, keep_bits=keep_bits)
        io.write_arr(
            arr=cor,
            output_name=cor_path,
            like_filename=ifg_path,
            driver=out_driver,
            options=options,
        )

    thread_map(
        process_ifg,
        path_tuples,
        max_workers=num_workers,
        desc="Estimating correlations",
    )

    return output_paths


def _create_vrt_conj(
    filename: Filename, output_filename: Filename, is_relative: bool = False
):
    """Create a VRT raster of the conjugate of `filename`."""
    xsize, ysize = io.get_raster_xysize(filename)
    rel = "1" if is_relative else "0"

    # See https://gdal.org/drivers/raster/vrt.html#default-pixel-functions
    vrt_template = f"""\
<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
    <VRTRasterBand dataType="CFloat32" band="1" subClass="VRTDerivedRasterBand">
        <PixelFunctionType>conj</PixelFunctionType>
        <SimpleSource>
            <SourceFilename relativeToVRT="{rel}">{filename}</SourceFilename>
        </SimpleSource>
    </VRTRasterBand>
</VRTDataset>
    """
    with open(output_filename, "w") as f:
        f.write(vrt_template)
    io.copy_projection(filename, output_filename)


def convert_pl_to_ifg(
    phase_linked_slc: Filename,
    reference_date: DateOrDatetime,
    output_dir: Filename,
    dry_run: bool = False,
) -> Path:
    """Convert a phase-linked SLC to an interferogram by conjugating the phase.

    The SLC has already been multiplied by the (conjugate) phase of a reference SLC,
    so it only needs to be conjugated to put it in the form (ref * sec.conj()).

    Parameters
    ----------
    phase_linked_slc : Filename
        Path to phase-linked SLC file.
    reference_date : DateOrDatetime
        Reference date of the interferogram.
    output_dir : Filename
        Directory to place the renamed file.
    dry_run : bool
        Flag indicating that the new ifgs shouldn't be written to disk.
        Default = False (the ifgs will be created/written to disk.)
        `dry_run=True` is used to plan out which ifgs will be formed
        before actually running the workflow.

    Returns
    -------
    Path
        Path to renamed file.

    """
    # The phase_linked_slc will be named with the secondary date.
    # Make the output from that, plus the given reference date
    secondary_date = get_dates(phase_linked_slc)[-1]
    date_str = utils.format_date_pair(reference_date, secondary_date)
    out_name = Path(output_dir) / f"{date_str}.int.vrt"
    if dry_run:
        return out_name
    out_name.parent.mkdir(parents=True, exist_ok=True)
    # Now make a VRT to do the .conj
    _create_vrt_conj(phase_linked_slc, out_name)
    return out_name
