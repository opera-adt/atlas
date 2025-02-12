from __future__ import annotations

import datetime
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import xarray
from pyproj import CRS, Transformer
from shapely.geometry import box

import dolphin.atmosphere._utils as utils
import dolphin.atmosphere.model_levels as ml

logger = logging.getLogger("dolphin")

_ZMIN = np.float64(-100)  # minimum required height
_ZREF = np.float64(26000)  # maximum integration height
# when not specified by user
_g0 = g0 = np.float64(9.80665)  # Standard gravitational constant
G1 = np.float64(9.80616)  # Gravitational constant @ 45° latitude used for corrections
# of earth's centrifugal force

TIME_RES = {
    "ECMWF": 1,
    "HRES": 6,
}


class WeatherModel(ABC):
    """Implement a generic weather model for getting estimated SAR delays."""

    def __init__(self):
        """Initiate the weather model."""
        # Initialize model-specific constants/parameters
        self._k1 = None
        self._k2 = None
        self._k3 = None
        self._humidityType = "q"
        self._a = []
        self._b = []

        self.files = None

        self._time_res = None  # time resolution of the weather model in hours

        self._lon_res = None
        self._lat_res = None
        self._x_res = None
        self._y_res = None

        self._classname = None
        self._dataset = None
        self._Name = None
        self._wmLoc = None

        self._model_level_type = "ml"

        self._valid_range = (
            datetime.datetime(1900, 1, 1).replace(
                tzinfo=datetime.timezone(offset=datetime.timedelta())
            ),
            datetime.datetime.now(datetime.timezone.utc).date(),
        )  # Tuple of min/max years where data is available.
        self._lag_time = datetime.timedelta(days=30)  # Availability lag time in days
        self._time = None
        self._bbox = None

        # Define fixed constants
        self._R_v = 461.524
        self._R_d = 287.06  # in our original code this was 287.053
        self._g0 = _g0  # gravity constant
        self._zmin = _ZMIN  # minimum integration height
        self._zmax = _ZREF  # max integration height
        self._proj = None

        # setup data structures
        self._levels = None
        self._xs = np.empty((1, 1, 1))  # Use generic x/y/z instead of lon/lat/height
        self._ys = np.empty((1, 1, 1))
        self._zs = np.empty((1, 1, 1))

        self._lats = None
        self._lons = None
        self._ll_bounds = None
        self._valid_bounds = box(-180, -90, 180, 90)  # Shapely box with WSEN bounds

        self._p = None
        self._q = None
        self._rh = None
        self._t = None
        self._e = None
        self._wet_refractivity = None
        self._hydrostatic_refractivity = None
        self._wet_ztd = None
        self._hydrostatic_ztd = None

    def __str__(self):
        string = "\n"
        string += "======Weather Model class object=====\n"
        string += "Weather model time: {}\n".format(self._time)
        string += "Latitude resolution: {}\n".format(self._lat_res)
        string += "Longitude resolution: {}\n".format(self._lon_res)
        string += "Native projection: {}\n".format(self._proj)
        string += "ZMIN: {}\n".format(self._zmin)
        string += "ZMAX: {}\n".format(self._zmax)
        string += "k1 = {}\n".format(self._k1)
        string += "k2 = {}\n".format(self._k2)
        string += "k3 = {}\n".format(self._k3)
        string += "Humidity type = {}\n".format(self._humidityType)
        string += "=====================================\n"
        string += "Class name: {}\n".format(self._classname)
        string += "Dataset: {}\n".format(self._dataset)
        string += "=====================================\n"
        string += "A: {}\n".format(self._a)
        string += "B: {}\n".format(self._b)
        if self._p is not None:
            string += "Number of points in Lon/Lat = {}/{}\n".format(*self._p.shape[:2])
            string += "Total number of grid points (3D): {}\n".format(
                np.prod(self._p.shape)
            )
        if self._xs.size == 0:
            string += "Minimum/Maximum y: {: 4.2f}/{: 4.2f}\n".format(
                np.nanmin(self._ys), np.nanmax(self._ys)
            )
            string += "Minimum/Maximum x: {: 4.2f}/{: 4.2f}\n".format(
                np.nanmin(self._xs), np.nanmax(self._xs)
            )
            string += "Minimum/Maximum zs/heights: {: 10.2f}/{: 10.2f}\n".format(
                np.nanmin(self._zs), np.nanmax(self._zs)
            )
        string += "=====================================\n"
        return str(string)

    def _get_time(self):
        return self._time

    def set_time(self, time, fmt="%Y-%m-%dT%H:%M:%S"):
        """Set the time for a weather model."""
        if isinstance(time, str):
            self._time = datetime.datetime.strptime(time, fmt)
        elif isinstance(time, datetime.datetime):
            self._time = time
        else:
            raise TypeError('"time" must be a string or a datetime object')
        if self._time.tzinfo is None:
            self._time = self._time.replace(
                tzinfo=datetime.timezone(offset=datetime.timedelta())
            )

    def get_latlon_bounds(self):
        """Get the lat lon bounding box."""
        return self._ll_bounds

    def set_latlon_bounds(self, ll_bounds, Nextra=2, output_spacing=None):
        """Correct lat/lon bounds.

        because not all of the weather models have valid data exactly
        bounded by -90/90 (lats) and -180/180 (lons); for GMAO and MERRA2,
        need to adjust the longitude higher end with an extra buffer;
        for other models, the exact bounds are close to  -90/90 (lats)
        and -180/180 (lons) and thus can be rounded to the above regions
        (either in the downloading-file API or subsetting-data API)
        without problems.
        """
        ex_buffer_lon_max = 0.0

        if self._Name in "HRRR HRRR-AK HRES".split():
            Nextra = 6  # have a bigger buffer

        else:
            ex_buffer_lon_max = self._lon_res

        # At boundary lats and lons, need to modify Nextra buffer
        # so that the lats and lons do not exceed the boundary
        S, N, W, E = ll_bounds

        # Adjust bounds if they get near the poles or IDL
        pixlat, pixlon = Nextra * self._lat_res, Nextra * self._lon_res

        S = np.max([S - pixlat, -90.0 + pixlat])
        N = np.min([N + pixlat, 90.0 - pixlat])
        W = np.max(
            [W - (pixlon + ex_buffer_lon_max), -180.0 + (pixlon + ex_buffer_lon_max)]
        )
        E = np.min(
            [E + (pixlon + ex_buffer_lon_max), 180.0 - pixlon - ex_buffer_lon_max]
        )
        if output_spacing is not None:
            S, N, W, E = [
                np.floor(S / output_spacing) * output_spacing,
                np.ceil(N / output_spacing) * output_spacing,
                np.floor(W / output_spacing) * output_spacing,
                np.ceil(E / output_spacing) * output_spacing,
            ]

        self._ll_bounds = np.array([S, N, W, E])

    def get_wm_loc(self):
        """Get the path to the direct with the weather model files."""
        wmLoc = Path.cwd() / "weather_files" if self._wmLoc is None else self._wmLoc
        return wmLoc

    def set_wm_loc(self, weather_model_directory: str):
        """Set the path to the directory with the weather model files."""
        self._wmLoc = weather_model_directory

    def load(self, _zlevels=None):
        """Call the load_weather method.

        Each model class should define a load_weather.
        method appropriate for that class.
        """
        # If the weather file has already been processed, do nothing
        outLoc = self.get_wm_loc()
        path_wm_raw = Path(outLoc) / "{}_{}.{}".format(
            self._Name,
            datetime.datetime.strftime(self._get_time(), "%Y_%m_%d_T%H_%M_%S"),
            "nc",
        )

        self._out_name = self.out_file(outLoc)

        if Path(self._out_name).exists():
            return self._out_name
        else:
            # Load the weather just for the query points
            self.load_weather(f=path_wm_raw)

            # Process the weather model data
            self._find_e()
            self._uniform_in_z(_zlevels=_zlevels)
            self._check_for_nans()

            # Compute Zenith delays at the weather model grid nodes
            self._get_ztd()
            return None

    @abstractmethod
    def load_weather(self, f=None):
        """Load weather model.

        Should be implemented in each weather model type class.
        """

    def set_level_type(self, levelType):
        """Set the level type to model levels or pressure levels."""
        if levelType in "ml pl nat prs".split():
            self._model_level_type = levelType
        else:
            raise RuntimeError(f"Level type {levelType} is not recognized")

        if levelType in "ml nat".split():
            self.__model_levels__()
        else:
            self.__pressure_levels__()

    # def _convertmb2Pa(self, pres):
    #     """Convert pressure in millibars to Pascals."""
    #     return 100 * pres

    def _get_heights(self, lats, geo_hgt, geo_ht_fill=np.nan):
        """Transform geo heights to WGS84 ellipsoidal heights."""
        geo_ht_fix = np.where(geo_hgt != geo_ht_fill, geo_hgt, np.nan)
        lats_full = np.broadcast_to(lats[..., np.newaxis], geo_ht_fix.shape)
        self._zs = utils.geo_to_ht(lats_full, geo_ht_fix)

    def _find_e(self):
        """Check the type of e-calculation needed."""
        if self._humidityType == "rh":
            self._find_e_from_rh()
        elif self._humidityType == "q":
            self._find_e_from_q()
        else:
            raise RuntimeError("Not a valid humidity type")
        self._rh = None
        self._q = None

    def _find_e_from_q(self):
        """Calculate e, partial pressure of water vapor."""
        svp = utils.find_svp(self._t)
        # We have q = w/(w + 1), so w = q/(1 - q)
        w = self._q / (1 - self._q)
        self._e = w * self._R_v * (self._p - svp) / self._R_d

    def _find_e_from_rh(self):
        """Calculate partial pressure of water vapor."""
        svp = utils.find_svp(self._t)
        self._e = self._rh / 100 * svp

    def get_wet_refractivity(self):
        """Calculate the wet delay from pressure, temperature, and e."""
        self._wet_refractivity = (
            self._k2 * self._e / self._t + self._k3 * self._e / self._t**2
        )

    def get_hydro_refractivity(self):
        """Calculate the hydrostatic delay from pressure and temperature."""
        self._hydrostatic_refractivity = self._k1 * self._p / self._t

    def _get_wet_refractivity(self):
        return self._wet_refractivity

    def _get_hydro_refractivity(self):
        return self._hydrostatic_refractivity

    def _adjust_grid(self, ll_bounds=None):
        """Pad the weather grid with a level at self._zmin.

        if it does not already go that low.
        <<The functionality below has been removed.>>
        <<It also removes levels that are above self._zmax,
        since they are not needed.>>.
        """
        if self._zmin < np.nanmin(self._zs):
            # first add in a new layer at zmin
            self._zs = np.insert(self._zs, 0, self._zmin)

            self._p = utils.pad_lower(self._p)
            self._t = utils.pad_lower(self._t)
            self._e = utils.pad_lower(self._e)
            self._wet_refractivity = utils.pad_lower(self._wet_refractivity)
            self._hydrostatic_refractivity = utils.pad_lower(
                self._hydrostatic_refractivity
            )
            if ll_bounds is not None:
                self._trim_extent(ll_bounds)

    def _get_ztd(self):
        """Compute the full slant tropospheric delay for each weather model grid node.

        using the reference height zref.
        """
        self.get_wet_refractivity()
        self.get_hydro_refractivity()
        self._adjust_grid(self.get_latlon_bounds())

        wet = self._get_wet_refractivity()
        hydro = self._get_hydro_refractivity()

        # Get the integrated ZTD
        wet_total, hydro_total = np.zeros(wet.shape), np.zeros(hydro.shape)
        for level in range(wet.shape[2]):
            wet_total[..., level] = 1e-6 * np.trapezoid(
                wet[..., level:], x=self._zs[level:], axis=2
            )
            hydro_total[..., level] = 1e-6 * np.trapezoid(
                hydro[..., level:], x=self._zs[level:], axis=2
            )
        self._hydrostatic_ztd = hydro_total
        self._wet_ztd = wet_total

    @property
    def bbox(self) -> list:
        """Obtains the bounding box of the weather model in lat/lon CRS.

        Returns
        -------
        list
            xmin, ymin, xmax, ymax

        Raises
        ------
        ValueError
           When `self.files` is None.

        """
        if self._bbox is None:
            path_weather_model = self.out_file(self.get_wm_loc())
            if not Path(path_weather_model).exists():
                raise ValueError("Need to save cropped weather model as netcdf")

            with xarray.load_dataset(path_weather_model) as ds:
                if "x" in ds and "y" in ds:
                    xmin, xmax = ds.x.min(), ds.x.max()
                    ymin, ymax = ds.y.min(), ds.y.max()
                else:
                    xmin, xmax = ds.longitude.min(), ds.longitude.max()
                    ymin, ymax = ds.latitude.min(), ds.latitude.max()

            wm_proj = self._proj
            xs, ys = [xmin, xmin, xmax, xmax], [ymin, ymax, ymin, ymax]
            t = Transformer.from_crs(wm_proj.to_epsg(), 4326, always_xy=True)
            lons, lats = t.transform(xs, ys)

            ## projected weather models may not be aligned N/S
            ## should only matter for warning messages
            W, E = np.min(lons), np.max(lons)
            # S, N = np.sort([lats[np.argmin(lons)], lats[np.argmax(lons)]])
            S, N = np.min(lats), np.max(lats)
            self._bbox = W, S, E, N

        return self._bbox

    def _trim_extent(self, extent):
        """Get the bounding box around a set of lats/lons."""
        lat = self._lats.copy()
        lon = self._lons.copy()
        lat[np.isnan(lat)] = np.nanmean(lat)
        lon[np.isnan(lon)] = np.nanmean(lon)
        mask = (
            (lat >= extent[0])
            & (lat <= extent[1])
            & (lon >= extent[2])
            & (lon <= extent[3])
        )
        ma1 = np.sum(mask, axis=1).astype("bool")
        ma2 = np.sum(mask, axis=0).astype("bool")
        if np.sum(ma1) == 0 and np.sum(ma2) == 0:
            # Don't need to remove any points
            return

        # indices of the part of the grid to keep
        ny, nx, nz = self._p.shape
        index1 = max(np.arange(len(ma1))[ma1][0] - 2, 0)
        index2 = min(np.arange(len(ma1))[ma1][-1] + 2, ny)
        index3 = max(np.arange(len(ma2))[ma2][0] - 2, 0)
        index4 = min(np.arange(len(ma2))[ma2][-1] + 2, nx)

        # subset around points of interest
        self._lons = self._lons[index1:index2, index3:index4]
        self._lats = self._lats[index1:index2, index3:index4]
        self._xs = self._xs[index3:index4]
        self._ys = self._ys[index1:index2]
        self._p = self._p[index1:index2, index3:index4, ...]
        self._t = self._t[index1:index2, index3:index4, ...]
        self._e = self._e[index1:index2, index3:index4, ...]

        self._wet_refractivity = self._wet_refractivity[
            index1:index2, index3:index4, ...
        ]
        self._hydrostatic_refractivity = self._hydrostatic_refractivity[
            index1:index2, index3:index4, :
        ]

    def _calculategeoh(self, z, lnsp):
        """Calculate pressure, geopotential, and geopotential height.

        from the surface pressure and model levels provided by a weather model.
        The model levels are numbered from the highest eleveation to the lowest.
        Inputs:
            self - weather model object with parameters a, b defined
            z    - 3-D array of surface heights for the location(s) of interest
            lnsp - log of the surface pressure
        Outputs:
            geopotential - The geopotential in units of height times acceleration
            pressurelvs  - The pressure at each of the model levels for each of
                           the input points
            geoheight    - The geopotential heights.
        """
        return utils.calcgeoh(
            lnsp, self._t, self._q, z, self._a, self._b, self._R_d, self._levels
        )

    def _uniform_in_z(self, _zlevels=None):
        """Interpolate all variables to a regular grid in z."""
        nx, ny = self._p.shape[:2]

        # new regular z-spacing
        if _zlevels is None:
            try:
                _zlevels = self._zlevels
            except BaseException:
                _zlevels = np.nanmean(self._zs, axis=(0, 1))

        new_zs = np.tile(_zlevels, (nx, ny, 1))

        # re-assign values to the uniform z
        self._t = utils.interpolate_along_axis(
            self._zs, self._t, new_zs, axis=2
        ).astype(np.float32)
        self._p = utils.interpolate_along_axis(
            self._zs, self._p, new_zs, axis=2
        ).astype(np.float32)
        self._e = utils.interpolate_along_axis(
            self._zs, self._e, new_zs, axis=2
        ).astype(np.float32)

        self._zs = _zlevels
        self._xs = np.unique(self._xs)
        self._ys = np.unique(self._ys)

    def _check_for_nans(self):
        """Fill in NaN-values."""
        self._p = utils.fillna3d(self._p)
        self._t = utils.fillna3d(
            self._t, fill_value=1e16
        )  # to avoid division by zero later on
        self._e = utils.fillna3d(self._e)

    def out_file(self, outLoc):
        """Path to the output file."""
        s = np.floor(self._ll_bounds[0])
        S = f"{np.abs(s):.0f}S" if s < 0 else f"{s:.0f}N"

        n = np.ceil(self._ll_bounds[1])
        N = f"{np.abs(n):.0f}S" if n < 0 else f"{n:.0f}N"

        w = np.floor(self._ll_bounds[2])
        W = f"{np.abs(w):.0f}W" if w < 0 else f"{w:.0f}E"

        e = np.ceil(self._ll_bounds[3])
        E = f"{np.abs(e):.0f}W" if e < 0 else f"{e:.0f}E"
        out_name = f'{self._time.strftime("%Y_%m_%d_T%H_%M_%S")}_{S}_{N}_{W}_{E}.nc'
        f = f"{self._Name}_" + out_name
        return Path(outLoc) / f

    def filename(self, time=None, outLoc="weather_files"):
        """Create a filename to store the weather model."""
        Path(outLoc).mkdir(parents=True, exist_ok=True)

        if time is None:
            if self._time is None:
                raise ValueError(
                    "Time must be specified before the file can be written"
                )
            else:
                time = self._time

        f = Path(outLoc) / "{}_{}.{}".format(
            self._Name, datetime.datetime.strftime(time, "%Y_%m_%d_T%H_%M_%S"), "nc"
        )

        self.files = [f]
        return f

    def write(self):
        """By calling the abstract/modular netcdf writer.

        write the weather model data and refractivity to an
        NETCDF4 file that can be accessed by external programs.
        """
        # Generate the filename
        f = self._out_name

        attrs_dict = {
            "Conventions": "CF-1.6",
            "datetime": datetime.datetime.strftime(self._time, "%Y_%m_%dT%H_%M_%S"),
            "date_created": datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S"),
            "title": "Weather model data and delay calculations",
            "model_name": self._Name,
        }

        dimension_dict = {
            "x": ("x", self._xs),
            "y": ("y", self._ys),
            "z": ("z", self._zs),
            "latitude": (("y", "x"), self._lats),
            "longitude": (("y", "x"), self._lons),
            "datetime_utc": self._time.replace(tzinfo=None),
        }

        dataset_dict = {
            "t": (("z", "y", "x"), self._t.swapaxes(0, 2).swapaxes(1, 2)),
            "p": (("z", "y", "x"), self._p.swapaxes(0, 2).swapaxes(1, 2)),
            "e": (("z", "y", "x"), self._e.swapaxes(0, 2).swapaxes(1, 2)),
            "wet": (
                ("z", "y", "x"),
                self._wet_refractivity.swapaxes(0, 2).swapaxes(1, 2),
            ),
            "hydro": (
                ("z", "y", "x"),
                self._hydrostatic_refractivity.swapaxes(0, 2).swapaxes(1, 2),
            ),
            "wet_total": (("z", "y", "x"), self._wet_ztd.swapaxes(0, 2).swapaxes(1, 2)),
            "hydro_total": (
                ("z", "y", "x"),
                self._hydrostatic_ztd.swapaxes(0, 2).swapaxes(1, 2),
            ),
        }

        ds = xarray.Dataset(
            data_vars=dataset_dict, coords=dimension_dict, attrs=attrs_dict
        )

        # Define units
        ds["t"].attrs["units"] = "K"
        ds["e"].attrs["units"] = "Pa"
        ds["p"].attrs["units"] = "Pa"
        ds["wet"].attrs["units"] = "dimentionless"
        ds["hydro"].attrs["units"] = "dimentionless"
        ds["wet_total"].attrs["units"] = "m"
        ds["hydro_total"].attrs["units"] = "m"

        # Define standard names
        ds["t"].attrs["standard_name"] = "temperature"
        ds["e"].attrs["standard_name"] = "humidity"
        ds["p"].attrs["standard_name"] = "pressure"
        ds["wet"].attrs["standard_name"] = "wet_refractivity"
        ds["hydro"].attrs["standard_name"] = "hydrostatic_refractivity"
        ds["wet_total"].attrs["standard_name"] = "total_wet_refractivity"
        ds["hydro_total"].attrs["standard_name"] = "total_hydrostatic_refractivity"

        # projection information
        ds["proj"] = int()
        for k, v in self._proj.to_cf().items():
            ds.proj.attrs[k] = v
        for var in ds.data_vars:
            ds[var].attrs["grid_mapping"] = "proj"

        # write to file and return the filename
        ds.to_netcdf(f)
        return f


class ECMWF(WeatherModel):
    """Implement ECMWF models."""

    def __init__(self, level_type="ml"):
        """Initiate ECMWF model."""
        # initialize a weather model
        WeatherModel.__init__(self)

        # model constants
        self._k1 = 0.776  # [K/Pa]
        self._k2 = 0.233  # [K/Pa]
        self._k3 = 3.75e3  # [K^2/Pa]

        self._time_res = TIME_RES["ECMWF"]

        self._lon_res = 0.25
        self._lat_res = 0.25
        self._proj = CRS.from_epsg(4326)

        # self._model_level_type = 'ml'  # Default
        self._Name = "ECMWF"
        self._dataset = "ecmwf"
        # self.__model_levels__()

        self._humidityType = "q"
        # Default, pressure levels are 'pl'

        self.set_level_type(level_type)

    def __pressure_levels__(self):
        self._zlevels = np.flipud(ml.LEVELS_25_HEIGHTS)
        self._levels = len(self._zlevels)

    def __model_levels__(self):
        self._levels = 137
        self._zlevels = np.flipud(ml.LEVELS_137_HEIGHTS)
        self._a = ml.A_137_HRES
        self._b = ml.B_137_HRES

    def load_weather(self, f=None):
        """Consistent class method to be implemented across all weather model types.

        As a result of calling this method, all of the variables (x, y, z, p, q,
        t, wet_refractivity, hydrostatic refractivity, e) should be fully
        populated.
        """
        f = self.files[0] if f is None else f
        self._load_model_level(f)

    def _load_model_level(self, fname):
        # read data from netcdf file

        lats, lons, xs, ys, t, q, lnsp, z = self._make_data_cubes(fname)

        # ECMWF appears to give me this backwards
        if lats[0] > lats[1]:
            z = z[::-1]
            lnsp = lnsp[::-1]
            t = t[:, ::-1]
            q = q[:, ::-1]
            lats = lats[::-1]
        # Lons is usually ok, but we'll throw in a check to be safe
        if lons[0] > lons[1]:
            z = z[..., ::-1]
            lnsp = lnsp[..., ::-1]
            t = t[..., ::-1]
            q = q[..., ::-1]
            lons = lons[::-1]
        # pyproj gets fussy if the latitude is wrong, plus our
        # interpolator isn't clever enough to pick up on the fact that
        # they are the same
        lons[lons > 180] -= 360

        self._t = t
        self._q = q
        geo_hgt, pres, hgt = self._calculategeoh(z, lnsp)

        self._lons, self._lats = np.meshgrid(lons, lats)

        # ys is latitude
        self._get_heights(self._lats, hgt.transpose(1, 2, 0))
        h = self._zs.copy()

        # We want to support both pressure levels and true pressure grids.
        # If the shape has one dimension, we'll scale it up to act as a
        # grid, otherwise we'll leave it alone.
        if len(pres.shape) == 1:
            self._p = np.broadcast_to(pres[:, np.newaxis, np.newaxis], self._zs.shape)
        else:
            self._p = pres

        # Re-structure everything from (heights, lats, lons) to (lons, lats, heights)
        self._p = self._p.transpose(1, 2, 0)
        self._t = self._t.transpose(1, 2, 0)
        self._q = self._q.transpose(1, 2, 0)

        # Flip all the axis so that zs are in order from bottom to top
        # lats / lons are simply replicated to all heights so they don't need flipped
        self._p = np.flip(self._p, axis=2)
        self._t = np.flip(self._t, axis=2)
        self._q = np.flip(self._q, axis=2)
        self._ys = self._lats.copy()
        self._xs = self._lons.copy()
        self._zs = np.flip(h, axis=2)

    def _load_pressure_level(self, filename, *args, **kwargs):
        with xarray.open_dataset(filename) as block:
            # Pull the data
            z = np.squeeze(block["z"].values)
            t = np.squeeze(block["t"].values)
            q = np.squeeze(block["q"].values)
            lats = np.squeeze(block.latitude.values)
            lons = np.squeeze(block.longitude.values)
            levels = np.squeeze(block.level.values) * 100

        z = np.flip(z, axis=1)

        # ECMWF appears to give me this backwards
        if lats[0] > lats[1]:
            z = z[::-1]
            t = t[:, ::-1]
            q = q[:, ::-1]
            lats = lats[::-1]
        # Lons is usually ok, but we'll throw in a check to be safe
        if lons[0] > lons[1]:
            z = z[..., ::-1]
            t = t[..., ::-1]
            q = q[..., ::-1]
            lons = lons[::-1]
        # pyproj gets fussy if the latitude is wrong, plus our
        # interpolator isn't clever enough to pick up on the fact that
        # they are the same
        lons[lons > 180] -= 360

        self._t = t
        self._q = q

        geo_hgt = (z / self._g0).transpose(1, 2, 0)

        # re-assign lons, lats to match heights
        self._lons, self._lats = np.meshgrid(lons, lats)

        # correct heights for latitude
        self._get_heights(self._lats, geo_hgt)

        self._p = np.broadcast_to(levels[np.newaxis, np.newaxis, :], self._zs.shape)

        # Re-structure from (heights, lats, lons) to (lons, lats, heights)
        self._t = self._t.transpose(1, 2, 0)
        self._q = self._q.transpose(1, 2, 0)
        self._ys = self._lats.copy()
        self._xs = self._lons.copy()

        # flip z to go from surface to toa
        self._p = np.flip(self._p, axis=2)
        self._t = np.flip(self._t, axis=2)
        self._q = np.flip(self._q, axis=2)

    def _make_data_cubes(self, fname):
        """Create a cube of data representing temperature and relative humidity.

        at specified pressure levels.
        """
        # get ll_bounds
        S, N, W, E = self._ll_bounds

        ds = xarray.open_dataset(fname)
        # ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
        ds = ds.assign_coords(longitude=(((ds.longitude) % 360) - 180))

        # mask based on query bounds
        m1 = (ds.latitude >= S) & (ds.latitude <= N)
        m2 = (ds.longitude >= W) & (ds.longitude <= E)
        block = ds.where(m1 & m2, drop=True)

        # Pull the data
        z = np.squeeze(block["z"].values)[0, ...]
        t = np.squeeze(block["t"].values)
        q = np.squeeze(block["q"].values)
        lnsp = np.squeeze(block["lnsp"].values)[0, ...]
        lats = np.squeeze(block.latitude.values)
        lons = np.squeeze(block.longitude.values)

        xs = lons.copy()
        ys = lats.copy()
        ds.close()

        if z.size == 0:
            raise RuntimeError(
                "There is no data in z, " "you may have a problem with your mask"
            )

        return lats, lons, xs, ys, t, q, lnsp, z


class HRES(ECMWF):
    """Implement HRES models."""

    def __init__(self, level_type="ml"):
        """Initiate HRES model."""
        # initialize a weather model
        WeatherModel.__init__(self)

        # model constants
        self._k1 = 0.776  # [K/Pa]
        self._k2 = 0.233  # [K/Pa]
        self._k3 = 3.75e3  # [K^2/Pa]

        # 9 km horizontal grid spacing. This is only used for extending
        # the download-buffer, i.e. not in subsequent processing.
        self._lon_res = 9.0 / 111  # 0.08108115
        self._lat_res = 9.0 / 111  # 0.08108115
        self._x_res = 9.0 / 111  # 0.08108115
        self._y_res = 9.0 / 111  # 0.08108115

        self._humidityType = "q"
        # Default, pressure levels are 'pl'
        self._expver = "1"
        self._classname = "od"
        self._dataset = "hres"
        self._Name = "HRES"
        self._proj = CRS.from_epsg(4326)

        self._time_res = TIME_RES[self._dataset.upper()]
        # Tuple of min/max years where data is available.
        self._valid_range = (
            datetime.datetime(1983, 4, 20).replace(
                tzinfo=datetime.timezone(offset=datetime.timedelta())
            ),
            datetime.datetime.now(datetime.timezone.utc),
        )
        # Availability lag time in days
        self._lag_time = datetime.timedelta(hours=6)

        self.set_level_type(level_type)

    def _update_a_b(self):
        # Before 2013-06-26, there were only 91 model levels.
        # The mapping coefficients below are extracted
        # based on https://www.ecmwf.int/en/forecasts/documentation-and-support/91-model-levels
        self._levels = 91
        self._zlevels = np.flipud(ml.LEVELS_91_HEIGHTS)
        self._a = ml.A_91_HRES
        self._b = ml.B_91_HRES

    def load_weather(self, f=None):
        """Consistent class method to be implemented across all weather model types.

        As a result of calling this method, all of the variables (x, y, z, p, q,
        t, wet_refractivity, hydrostatic refractivity, e) should be fully
        populated.
        """
        f = self.files[0] if f is None else f

        if self._model_level_type == "ml":
            if self._time < datetime.datetime(2013, 6, 26, 0, 0, 0):
                self._update_a_b()
            self._load_model_level(f)
        elif self._model_level_type == "pl":
            self._load_pressure_levels(f)
