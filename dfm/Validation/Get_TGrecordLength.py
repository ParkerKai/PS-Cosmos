# ===============================================================================
# %% Import Modules
# ===============================================================================
import sys
import os
import scipy
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib
from glob import glob
import geopandas as gpd
import pandas as pd
from scipy.interpolate import interp1d

# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the DFM data resides
dir_in_TG = r"D:\Kai\DataDownloads\01_data\WaterLevels\TideGuage"

Gauges = [
    "9443090",  # Neah Bay
    "9444090",  # Port Anageles
    "9444900",  # Port Townsend
    "9447130",  # SEattle
    "9446484",  # Tacoma
    "9449880",  # Friday Harbor
    "9449424",  # Cherry POint
]

# sites=['cdip162']

Gauge_name = [
    "Neah Bay",
    "Port Angeles",
    "Port Townsend",
    "Seattle",
    "Tacoma",
    "Friday Harbor",
    "Cherry Point",
]


# Variable to process (calculate stats and make plots)
Var_process = "wl"  # 'wl', 'tide', 'ntr'

# ===============================================================================
# %% Define some functions
# ===============================================================================
sys.path.append(r"C:\Users\kai\Documents\Github\Kai_Python\General_Functions")
from Kai_MatlabTools import matlab2datetime

from Kai_ModelValidation import TaylorDiagram, mean_absolute_difference_corrected
from Kai_ModelValidation import ModelStat_Var, intersect_Var
from Kai_ModelValidation import bias, SpiderPlot


def gappy_interp(xint, x0, y0, *, maxgap=None, **kwargs):
    """
    Interpolate as scipy.interpolate.CubicSpline,
    but fill np.NaN is gaps of x0 that are greater than *maxgap*.

    xint : 1-D sequence of np.datetime64[ns]
        The x-coordinates at which to evaluate the interpolated values.
    x0 : 1-D sequence of np.datetime64[ns]
        The x-coordinates of the data points, must be increasing if argument
        period is not specified. Otherwise, xp is internally sorted after
        normalizing the periodic boundaries with x0 = x0 % period.
    y0 : 1-D sequence of float or complex
        The y-coordinates of the data points, same length as x0.
        If nans are present they will be removed
    maxgap : np.timedelta64   e.g. np.timedelta64(1, 'D')
        maximum gap size in xint to interpolate over.  Data between gaps is
        filled with NaN.

    **kwargs :
        Passed to `scipy.interpolate.CubicSpline`.

    """

    # See if there are nans
    if np.sum(np.isnan(y0)) > 0:
        print(f"{np.sum(np.isnan(y0))} Nans Found. Removing for Interpolation")
        x0 = x0[~np.isnan(y0)]
        y0 = y0[~np.isnan(y0)]

    f = scipy.interpolate.PchipInterpolator(x0, y0, **kwargs)
    yint = f(xint)
    # yint = np.interp(xint, x0, y0, **kwargs)   # original version with linear np interolate

    # figure out which x0 each xint belongs to:
    x_index = np.searchsorted(x0, xint, side="right")
    x_index = np.clip(x_index, 0, len(x0) - 1)

    # figure out the space between sample pairs
    dx = np.concatenate(([0], np.diff(x0)))
    # get the gap size for each xint data point:
    # get the indices of xint that are too large:
    index = dx[x_index] > maxgap

    # this is fine, except the degenerate case when a xint point falls
    # directly on a x0 value.  In that case we want to keep the data at
    # that point.  So we just choose the other inequality for the index:

    # as above, but use side='right':
    x_index = np.searchsorted(x0, xint, side="right")
    x_index = np.clip(x_index, 0, len(x0) - 1)
    dx = np.concatenate(([0], np.diff(x0)))
    index = np.logical_and(index, (dx[x_index] > maxgap))

    # set interpolated values where xint is inside a big gap to NaN:
    yint[index] = np.nan

    return yint


def Read_TG_Mat(dir_in_TG, gauge_ID):
    file_in = os.path.join(dir_in_TG, f"NOAA_TG_{gauge_ID}.mat")
    dat = scipy.io.loadmat(file_in)

    t_obs = matlab2datetime(dat["date_obs"].flatten(), "min")
    t_pre = matlab2datetime(dat["date_pred"].flatten(), "min")

    # interpolate tides onto the observed timeseries. Do this rather than intersection
    # To preserve the higher temporal resolution of the observations

    tide = gappy_interp(
        t_obs.to_numpy(),
        t_pre.to_numpy(),
        dat["wl_pre"].flatten(),
        maxgap=np.timedelta64(1, "D"),  # 1 day in nanoseconds
        extrapolate=False,
    )

    # f = scipy.interpolate.CubicSpline(toTimestamp(t_pre.to_numpy()),
    #                                   dat['wl_pre'].flatten(),
    #                                   extrapolate=False)

    # tide = f(toTimestamp(t_obs.to_numpy()))

    data_vars = {
        "wl": (
            ["time"],
            dat["wl_obs"].flatten(),
            {"units": "m", "long_name": "Water Level Observed"},
        ),
        "tide": (
            ["time"],
            tide,
            {"units": "m", "long_name": "Tide (Water Level Predicted)"},
        ),
        "ntr": (
            ["time"],
            dat["wl_obs"].flatten() - tide,
            {
                "units": "m",
                "long_name": "Non-Tidal Residual (Observed vs. predicted Water Levels)",
            },
        ),
        # 'quality':(['t_obs'],  dat['quality'].flatten(),
        #              {'units': 'm',
        #               'standard_name':'Sigma'}),
        # 'sigma':(['t_obs'],  dat['sigma'].flatten(),
        #              {'units': 'None',
        #               'standard_name':'Quality Code'}),
    }

    # define coordinates
    coords = {"time": (["time"], t_obs, {"standard_name": "time observed"})}
    # create dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "station": dat["station_info"][0][0][0][0][0][0],
            "Datum": dat["station_info"][0][0][0][0][0][2],
            "Units": dat["station_info"][0][0][0][0][0][3],
            "TimeZone": dat["station_info"][0][0][0][0][0][4],
        },
    )

    return ds


# ===============================================================================
# %% Read in the Tide Gauge data
# ===============================================================================
ds_gauge = []
record_length = np.zeros(len(Gauges))
for cnt, gauge_ID in enumerate(Gauges):
    ds = Read_TG_Mat(dir_in_TG, gauge_ID)

    # Conversion to NAVD88 (from VDATUM). Download of NAVD unavailable for gauge.
    # Port Townsend
    if gauge_ID == "9444900":
        ds["wl"] = ds["wl"] + 1.190
        ds["tide"] = ds["tide"] + 1.190
        ds.attrs["Datum"] = "NAVD88"

    # Friday Harbor
    if gauge_ID == "9449880":
        ds["wl"] = ds["wl"] + 1.271
        ds["tide"] = ds["tide"] + 1.271
        ds.attrs["Datum"] = "NAVD88"

    # Cherry POint
    if gauge_ID == "9449424":
        ds["wl"] = ds["wl"] + 1.317
        ds["tide"] = ds["tide"] + 1.317
        ds.attrs["Datum"] = "NAVD88"

    # Can't use resample as it doesn't preserve gaps
    # ds_sel = ds.resample(time="10min").interpolate("linear")

    # Resample to 10min
    rounded_dt = pd.to_datetime(ds["time"][0].values).round("D")
    t_start = np.datetime64(rounded_dt)

    rounded_dt = pd.to_datetime(ds["time"][-1].values).round("D")
    t_end = np.datetime64(rounded_dt)

    t_new = np.arange(t_start, t_end, np.timedelta64(10, "m"), dtype="datetime64[ns]")

    wl = gappy_interp(
        t_new,
        ds["time"].values,
        ds["wl"].values,
        maxgap=np.timedelta64(1, "D"),  # in np.timedelta64
        extrapolate=False,
    )

    tide = gappy_interp(
        t_new,
        ds["time"].values,
        ds["tide"].values,
        maxgap=np.timedelta64(1, "D"),  # in np.timedelta64
        extrapolate=False,
    )

    ntr = gappy_interp(
        t_new,
        ds["time"].values,
        ds["ntr"].values,
        maxgap=np.timedelta64(1, "D"),  # in np.timedelta64
        extrapolate=False,
    )

    # Add NTR as the subtraction of full model minus tide only
    data_vars = {
        "wl": (["time"], wl, {"units": "m", "long_name": "waterlevel"}),
        "tide": (["time"], tide, {"units": "m", "long_name": "tide"}),
        "ntr": (["time"], ntr, {"units": "m", "long_name": "non-tidal residual"}),
    }

    coords = {"time": (["time"], t_new, {"standard_name": "time"})}

    record_length[cnt] = np.sum(~np.isnan(wl)) * 10 / (60 * 24 * 365.25)  # in years, based on 10min resampled data

