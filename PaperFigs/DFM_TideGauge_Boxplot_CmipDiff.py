# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:56:45 2025

This script compares DFM outputs vs Tidegauge data

@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"

# ===============================================================================
# %% Import Modules
# ===============================================================================
import sys
import os
import scipy
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib
from glob import glob
import geopandas as gpd
import scipy.optimize as optimize
import pyextremes


THIS SCRIPT IS NOT FINISHED
ABAONDONED AS DECIDED TO GO A DIFFERENT DIRECTION

# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the DFM data resides
# dir_in = r'D:\DFM'
dir_in_model = r"Y:\PS_Cosmos\02_models\DFM_Regional\ERA5"
dir_in_TG = r"Y:\PS_Cosmos\01_data\WaterLevels\TideGauge"
dir_in_cmip6 = r"Y:\PS_Cosmos\GIS\DFM\DFM_CmipDiff_byModel"
dir_out = r"Y:\PS_Cosmos\Figures\DFM\validation"

# Gauges = [
#     "9443090",  # Neah Bay
#     "9444090",  # Port Anageles
#     "9444900",  # Port Townsend
#     "9447130",  # Seattle
#     "9446484",  # Tacoma
#     "9449880",  # Friday Harbor
#     "9449424", # Cherry POint
# ]


Mod_list = [
    "CNRM-CM6-1-HR",
    "EC-Earth_HR",
    "GFDL",
    "HadGEM_GC31_HH",
    "HadGEM_GC31_HM_highRes",
    "HadGEM_GC31_HM_highResSST",
    "CMCC-CM2-VHR4",
]


gauge_ID = "9447130"

slr = "000"


# Variable to process (calculate stats and make plots)
Var_process = "wl"  # 'wl', 'tide', 'ntr'

# ===============================================================================
# %% Define some functions
# ===============================================================================
sys.path.append(r"C:\Users\kaparker\GitHub\Python\Kai_Python\General_Functions")
from Kai_MatlabTools import matlab2datetime

from Kai_Timeseries import gappy_interp
from Kai_EVATools import calc_RIs_emp, calc_RIs_GPD


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
        maxgap=np.timedelta64(1, "D"),  # 1 day
        extrapolate=False,
    )

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




# Use a function minimizer to figure out the actual threshold we want
def threshold_min_fun(thresh, Num_Exce, filt, data):
    from pypot.threshold_selection import get_extremes_peaks_over_threshold

    pks = get_extremes_peaks_over_threshold(data, thresh, r=filt)

    # Number of Peaks for this threshold
    num_peaks = pks.shape[0]

    # What is the difference between this and the amount we want
    # This is the function we are trying to minimize to zero
    Diff = np.absolute(Num_Exce - num_peaks)

    return Diff


def POT_theshold_SetNum(data, y_lab, r, Num_Exce):
    """select threshold for PoT analysis
    using a set number of exceedences.

    args:
        data (pd.DataFrame): data
        y_lab: (str) label of y column
        r (str): time delta string to define independence
        Num_Exce (np.float): number of exceedences we want

    returns:
        threshold (np.float)
    """

    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        if isinstance(data, pd.Series):
            data = data.to_frame(name=y_lab)

    else:
        print(f"Unrecognized input data type {type(data)}")
        print("Input must be a pandas dataframe or series")

    # Guess for the threshold
    B = data.sort_values(by=y_lab, na_position="last", ascending=False)
    thresh_guess = B[y_lab].iloc[Num_Exce]

    # Minimize
    bnds = optimize.Bounds(lb=B[y_lab].mean(), ub=B[y_lab].iloc[1])
    Optim_out = optimize.minimize(
        threshold_min_fun,
        x0=thresh_guess,
        args=(Num_Exce, r, data[y_lab]),
        bounds=bnds,
        method="Nelder-Mead",
    )

    if Optim_out["success"]:
        threshold = Optim_out["x"]
    else:
        print("Something went wrong!")

    return threshold


def GetExtremes(data):
    # Data: Xarray dataset


    # Load TWL values into memeory as a pandas series
    twl = pd.Series(
        data["wl"].values , index=data["time"].values
        )

    # only process if more than 60% the record is real values (not nans)
    if twl.isna().sum() < twl.shape[0] * 0.5:
        try:
            # FInd extremes

            # Get number of exceedances to grab
            num_years = np.unique(twl.index.year).shape[0]
            Npryr = 1
            Num_Exce = (num_years - 1) * Npryr

            selected_thresh = POT_theshold_SetNum(
                twl,  # dataset
                "twl",  # column name of pd series
                "72h",  # independence time delta r
                Num_Exce,
            )  # number of exceedences we want

            extremes = pyextremes.extremes.get_extremes(
                twl,
                method="POT",
                extremes_type="high",
                threshold=selected_thresh,
                r="72h",
            )

            extremes = extremes.to_frame()

            # Get data associated with extremes
            extremes.insert(1, "ntr", data["ntr"].loc[extremes.index].values)
            extremes.insert(2, "tide", data["tide"].loc[extremes.index].values)

            extremes.rename(columns={"extreme values": "wl"}, inplace=True)


        except Exception as e:
            print(f"Dataset has Maximum finding issues: {e}")


    return extremes


# ===============================================================================
# %% Read in the Tide Gauge data
# ===============================================================================

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

ds_gauge = xr.Dataset(data_vars=data_vars, coords=coords)



# Calculate extremes




# ===============================================================================
# %% Read in the ERA5 Model Data
# ===============================================================================

# Load the data

files = glob(
    os.path.join(dir_in_model, "ERA5", "ERA5_000", "Results_Combined", "DFM_wl*")
)
ds_full = xr.open_mfdataset(
    files, engine="h5netcdf", parallel=True, chunks={"time": -1, "station": 1}
)

files = glob(
    os.path.join(
        dir_in_model, "ERA5_tidal_Results", "Results_Combined", "000", "DFM_wl*"
    )
)
ds_tidal = xr.open_mfdataset(
    files, engine="h5netcdf", parallel=True, chunks={"time": -1, "station": 1}
)

station_id = ds_full["station"].values


# FInd index for station_id
stat_id = next((cnt for cnt, s in enumerate(station_id) if gauge_ID in str(s)), None)

# Subset to station
ds_sel = ds_full.isel(station=stat_id)
# ds_sel = ds_sel.sel(time=slice("1995-01-01", "2025-01-01"))
ds_sel["waterlevel"] = ds_sel["waterlevel"] / 10000

# Add tide from tide only runs
ds_sel_tide = ds_tidal.isel(station=stat_id)
# ds_sel_tide = ds_sel_tide.sel(time=slice("1995-01-01", "2025-01-01"))
ds_sel_tide["waterlevel"] = ds_sel_tide["waterlevel"] / 10000
ds_sel = ds_sel.assign(tide=ds_sel_tide["waterlevel"])

# Add NTR as the subtraction of full model minus tide only
data_vars = {
    "ntr": (
        ["time"],
        ds_sel["waterlevel"].data - ds_sel["tide"].data,
        {"units": "m", "long_name": "Non-Tidal Residual"},
    )
}

coords = {"time": (["time"], ds_sel["time"].data, {"standard_name": "time"})}

ds = xr.Dataset(data_vars=data_vars, coords=coords)

# create dataset
ds_sel = ds_sel.assign(ntr=ds["ntr"])

# Change variable names
ds_model = ds_sel.rename({"waterlevel": "wl"})



# ===============================================================================
# %% Read in the CMIP6 Model Data
# ===============================================================================





# ===============================================================================
# %% Get Extremes
# ===============================================================================

model_extremes = GetExtremes(ds_model)
gauge_extremes = GetExtremes(ds_gauge)




# ===============================================================================
# %% Plots
# ===============================================================================


fig, ax = matplotlib.pyplot.subplots(1, 1, figsize=[8, 6])



