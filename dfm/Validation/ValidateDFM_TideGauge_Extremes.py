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
# dir_in = r'D:\DFM'
dir_in_model = r"Y:\PS_Cosmos\02_models\DFM_Regional\ERA5"
dir_in_TG = r"Y:\PS_Cosmos\01_data\WaterLevels\TideGauge"
dir_out = r"Y:\PS_Cosmos\Figures\DFM\validation"

Gauges = [
    "9443090",  # Neah Bay
    "9444090",  # Port Anageles
    "9444900",  # Port Townsend
    "9447130",  # Seattle
    "9446484",  # Tacoma
    "9449880",  # Friday Harbor
    "9449424",
]  # Cherry POint

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
sys.path.append(
    r"C:\Users\kaparker\Documents\GitHub\Python\Kai_Python\General_Functions"
)
from Kai_MatlabTools import matlab2datetime

from Kai_ModelValidation import TaylorDiagram, mean_absolute_difference_corrected
from Kai_ModelValidation import ModelStat_Var, intersect_Var
from Kai_ModelValidation import bias, SpiderPlot
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


# ===============================================================================
# %% Read in the Tide Gauge data
# ===============================================================================
ds_gauge = []
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

    ds_sel = xr.Dataset(data_vars=data_vars, coords=coords)

    ds_gauge.append(ds_sel)


# ===============================================================================
# %% Read in the Model Data
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

ds_model = []
for cnt, gauge_ID in enumerate(Gauges):
    # FInd index for station_id
    stat_id = next(
        (cnt for cnt, s in enumerate(station_id) if gauge_ID in str(s)), None
    )

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
    ds_sel = ds_sel.rename({"waterlevel": "wl"})

    ds_model.append(ds_sel)


# ===============================================================================
# %% Model validation Statistics
# ===============================================================================

RPs = np.arange(1, 100, 1)

RIs_Gauges = []
for cnt1, gauge_ID in enumerate(Gauges):
    print(f"Processing Gauge {gauge_ID}")

    pull_gauge = ds_gauge[cnt1]
    pull_model = ds_model[cnt1]

    RIs_gauge_emp = calc_RIs_emp(
        pull_gauge[Var_process].values, pull_gauge["time"].values, RPs
    )

    RIs_Mod_emp = calc_RIs_emp(
        pull_model[Var_process].values, pull_model["time"].values, RPs
    )

    RIs_Mod_GPD, RIs_Mod_GPD_l, RIs_Mod_GPD_h = calc_RIs_GPD(
        pull_model[Var_process].values, pull_model["time"].values, RPs
    )

    Rps = pd.DataFrame(
        {
            "RI_emp_Gauge": RIs_gauge_emp.flatten(),
            "RI_emp_Model": RIs_Mod_emp.flatten(),
            "RI_Mod_Gauge": RIs_Mod_GPD.flatten(),
            "RI_Mod_Gauge_low": RIs_Mod_GPD_l.flatten(),
            "RI_Mod_Gauge_high": RIs_Mod_GPD_h.flatten(),
        },
        index=RPs,
    )

    RIs_Gauges.append(Rps)

# ===============================================================================
# %% Plots
# ===============================================================================

data = RIs_Gauges[3]

fig, ax = matplotlib.pyplot.subplots(1, 1, figsize=[8, 6])

ax.plot(data.index, data["RI_emp_Gauge"], "r", label="Gauge Empirical")
ax.plot(data.index, data["RI_emp_Model"], "b", label="Model Empirical")
ax.plot(data.index, data["RI_Mod_Gauge"], "g", label="Gauge GPD")
ax.fill_between(
    data.index,
    data["RI_Mod_Gauge_low"],
    data["RI_Mod_Gauge_high"],
    color="g",
    alpha=0.5,
    label="Gauge GPD 95% CI",
)
ax.grid()
ax.legend()
ax.set_title("Seattle Extreme Water levels")
ax.set_xlabel("Return Period (year)")
ax.set_ylabel("Return Level (WL,m)")


fig.savefig(os.path.join(dir_out, "ReturnIntervalCurves_Seattle.tiff"), dpi=300)


fig, ax = matplotlib.pyplot.subplots(2, 4, figsize=[10, 4])
ax = ax.flatten()

for cnt, gauge_ID in enumerate(Gauges):
    data = RIs_Gauges[cnt]

    ax[cnt].plot(data.index, data["RI_emp_Gauge"], "r", label="Gauge Empirical")
    ax[cnt].plot(data.index, data["RI_emp_Model"], "b", label="Model Empirical")
    ax[cnt].plot(data.index, data["RI_Mod_Gauge"], "g", label="Gauge GPD")
    ax[cnt].fill_between(
        data.index,
        data["RI_Mod_Gauge_low"],
        data["RI_Mod_Gauge_high"],
        color="g",
        alpha=0.5,
        label="Gauge GPD 95% CI",
    )
    # ax[cnt].set_ylim([2.7,4.3])
    ax[cnt].grid()
    ax[cnt].set_title(Gauge_name[cnt])

    if cnt < 4:
        ax[cnt].set_xticklabels([])

    # if (cnt == 1) or (cnt == 2) or (cnt == 3) or (cnt == 5) or (cnt == 6) or (cnt == 7):
    #    ax[cnt].set_yticklabels([])


ax[7].set_xticklabels([])
ax[7].set_yticklabels([])

# Add legend to the unused subplot (ax[7])
ax[7].axis("off")
handles, labels = ax[0].get_legend_handles_labels()
ax[7].legend(handles, labels, loc="center")


fig.savefig(os.path.join(dir_out, "ReturnIntervalCurves_AllGauges.tiff"), dpi=300)
