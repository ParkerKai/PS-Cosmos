# -*- coding: utf-8 -*-
"""
Created on Sept, 29 2025

This script plots a timeseries of Waves and Waterlevels with the difference predicted by CMIP6


@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"


# ===============================================================================
# %% Import Modules
# ===============================================================================
import os
from glob import glob
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
import pickle
import scipy
import h5py
import sys
from matplotlib import pyplot as plt

# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the DFM data resides
# dir_in = r'D:\DFM'
dir_in_era5 = r"Y:\PS_Cosmos\02_models\DFM_Regional\ERA5\ERA5"
dir_in_diff = r"Y:\PS_Cosmos\02_models\DFM_Regional\cdf_diff"
dir_in_TG = r"Y:\PS_Cosmos\01_data\WaterLevels\TideGauge"


dir_in_waves = r"Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries"
dir_out = r"Y:\PS_Cosmos\Figures\Paper"

# SLR_list =['000','025','050','100','150','200','300']
SLR = "000"
county = "King"

# Station (Seattle: 9447130)
Stat_Lat = float(47) + (float(36.2) / 60)
Stat_Lon = -(float(122) + (float(20.4) / 60))

year = 2021

# ===============================================================================
# %% Define some functions
# ===============================================================================
sys.path.append(r"C:\Users\kaparker\GitHub\Python\Kai_Python\General_Functions")
from Kai_MatlabTools import matlab2datetime
from Kai_GeoTools import distance_ll
from Kai_Timeseries import gappy_interp


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
gauge_ID = "9447130"  # Seattle
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

ds_wl_gauge = xr.Dataset(data_vars=data_vars, coords=coords)


# ===============================================================================
# %% Load the WL data
# ===============================================================================

# files = glob(os.path.join(dir_in_era5,'ERA5_cdf*'))
files = os.path.join(
    dir_in_era5, f"ERA5_{SLR}", "Results_Combined", f"ERA5_cdf_{year}.nc"
)

ds_wl = xr.open_mfdataset(
    files, engine="h5netcdf", parallel=True, chunks={"station": 1, "time": -1}
)

# Reset station as unicode
# THis is so adding ds_diff can match stations
station_id = ds_wl["station"].to_numpy().astype("unicode")
ds_wl = ds_wl.assign_coords({"station": station_id})

# ===============================================================================
# %% Load the CMIP6 difference data
# ===============================================================================

# Load the cmip6 difference data (monthly)
files = [
    os.path.join(dir_in_diff, SLR, f"ERA5wl_Diff_{year}_{month:02d}.nc")
    for month in np.arange(1, 13, 1)
]


ds_diff = xr.open_mfdataset(
    files,
    engine="netcdf4",
    parallel=True,
    chunks={"station": 1, "time": -1, "cmip6": 1},
)

# Add to create a new file
ds_diff = ds_diff.chunk({"station": 1, "time": -1, "cmip6": 1})

# Convert some of the data to integers for filesize savings

ds_diff = ds_diff.assign_coords({"station": station_id})
ds_wl["cmip_diff"] = ds_diff["cmip_diff"]


# SEt some attributes to the varialbes
ds_wl["cmip6"].attrs = {"long_name": "Cmip6 Model (HighResMIP)"}
ds_wl["waterlevel"].attrs = {
    "units": "meters/10000",
    "standard_name": "sea_surface_height",
    "long_name": "water level",
    "reference": "NAVD88",
}

ds_wl["wl_quants"].attrs = {
    "units": "None",
    "long_name": "Waterlevel Quantile (Monthly)",
    "Desc": "Quantiles determined monthly for all data in timeseries within specific month",
}

ds_wl["cmip_diff"].attrs = {
    "long_name": "Cmip6 Difference in water levels",
    "units": "meters/10000",
    "Desc": "Cmip6 difference for each ERA5 Waterlevel value (as determined by monthly quantile)",
}

# Drop data to just a single station
# Find index for stations.
Lat = ds_wl["lat"].values
Lon = ds_wl["lon"].values

dist = distance_ll(np.column_stack((Lat, Lon)), np.column_stack((Stat_Lat, Stat_Lon)))
ind_pull = np.argmin(dist)
print(
    f"Water Level station is {dist[ind_pull]} km away from selected station (Lat: {Stat_Lat}, Lon: {Stat_Lon})"
)

ds_wl = ds_wl.isel(station=ind_pull, drop=True)


# ===============================================================================
# %% Load the Wave data
# ===============================================================================

ds_waves = xr.open_mfdataset(
    os.path.join(dir_in_waves, f"LUT_{county}_CMIP6_Diff", SLR, f"ERA5_{year}_Diff.nc"),
    engine="netcdf4",
    parallel=True,
    chunks={"time": -1, "station": 1, "cmip6": 1},
)


# Drop data to just a single station
# Find index for stations.
Lat = ds_waves["Lat"].values
Lon = ds_waves["lon"].values

dist = distance_ll(np.column_stack((Lat, Lon)), np.column_stack((Stat_Lat, Stat_Lon)))
ind_pull = np.argmin(dist)
print(f"Wave station is {dist[ind_pull]} km from {county} station")
ds_waves = ds_waves.isel(station=ind_pull, drop=True)


# ===============================================================================
# %% Plots
# ===============================================================================

# fig, ax = matplotlib.pyplot.subplots(1, 1)
# fig.set_size_inches(8, 6)

# l1 = ax.plot(ds["time"], ds["Hs"], color="k", label="WaterLevel")

# for ii in range(ds["cmip6"].size):
#     ax.plot(ds["time"], ds["Hs"] + ds["cmip_diff"].isel(cmip6=ii))


# ax.set_xlim(pd.Timestamp("1942-11-01"), pd.Timestamp("1942-11-10"))
# # ax.set_xlim(100,200)
# ax.grid()
# ax.set_title("Wave Height")
# ax.set_ylabel("Hs (m)")
# ax.set_xlabel("Date")

# ax.legend(
#     ["ERA5", "CMCC", "CNRM", "EcEarth", "GFDL", "HadGemHH", "HadGemHM", "HadGemHMsst"]
# )
# fig.savefig(os.path.join(dir_out, "HS_Ts_AllMods.tiff"), dpi=600)


# #######################################################################

fig, ax = plt.subplots(2, 1)
fig.set_size_inches(8, 6)


ds_wl_gauge
l1 = ax[0].plot(
    ds_wl_gauge["time"], ds_wl_gauge["wl"], color="k", linestyle=":", label="gauge"
)


l2 = ax[0].plot(
    ds_wl["time"], ds_wl["waterlevel"] / 10000, color="r", label="waterlevel"
)

l3 = ax[0].fill_between(
    ds_wl["time"],
    ds_wl["waterlevel"] / 10000 + ds_wl["cmip_diff"].max(dim="cmip6") / 10000,
    ds_wl["waterlevel"] / 10000 + ds_wl["cmip_diff"].min(dim="cmip6") / 10000,
    alpha=0.5,
)

l4 = ax[0].plot(
    ds_wl["time"],
    ds_wl["waterlevel"] / 10000 + ds_wl["cmip_diff"].mean(dim="cmip6") / 10000,
    color="b",
)


ax[0].set_xlim(pd.Timestamp(f"{year}-12-20"), pd.Timestamp(f"{year}-12-31"))
ax[0].set_ylim(-2, 4.5)
ax[0].grid()
ax[0].set_title("Water Level")
ax[0].set_ylabel("WL (m)")
ax[0].xaxis.set_ticklabels([])

###############################

l1 = ax[1].plot(ds_waves["time"], ds_waves["Hs"], color="k", label="Waves")

l2 = ax[1].fill_between(
    ds_waves["time"],
    ds_waves["Hs"] + ds_waves["cmip_diff"].max(dim="cmip6"),
    ds_waves["Hs"] + ds_waves["cmip_diff"].min(dim="cmip6"),
    alpha=0.5,
)

l3 = ax[1].plot(
    ds_waves["time"],
    ds_waves["Hs"] + ds_waves["cmip_diff"].mean(dim="cmip6"),
    color="b",
)

ax[1].set_xlim(pd.Timestamp(f"{year}-12-20"), pd.Timestamp(f"{year}-12-31"))
ax[1].set_ylim(0, 0.4)
ax[1].grid()
ax[1].set_title("Significant Wave Height")
ax[1].set_ylabel("Hs (m)")
ax[1].set_xlabel("Date")
ax[1].legend(["ERA5", "Cmip6 Model Range", "Cmip6 Model Mean"], loc="upper left")


fig.savefig(os.path.join(dir_out, "Wl_Hs_Ts_Range.tiff"), dpi=600)
