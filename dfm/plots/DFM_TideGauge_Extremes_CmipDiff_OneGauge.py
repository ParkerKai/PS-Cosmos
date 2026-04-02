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
import h5py
from scipy.stats import t


# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the DFM data resides
# dir_in = r'D:\DFM'
dir_in_model = r"Y:\PS_Cosmos\02_models\DFM_Regional\ERA5"
dir_in_TG = r"Y:\PS_Cosmos\01_data\WaterLevels\TideGauge"
dir_in_cmip6 = r"Y:\PS_Cosmos\GIS\DFM\DFM_CmipDiff_byModel"
dir_out = r"Y:\PS_Cosmos\Figures\Paper"
dir_in_waves = r"Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries"
dir_in_cmip6_waves = r"Y:\PS_Cosmos\GIS\Waves\DFM_CmipDiff_byModel"

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
county = "King"

# Variable to process (calculate stats and make plots)
Var_process = "wl"  # 'wl', 'tide', 'ntr'

RPs_want = [1, 5, 10, 15, 30]


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


def LoadWaveLUTmats(file_in):
    # Load the .mat file using h5py
    with h5py.File(file_in, "r") as f:
        group = f["LUTout"]
        hs = group["Hs"][()]
        dm = group["Dm"][()]
        tp = group["Tp"][()]
        t = group["t"][:, 0][()]
        lat = np.squeeze(group["lat"][()])
        lon = np.squeeze(group["lon"][()])
        depth = np.squeeze(group["depth"][()])
        stat = np.arange(0, len(lon), dtype="int32")
        dfm_id = np.squeeze(group["DFMid"][()])

        # Convert from matlab to pandas datetimeIndex.
        # timeseries is in hours so round to hours to clean up conversion error.
        t_dt = matlab2datetime(t, "h")

    # Turn into an Xarray dataset
    ds = xr.Dataset(
        {
            "Hs": xr.DataArray(
                data=hs,  # enter data here
                dims=["time", "station"],
                coords={"time": t_dt, "station": stat},
                attrs={"_FillValue": -9999, "units": "meters"},
            ),
            "Dm": xr.DataArray(
                data=dm,  # enter data here
                dims=["time", "station"],
                coords={"time": t_dt, "station": stat},
                attrs={"_FillValue": -9999, "units": "degrees"},
            ),
            "Tp": xr.DataArray(
                data=tp,  # enter data here
                dims=["time", "station"],
                coords={"time": t_dt, "station": stat},
                attrs={"_FillValue": -9999, "units": "seconds"},
            ),
            "Lat": xr.DataArray(
                data=lat,  # enter data here
                dims=["station"],
                coords={"station": stat},
                attrs={"_FillValue": -9999, "units": "Degree"},
            ),
            "Lon": xr.DataArray(
                data=lon,  # enter data here
                dims=["station"],
                coords={"station": stat},
                attrs={"_FillValue": -9999, "units": "Degree"},
            ),
            "depth": xr.DataArray(
                data=depth,  # enter data here
                dims=["station"],
                coords={"station": stat},
                attrs={"_FillValue": -9999, "units": "m"},
            ),
            "DFMid": xr.DataArray(
                data=dfm_id,  # enter data here
                dims=["station"],
                coords={"station": stat},
                attrs={"_FillValue": -9999, "units": "ID"},
            ),
        },
        attrs={
            "DataSource": rf"Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries\LUT_output_{county}_{Period}",
            "ProducedBy": "Anita Englestad and Kai Parker",
            "General": "Tm,Tp and Dm are NaN for Hs < 0.05m. Output is for lat_10mIso/lon_10mIso locations. lat_ncPoint and lon_ncPoint are the original nc points to which the closest gridpoints on the 10m isobath were found",
        },
    )

    return ds


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

ds_wl_gauge = xr.Dataset(data_vars=data_vars, coords=coords)


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
ds_wl_model = ds_sel.rename({"waterlevel": "wl"})


# ===============================================================================
# %% Load the wave data
# ===============================================================================
Period = "ERA5"

print(f"Processing for: {county}")

if county == "SanJuan":
    file_in = os.path.join(
        dir_in_waves,
        f"LUT_output_{county}_{Period}",
        f"LUT_output_{county}_{Period}_5mIsobath.mat",
    )
else:
    file_in = os.path.join(
        dir_in_waves,
        f"LUT_output_{county}_{Period}",
        f"LUT_output_{county}_{Period}_10mIsobath.mat",
    )

ds = LoadWaveLUTmats(file_in)
ds = ds[["Hs", "Lon", "Lat"]]
WaveStat = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(ds["Lon"].values, ds["Lat"].values, crs="EPSG:4326")
)

# 1. Extract lat/lon from ds_model
model_lon = ds_wl_model["lon"].isel(time=0).values
model_lat = ds_wl_model["lat"].isel(time=0).values

# 2. Find the closest point in data
distance = WaveStat.geometry.distance(
    gpd.points_from_xy([model_lon], [model_lat], crs="EPSG:4326")[0]
)
ind_station_pull = distance.idxmin()

ds_wave_era5 = ds.isel(station=ind_station_pull)

# ===============================================================================
# %% CMIP6
# ===============================================================================


# ===============================================================================
# %% Model RI Statistics
# ===============================================================================

RPs = np.arange(1, 100, 1)


pull_gauge = ds_wl_gauge
pull_model = ds_wl_model

RIs_gauge_emp = calc_RIs_emp(
    ds_wl_gauge[Var_process].values, ds_wl_gauge["time"].values, RPs
)

RIs_Mod_emp = calc_RIs_emp(
    ds_wl_model[Var_process].values, ds_wl_model["time"].values, RPs
)

RIs_Mod_GPD, RIs_Mod_GPD_l, RIs_Mod_GPD_h = calc_RIs_GPD(
    ds_wl_model[Var_process].values, ds_wl_model["time"].values, RPs
)

RIs_WL = pd.DataFrame(
    {
        "RI_emp_Gauge": RIs_gauge_emp.flatten(),
        "RI_emp_Model": RIs_Mod_emp.flatten(),
        "RI_GPD": RIs_Mod_GPD.flatten(),
        "RI_GPD_low": RIs_Mod_GPD_l.flatten(),
        "RI_GPD_high": RIs_Mod_GPD_h.flatten(),
    },
    index=RPs,
)

RIs_Mod_emp = calc_RIs_emp(ds_wave_era5["Hs"].values, ds_wave_era5["time"].values, RPs)


RIs_Mod_GPD, RIs_Mod_GPD_l, RIs_Mod_GPD_h = calc_RIs_GPD(
    ds_wave_era5["Hs"].values, ds_wave_era5["time"].values, RPs
)


RIs_Wave = pd.DataFrame(
    {
        "RI_emp": RIs_Mod_emp.flatten(),
        "RI_gpd": RIs_Mod_GPD.flatten(),
        "RI_gpd_l": RIs_Mod_GPD_l.flatten(),
        "RI_gpd_h": RIs_Mod_GPD_h.flatten(),
    },
    index=RPs,
)


# ===============================================================================
# %% Read in the CMIP6 WL Model Data
# ===============================================================================
slr = "000"

for cnt, Mod in enumerate(Mod_list):
    print(f"Processing: {Mod}")
    file_in = os.path.join(dir_in_cmip6, f"Dfm_diff{slr}_{Mod}.shp")

    data = gpd.read_file(file_in)

    if cnt == 0:
        # 1. Extract lat/lon from ds_model
        model_lon = ds_wl_model["lon"].isel(time=0).values
        model_lat = ds_wl_model["lat"].isel(time=0).values

        # 2. Find the closest point in data
        data["distance"] = data.geometry.distance(
            gpd.points_from_xy([model_lon], [model_lat], crs=data.crs)[0]
        )
        ind_station_pull = data["distance"].idxmin()

        cmip6 = data.loc[ind_station_pull, :].drop(labels=["geometry", "distance"])

    else:
        cmip6 = pd.concat(
            [cmip6, data.loc[ind_station_pull, :].drop(labels="geometry")], axis=1
        )


Cmip6_RIs = cmip6.loc[["RI_1", "RP_5", "RP_10", "RP_15", "RP_30"]].to_numpy()

average_RIs = RIs_WL["RI_emp_Model"].iloc[[0, 4, 9, 14, 29]].to_numpy()
average_RIs_matrix = np.tile(average_RIs.reshape(-1, 1), (1, 7))

Cmip6_RIs = average_RIs_matrix + Cmip6_RIs
Cmip6_RPs = np.array([1, 5, 10, 15, 30])


Cmip6_WL_RIs = np.array(Cmip6_RIs, dtype=float)
Cmip6_WL_RIs_CIs = t.interval(
    0.95,
    Cmip6_WL_RIs.shape[1],
    loc=np.mean(Cmip6_WL_RIs, 1),
    scale=np.std(Cmip6_WL_RIs, 1),
)


# ===============================================================================
# %% Read in the CMIP6 Wave Model Data
# ===============================================================================
slr = "000"

Mod_list = ["CNRM", "EcEarth", "GFDL", "HadGemHH", "HadGemHM", "HadGemHMsst", "CMCC"]

for cnt, Mod in enumerate(Mod_list):
    print(f"Processing: {Mod}")
    file_in = os.path.join(dir_in_cmip6_waves, f"Wave_diff{slr}_{Mod}.shp")

    data = gpd.read_file(file_in)

    if cnt == 0:
        # 1. Extract lat/lon from ds_model
        model_lon = ds_wl_model["lon"].isel(time=0).values
        model_lat = ds_wl_model["lat"].isel(time=0).values

        # 2. Find the closest point in data
        data["distance"] = data.geometry.distance(
            gpd.points_from_xy([model_lon], [model_lat], crs=data.crs)[0]
        )
        ind_station_pull = data["distance"].idxmin()

        cmip6 = data.loc[ind_station_pull, :].drop(labels=["geometry", "distance"])

    else:
        cmip6 = pd.concat(
            [cmip6, data.loc[ind_station_pull, :].drop(labels="geometry")], axis=1
        )


Cmip6_RIs = cmip6.loc[["RI_1", "RP_5", "RP_10", "RP_15", "RP_30"]].to_numpy()

average_RIs = RIs_Wave["RI_emp"].iloc[[0, 4, 9, 14, 29]].to_numpy()
average_RIs_matrix = np.tile(average_RIs.reshape(-1, 1), (1, 7))

Cmip6_RIs = average_RIs_matrix + Cmip6_RIs
Cmip6_RPs = np.array([1, 5, 10, 15, 30])


Cmip6_Wave_RIs = np.array(Cmip6_RIs, dtype=float)


Cmip6_Wave_RIs_CIs = t.interval(
    0.95,
    Cmip6_Wave_RIs.shape[1],
    loc=np.mean(Cmip6_Wave_RIs, 1),
    scale=np.std(Cmip6_Wave_RIs, 1),
)


# ===============================================================================
# %% Plots
# ===============================================================================


fig, ax = matplotlib.pyplot.subplots(2, 1, figsize=[8, 7])

# ax[0].plot(RIs_WL.index, RIs_WL["RI_emp_Gauge"], "r.",label="Gauge Empirical")
ax[0].plot(
    RIs_WL.index, RIs_WL["RI_emp_Model"], "k", label="Model Empirical", linewidth=2
)
ax[0].plot(
    Cmip6_RPs, np.average(Cmip6_WL_RIs, 1), "b", label="CMIP6 Average", linewidth=2
)


# ax.plot(RIs_Gauges.index, RIs_Gauges["RI_Mod_Gauge"], "g", label="Gauge GPD")
ax[0].plot(Cmip6_RPs, Cmip6_WL_RIs[:, 0], "--", label="CNRM")
ax[0].plot(Cmip6_RPs, Cmip6_WL_RIs[:, 1], "--", label="EC-Earth")
ax[0].plot(Cmip6_RPs, Cmip6_WL_RIs[:, 2], "--", label="GFDL")
ax[0].plot(Cmip6_RPs, Cmip6_WL_RIs[:, 3], "--", label="HadGEM_HH")
ax[0].plot(Cmip6_RPs, Cmip6_WL_RIs[:, 4], "--", label="HadGEM_HM")
ax[0].plot(Cmip6_RPs, Cmip6_WL_RIs[:, 5], "--", label="HadGEM_HMsst")
ax[0].plot(Cmip6_RPs, Cmip6_WL_RIs[:, 6], "--", label="CMCC")

ax[0].fill_between(
    Cmip6_RPs,
    Cmip6_WL_RIs_CIs[0],
    Cmip6_WL_RIs_CIs[1],
    color="k",
    alpha=0.5,
    label="Gauge GPD 95% CI",
)

ax[0].legend()
ax[0].set_title("Seattle Extreme Water levels")
ax[0].set_ylabel("Return Level (WL,m)")

ax[0].set_xscale("log")
ax[0].set_xlim([1, 30])
ax[0].set_xticklabels([])
ax[0].grid(which="both")


ax[1].plot(
    RIs_Wave.index, RIs_Wave["RI_emp"], "k", label="Model Empirical", linewidth=2
)
ax[1].plot(
    Cmip6_RPs, np.average(Cmip6_Wave_RIs, 1), "b", label="CMIP6 Average", linewidth=2
)

ax[1].plot(Cmip6_RPs, Cmip6_Wave_RIs[:, 0], "--", label="CNRM")
ax[1].plot(Cmip6_RPs, Cmip6_Wave_RIs[:, 1], "--", label="EC-Earth")
ax[1].plot(Cmip6_RPs, Cmip6_Wave_RIs[:, 2], "--", label="GFDL")
ax[1].plot(Cmip6_RPs, Cmip6_Wave_RIs[:, 3], "--", label="HadGEM_HH")
ax[1].plot(Cmip6_RPs, Cmip6_Wave_RIs[:, 4], "--", label="HadGEM_HM")
ax[1].plot(Cmip6_RPs, Cmip6_Wave_RIs[:, 5], "--", label="HadGEM_HMsst")
ax[1].plot(Cmip6_RPs, Cmip6_Wave_RIs[:, 6], "--", label="CMCC")

ax[1].fill_between(
    Cmip6_RPs,
    Cmip6_Wave_RIs_CIs[0],
    Cmip6_Wave_RIs_CIs[1],
    color="k",
    alpha=0.5,
    label="Gauge GPD 95% CI",
)


# ax[1].legend()
ax[1].set_title("Seattle Extreme Wave Heights")
ax[1].set_xlabel("Return Period (year)")
ax[1].set_ylabel("Return Level (Hs,m)")

ax[1].set_xscale("log")
ax[1].set_xlim([1, 30])
ax[1].grid(which="both")


fig.savefig(os.path.join(dir_out, "ReturnIntervalCurves_Seattle.tiff"), dpi=300)

# %%
