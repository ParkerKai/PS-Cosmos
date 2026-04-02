# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:45:11 2024

This script calculates Statistics for a wave dataset and exports it as a shapefile.
Can run for ERA5 or CMIP6 historic/future data

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
import scipy
import geopandas as gpd
import h5py
import pyextremes
import scipy.optimize as optimize
import scipy
import sys

# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the DFM data resides
# dir_in = r'D:\DFM'
dir_in = r"Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries"
dir_out = r"Y:\PS_Cosmos\GIS\Waves"

county_list = [
    "King",
    "Pierce",
    "Thurston",
]

# Return Periods wanted
# RPs_want = [1, 10, 25, 50, 80]
RPs_want = [1, 5, 10, 15, 30]

# 'CMIP6_historic' 'CMIP6_future' 'ERA5'
Period = "ERA5"

# Model  (if cmip6)  CMCC CNRM EcEarth GFDL HadGemHH HadGemHM HadGemHMsst
Mod = "HadGemHMsst"


# ===============================================================================
# %% Define some functions
# ===============================================================================
sys.path.append(
    r"C:\Users\kaparker\GitHub\Python\Kai_Python\General_Functions"
)
from Kai_MatlabTools import matlab2datetime
from Kai_XarrayTools import Get_Station_index
from Kai_EVATools import calc_RIs_emp


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
        #dfm_id = np.squeeze(group["DFMid"][()])

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
            #"DFMid": xr.DataArray(
            #    data=dfm_id,  # enter data here
            #    dims=["station"],
            #    coords={"station": stat},
            #    attrs={"_FillValue": -9999, "units": "ID"},
            #),
        },
        attrs={
            "DataSource": rf"Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries\LUT_output_{county}_{Period}",
            "ProducedBy": "Anita Englestad and Kai Parker",
            "General": "Tm,Tp and Dm are NaN for Hs < 0.05m. Output is for lat_10mIso/lon_10mIso locations. lat_ncPoint and lon_ncPoint are the original nc points to which the closest gridpoints on the 10m isobath were found",
        },
    )

    return ds


# ===============================================================================
# %% Save
# ===============================================================================

stat_save = []
for county in county_list:
    print(f"Processing for: {county}")

    if Period == "ERA5":
        if county == "SanJuan":
            file_in = os.path.join(
                dir_in,
                f"LUT_output_{county}_{Period}",
                f"LUT_output_{county}_{Period}_5mIsobath.mat",
            )
        else:
            file_in = os.path.join(
                dir_in,
                f"LUT_output_{county}_{Period}",
                f"LUT_output_{county}_{Period}_10mIsobath.mat",
            )

    elif Period == "CMIP6_historical":
        file_in = os.path.join(
            dir_in,
            f"LUT_output_{county}_{Period}",
            f"LUT_output_{county}_{Mod}_his.mat",
        )

    elif Period == "CMIP6_future":
        file_in = os.path.join(
            dir_in,
            f"LUT_output_{county}_{Period}",
            f"{Mod}",
            f"LUT_output_{county}_{Mod}_SLR000.mat",
        )

    ds = LoadWaveLUTmats(file_in)

    data = ds["Hs"].values
    RIs = calc_RIs_emp(data, ds["time"].values, RPs=np.array(RPs_want))

    # Turn into a dataframe
    d = {
        "Mean": np.nanmean(data, axis=0),
        "Max": np.nanmax(data, axis=0),
        "Std": np.nanstd(data, axis=0),
        "Q99": np.nanquantile(data, 0.99, axis=0),
        "Q95": np.nanquantile(data, 0.95, axis=0),
        "RI_1": RIs[0, :],
        "RP_5": RIs[1, :],
        "RP_10": RIs[2, :],
        "RP_15": RIs[3, :],
        "RP_30": RIs[4, :],
        "NumNan": np.count_nonzero(np.isnan(data), axis=0),
    }

    # Convert to pandas dataframe
    stats = pd.DataFrame(data=d)

    stats = stats.assign(county=np.full(stats.shape[0], county))

    # add geometry and turn into geopandas dataset
    geometry = gpd.points_from_xy(x=ds["Lon"], y=ds["Lat"], crs="EPSG:4326")

    # Convert to a geopandas dataframe and save
    stat_save.append(gpd.GeoDataFrame(data=stats, geometry=geometry))


# Aggregate across counties and save.
stat_save = pd.concat(stat_save, axis=0)
if Period == "ERA5":
    file_out = os.path.join(dir_out, Period, f"WaveHs_{Period}.shp")

elif (Period == "CMIP6_historical") or (Period == "CMIP6_future"):
    file_out = os.path.join(dir_out, Period, f"WaveHs_{Period}_{Mod}.shp")

print(f"Saving file: {file_out}")

stat_save.to_file(file_out)
