# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:22:20 2024

This script plots the waterlevels for CMIP6 historic-future runs


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
import geopandas as gpd
import scipy
import sys

# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the DFM data resides
# dir_in = r'D:\DFM'
dir_in = r"Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries"
dir_out = r"Y:\PS_Cosmos\GIS\Waves\Wave_CmipDiff_byModel"

# Model to process
# od_list = ['CMCC','CNRM','EcEarth','GFDL','HadGemHH','HadGemHM','HadGemHMsst']
Mod_list = ['CMCC','CNRM',"EcEarth", "GFDL", "HadGemHH", "HadGemHM", "HadGemHMsst"]

SLR_list = ["000", "025", "050", "100", "150", "200", "300"]

county_list = [
    "Kitsap",
    "Snohomish",
    "IslandCounty",
    "Skagit",
    "Jefferson",
    "King",
    "Pierce",
    "Thurston",
    "Whatcom",
    "Mason",
    "SanJuan",
    'Clallam',
]   #     "Clallam",




# ===============================================================================
# %% Define some functions
# ===============================================================================
sys.path.append(
    r"C:\Users\kaparker\GitHub\Python\Kai_Python\General_Functions"
)
from Kai_MatlabTools import matlab2datetime


def CalcStats(data, Var):
    # Data: Xarray dataset
    # dim: Condense along this dimention. So statistics are for this dimention (e.ge. mean of time if dimention is time

    import pyextremes

    num_stat = data[Var].shape[1]
    time = data["time"].values

    stats = np.full((num_stat, 10), np.nan)
    for cnt in range(num_stat):
        print(f"Processing Station: {cnt}")
        # Load values into memeory
        data_pull = data[Var].isel(station=cnt).values

        if (np.nansum(data_pull) == 0) or (
            np.count_nonzero(np.isnan(data_pull)) == len(data_pull)) or (
            data['lon'].isel(station=cnt).values < -124.526):
                
            print("No Data Station")

        
        
        else:
            # Calculate statistics on data
            stats[cnt, 0] = np.nanmean(data_pull)  # Mean
            stats[cnt, 1] = np.nanmax(data_pull)  # Max
            stats[cnt, 2] = np.nanstd(data_pull)  # Standard Deviation
            stats[cnt, 3] = np.nanquantile(data_pull, 0.99)  # 99th Quantile
            stats[cnt, 4] = np.nanquantile(data_pull, 0.95)  # 95th Quantile

            # Load TWL values into memeory as a pandas series
            data_series = pd.Series(data_pull, index=time)

            # Get number of exceedances to grab
            num_years = np.unique(data_series.index.year).shape[0]
            Npryr = 3
            Num_Exce = (num_years - 1) * Npryr

            selected_thresh = POT_theshold_SetNum(
                data_series,  # dataset
                Var,  # column name of pd series
                "72h",  # independence time delta r
                Num_Exce,
            )  # number of exceedences we want

            extremes = pyextremes.extremes.get_extremes(
                data_series,
                method="POT",
                extremes_type="high",
                threshold=selected_thresh,
                r="72h",
            )

            return_periods = pyextremes.get_return_periods(
                ts=data_series,
                extremes=extremes,
                extremes_method="POT",
                extremes_type="high",
                return_period_size="365.2425D",
                plotting_position="weibull",
            )

            return_periods.sort_values("return period", ascending=True, inplace=True)

            interp_rp = scipy.interpolate.interp1d(
                return_periods["return period"].to_numpy(),
                return_periods["extreme values"].to_numpy(),
                copy=False,
                assume_sorted=True,
                fill_value=(return_periods["extreme values"].min(), np.nan),
                bounds_error=False,
            )

            stats[cnt, 5] = interp_rp(np.array(1))
            stats[cnt, 6] = interp_rp(np.array(5))
            stats[cnt, 7] = interp_rp(np.array(10))
            stats[cnt, 8] = interp_rp(np.array(15))
            stats[cnt, 9] = interp_rp(np.array(30))

    # Turn into a dataframe
    d = {
        "Mean": stats[:, 0],
        "Max": stats[:, 1],
        "Std": stats[:, 2],
        "Q99": stats[:, 3],
        "Q95": stats[:, 4],
        "RI_1": stats[:, 5],
        "RP_5": stats[:, 6],
        "RP_10": stats[:, 7],
        "RP_15": stats[:, 8],
        "RP_30": stats[:, 9],
    }

    # Convert to pandas dataframe
    stats_out = pd.DataFrame(data=d)

    # add geometry and turn into geopandas dataset
    geometry = gpd.points_from_xy(x=data["lon"], y=data["lat"], crs="EPSG:4326")

    out = gpd.GeoDataFrame(data=stats_out, geometry=geometry)

    return out


def CalcDiff(dataH, dataF):
   
    # Convert to numpy for easy diff calculating
    H = pd.DataFrame(dataH.drop(columns="geometry")).to_numpy()
    F = pd.DataFrame(dataF.drop(columns="geometry")).to_numpy()

    # Calculate the difference (future - Historic)
    diff = F - H

    # Pull out the columns of the stats array
    columns = dataH.columns

    # Re-build the pandas dataframe
    df_diff = pd.DataFrame(diff, columns=columns[0:-1])

    # Convert to a geopandas dataframe
    out = gpd.GeoDataFrame(data=df_diff, geometry=dataH["geometry"])

    return out


def LoadWaveLUTmats(file_in, Var):
    import h5py

    # Load the .mat file using h5py
    with h5py.File(file_in, "r") as f:
        group = f["LUTout"]
        data = group[Var][()]
        t = group["t"][:, 0][()]

        # Convert from matlab to pandas datetimeIndex.
        # timeseries is in hours so round to hours to clean up conversion error.
        t_dt = matlab2datetime(t, "h")

        num_stat = data.shape[1]
        # Wrestle into an xarray dataset
        data_vars = {
            Var: (["time", "station"], data),
            "lat": (["station"], group["lat"][()].flatten()),
            "lon": (["station"], group["lon"][()].flatten()),
            "depth": (["station"], group["depth"][()].flatten()),
        }

        # define coordinates
        coords = {
            "time": (["time"], t_dt, {"standard_name": "time"}),
            "station": (["station"], np.arange(0, num_stat, 1)),
        }

        # create dataset
        ds = xr.Dataset(data_vars=data_vars, coords=coords)

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
    bnds = scipy.optimize.Bounds(lb=B[y_lab].mean(), ub=B[y_lab].iloc[1])
    Optim_out = scipy.optimize.minimize(
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


# ===============================================================================
# %% Load the data
# ===============================================================================
# CMIP6 His


for Mod in Mod_list:
    for SLR in SLR_list:
        diff_save = []
        for county in county_list:
            print(f"Processing for {Mod}, {SLR}, {county}")

            file_in = os.path.join(
                dir_in,
                f"LUT_output_{county}_CMIP6_historical",
                f"LUT_output_{county}_{Mod}_his.mat",
            )
            data_H = LoadWaveLUTmats(file_in, "Hs")

            stats_H = CalcStats(data_H, "Hs")

            # CMIP6 Fut
            file_in = os.path.join(
                dir_in,
                f"LUT_output_{county}_CMIP6_future",
                Mod,
                f"LUT_output_{county}_{Mod}_SLR{SLR}.mat",
            )
            data_F = LoadWaveLUTmats(file_in, "Hs")

            stats_F = CalcStats(data_F, "Hs")
            diff = CalcDiff(stats_H, stats_F)

            diff = diff.assign(county=np.full(diff.shape[0], county))

            diff_save.append(diff)

        # Aggregate across counties and save.
        diff_out = pd.concat(diff_save, axis=0)
        diff_out.to_file(os.path.join(dir_out, f"Wave_diff{SLR}_{Mod}.shp"))
