# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:14:27 2024

This script calculates the cdf  for each model wave data.
CDF is calculated monthly
This can then be applied to the reanalysis period.


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
import numpy as np
import xarray as xr
import pandas as pd
import h5py

# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the DFM data resides
# dir_in = r'D:\DFM'
dir_in = r"Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries"
dir_out = r"Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries"

# Model to process
ModList = [ "CMCC", "CNRM", "EcEarth", "GFDL", "HadGemHH", "HadGemHM", "HadGemHMsst"]   # "CMCC", "CNRM", "EcEarth", "GFDL", "HadGemHH", "HadGemHM", "HadGemHMsst"

# Period to process
Per = "future"

SLR_list = ["000", "025", "050", "100", "150", "200", "300"]

Var = "Hs"  #  ['Dm', 'Hs', 'Tm', 'Tp']


# county_list = [
#     "Kitsap",
#     "Clallam",
#     "Snohomish",
#     "IslandCounty",
#     "Skagit",
#     "Jefferson",
#     "King",
#     "Pierce",
#     "Thurston",
#     "Whatcom",
#     "Mason",
#     "SanJuan",
# ]

county_list = [
    "IslandCounty","Jefferson"]

# ===============================================================================
# %% Define some functions
# ===============================================================================


def matlab2datetime(matlab_datenum, round_to):
    import pandas as pd

    t = pd.to_datetime(matlab_datenum - 719529, unit="D")

    # Solution isn't exact so this round cleans things up
    t_clean = t.round(freq=round_to)  # hour

    return t_clean


def emp_cdf_xr(data, stat):
    import scipy

    data = data[~np.isnan(data)]

    # Calculate the ecdf
    res = scipy.stats.ecdf(data)

    data_out = pd.DataFrame(
        data={
            "values": res.cdf.quantiles,
            "cdf": res.cdf.probabilities,
            "stat": stat * np.ones((len(res.cdf.quantiles),), dtype=int),
        }
    )

    return data_out


def emp_cdf(data, var):
    import pandas as pd

    # data: Xarray dataset
    # var: variable to calculate the cdf for

    # Load the xarray data into memory
    cdf = []
    if isinstance(data, pd.DataFrame):
        num_stats = data.shape[1]

    elif isinstance(data, xr.DataArray):
        num_stats = data.dims["station"]

    for stat in range(num_stats):  #
        # print("processing Station: {}".format(stat))
        # pull data at this station

        if isinstance(data, pd.DataFrame):
            vals = data.loc[:, stat].to_numpy()

        elif isinstance(data, xr.DataArray):
            vals = data[var].isel(station=stat).values

        # Calculate the cdf

        cdf.append(emp_cdf_xr(vals, stat))

    out = pd.concat(cdf)

    return out


def monthly_CDF(data, var, month):
    if isinstance(data, pd.DataFrame):
        data_month = data.iloc[data.index.month == month]
    elif isinstance(data, xr.DataArray):
        data_month = data.sel(time=data.time.dt.month.isin(month))

    cdf_month = emp_cdf(data_month, var)

    return cdf_month


# ===============================================================================
# %% Load the data
# ===============================================================================
for county in county_list:
    for Mod in ModList:
        # CMIP6 Fut
        for SLR in SLR_list:
            print(f"Processing {county}, {Mod} for future Period {SLR} SLR")

            file_in = os.path.join(
                dir_in,
                f"LUT_output_{county}_CMIP6_{Per}",
                Mod,
                f"LUT_output_{county}_{Mod}_SLR{SLR}.mat",
            )

            # Load the .mat file using h5py
            with h5py.File(file_in, "r") as f:
                group = f["LUTout"]
                data = group[Var][()]
                t = group["t"][:, 0][()]

                # Convert from matlab to pandas datetimeIndex.
                # timeseries is in hours so round to hours to clean up conversion error.
                t_dt = matlab2datetime(t, "h")

                # Convert matlab datenum to pandas timestamp
                cmip = pd.DataFrame(data, index=t_dt)

            # split by month
            for month in np.arange(1, 13, 1, dtype=int):
                print("Processing Month: {}".format(month))

                cdf_month = monthly_CDF(cmip, "waterlevel", month)

                sub_dir_out = os.path.join(
                    dir_out, f"LUT_output_{county}_CMIP6_{Per}", Mod
                )

                if not os.path.exists(sub_dir_out):
                    os.makedirs(sub_dir_out)

                cdf_month.to_pickle(
                    os.path.join(
                        sub_dir_out,
                        f"CDFmonthly_{month:02d}_{Mod:s}_{county}_SLR{SLR:s}.pkl",
                    )
                )
