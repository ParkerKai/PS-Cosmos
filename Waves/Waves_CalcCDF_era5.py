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
import dask.distributed
import h5py
import scipy

# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the DFM data resides
# dir_in = r'D:\DFM'
dir_in = r"Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries"
dir_out = r"Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries"


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
]
county_list = ['Clallam']

Var = "Hs"  #  ['Dm', 'Hs', 'Tm', 'Tp']

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
    # data: Xarray dataset
    # var: variable to calculate the cdf for

    # Load the xarray data into memory
    quants = np.full(shape=data[var].shape, fill_value=np.nan, dtype="float64")
    num_stats = data.dims["station"]
    for stat in range(num_stats):  # data.dims['station']
        print("processing Station: {}".format(stat))
        # pull data at this station

        vals_era5 = data[var].isel(station=stat).values

        # Calculate the cdf
        # cdf.append(emp_cdf_xr(vals,stat))
        cdf = emp_cdf_xr(vals_era5, stat)

        # Break into values and quantiles
        era5_cdf_vals = cdf["values"].to_numpy()
        era5_cdf_quants = cdf["cdf"].to_numpy()

        if era5_cdf_quants.min() > 0.1:
            print('ERROR!')
                    
        # quants[: ,stat] = interp2quant(era5_cdf_vals,era5_cdf_quants,vals_era5).compute()
        if era5_cdf_vals.shape[0] != 0:
            quants[:, stat] = interp2quant(era5_cdf_vals, era5_cdf_quants, vals_era5)

    return quants


def interp2quant(cdf_vals, cdf_quant, data_vals):
    # Determine CDF based on the pre-calculated ERA5 cdf
    interp_era5 = scipy.interpolate.interp1d(
        cdf_vals,
        cdf_quant,
        fill_value=(0, 1),
        copy=False,
        assume_sorted=True,
        bounds_error=False,
    )

    quants = interp_era5(data_vals)

    return quants


def output_yearly(data, dir_out, fname):
    import time

    max_retries = 10
    delay = 1

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    year_out = np.unique(data.time.dt.year)
    for year in year_out:
        out = data.isel(time=data.time.dt.year.isin(year))

        file_out = os.path.join(dir_out, fname.format(year=year))

        if os.path.exists(file_out):
            print(f"{year} already exists so skipping")

        else:
            # Output with error handling
            for attempt in range(1, max_retries + 1):
                try:
                    out.to_netcdf(file_out, engine="netcdf4")
                    print(f"Successfully saved {year}")
                    break

                except Exception as e:
                    print(f"Attempt {attempt} failed: {e}")
                    if attempt < max_retries:
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)

                    else:
                        print("All attempts failed. Dataset not saved.")
                        raise  # Re-raise the last exception if all retries fa


# ===============================================================================
# %% Load the data
# ===============================================================================
# ERA5

for county in county_list:
    print(f"Processing for ERA5 Period: {county}")

    file_in = os.path.join(
        dir_in,
        f"LUT_output_{county}_ERA5",
        f"LUT_output_{county}_ERA5_10mIsobath.mat",
    )

    # Load the .mat file using h5py
    with h5py.File(file_in, "r") as f:
        group = f["LUTout"]
        hs = group["Hs"][()]
        dm = group["Dm"][()]
        tm = group["Tm"][()]
        t = group["t"][:, 0][()]
        lat = np.squeeze(group["lat"][()])
        lon = np.squeeze(group["lon"][()])
        depth = np.squeeze(group["depth"][()])
        stat = np.arange(0, len(lon), dtype="int32")

        # Convert from matlab to pandas datetimeIndex.
        # timeseries is in hours so round to hours to clean up conversion error.
        t_dt = matlab2datetime(t, "h")

    # Turn into an Xarray dataset
    ds_era5 = xr.Dataset(
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
            "Tm": xr.DataArray(
                data=tm,  # enter data here
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
        },
        attrs={
            "DataSource": file_in,
            "ProducedBy": "Anita Englestad and Kai Parker",
            "General": "Tm,Tp and Dm are NaN for Hs < 0.05m. Output is for lat_10mIso/lon_10mIso locations. lat_ncPoint and lon_ncPoint are the original nc points to which the closest gridpoints on the 10m isobath were found",
        },
    )

    # split by month
    quants_full = np.full(ds_era5["Hs"].shape, np.nan, dtype="float32")
    for month in np.arange(1, 13, 1, dtype=int):
        print(f"processing month {month:02d}")

        ind_month = ds_era5.time.dt.month.isin(month)
        data_month = ds_era5.sel(time=ind_month)

        cdf_month = emp_cdf(data_month, "Hs")

        quants_full[ind_month, :] = cdf_month

    # Save into the original Xarray dataset
    ds_era5["hs_quants"] = (["time", "station"], quants_full)

    # Rechunk
    ds_era5["hs_quants"] = ds_era5["hs_quants"].chunk({"station": 1, "time": 52560})
    ds_era5["Hs"] = ds_era5["Hs"].chunk({"station": 1, "time": 52560})
    ds_era5["Dm"] = ds_era5["Dm"].chunk({"station": 1, "time": 52560})
    ds_era5["Tm"] = ds_era5["Tm"].chunk({"station": 1, "time": 52560})

    # output_yearly(
    #     ds_era5,
    #     os.path.join(
    #         dir_out,
    #         f"LUT_output_{county}_ERA5",
    #         "netcdf")
    #     ,
    #     f"LUT_output_{county}" + "_{year}.nc"
    # )

    if not os.path.exists(os.path.join(dir_out, f"LUT_output_{county}_ERA5", "netcdf")):
        os.makedirs(os.path.join(dir_out, f"LUT_output_{county}_ERA5", "netcdf"))

    ds_era5.to_netcdf(
        os.path.join(
            dir_out, f"LUT_output_{county}_ERA5", "netcdf", f"LUT_output_{county}.nc"
        ),
        engine="netcdf4",
    )

print("Finished")
