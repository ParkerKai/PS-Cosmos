# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 15:19:18 2024

This script loads in the calculatee cdf for each cmip6 model and then finds the difference as applies to
to the ERA5 period. Specifically for each quantile value for the ERA5 period it finds
the different predicted from cmip6 historic to future. It does this for each month
and each CMIP6 model and then saves as a netcdf.


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
import pickle
import scipy

import dask.distributed
from dask.distributed import Client, LocalCluster
from dask.delayed import delayed
from dask import compute


# ===============================================================================
# %% Define some functions
# ===============================================================================

@dask.delayed()
def interp2quant(cdf_vals, cdf_quant, data_vals):
    # Determine CDF based on the pre-calculated ERA5 cdf
    interp_dat = scipy.interpolate.interp1d(
        cdf_vals,
        cdf_quant,
        fill_value=(0, 1),
        copy=False,
        assume_sorted=True,
        bounds_error=False,
    )

    quants = interp_dat(data_vals)

    return quants


def _station_diff_numpy(cdf_H, cdf_F, cdf_R, data_month_1d):
    """
    Pure-NumPy per-station computation:
    switched to numpy over scipy for bettter performance
    - cdf_* dicts hold columns: 'cdf','values','stat'
    - data_month_1d: 1D time series for a single station in the selected month

    """
    # Extract arrays for this station id (the caller passes already-filtered arrays)
    cdf_H_cdf, cdf_H_val = cdf_H['cdf'], cdf_H['values']
    cdf_F_cdf, cdf_F_val = cdf_F['cdf'], cdf_F['values']
    cdf_R_cdf, cdf_R_val = cdf_R['cdf'], cdf_R['values']

    if cdf_R_val.size == 0:
        # No ERA5 CDF for this station
        quant_era5 = np.full(data_month_1d.shape[0], np.nan, dtype='float32')
        diff = np.full(data_month_1d.shape[0], np.nan, dtype='float32')
        return diff, quant_era5

    # 1) Map ERA5 observed values -> quantiles in ERA5 CDF [0,1]
    quant_era5 = np.interp(
        data_month_1d,
        cdf_R_val,  # x: values
        cdf_R_cdf,  # y: quant
        left=0.0,
        right=1.0
    ).astype('float32')

    # 2) Map those quantiles to values in Future and Historic CDFs, then diff
    vals_F = np.interp(
        quant_era5,
        cdf_F_cdf, cdf_F_val,
        left=cdf_F_val.min(), right=cdf_F_val.max()
    ).astype('float32')

    vals_H = np.interp(
        quant_era5,
        cdf_H_cdf, cdf_H_val,
        left=cdf_H_val.min(), right=cdf_H_val.max()
    ).astype('float32')

    diff = (vals_F - vals_H).astype('float32')
    return diff, quant_era5


def calc_diff_numpy(cdf_H, cdf_F, cdf_R, data_month, var):
    """
    Parallel per-station evaluation.
    cdf_* are dict-like with arrays or pandas Series; we convert once here.
    """
    n_stat = data_month.dims["Q_contour_gauges_contour"]
    diff = np.full(data_month[var].shape, np.nan, dtype="float32")
    quants = np.full(data_month[var].shape, np.nan, dtype="float32")

    # Prepare delayed tasks
    tasks = []
    for stat in range(n_stat):
        # Filter CDFs to this station as *NumPy arrays* once
        cdf_H_stat = {
            'cdf': cdf_H["cdf"].loc[cdf_H["stat"] == stat].to_numpy(),
            'values': cdf_H["values"].loc[cdf_H["stat"] == stat].to_numpy()
        }
        cdf_F_stat = {
            'cdf': cdf_F["cdf"].loc[cdf_F["stat"] == stat].to_numpy(),
            'values': cdf_F["values"].loc[cdf_F["stat"] == stat].to_numpy()
        }
        cdf_R_stat = {
            'cdf': cdf_R["cdf"].loc[cdf_R["stat"] == stat].to_numpy(),
            'values': cdf_R["values"].loc[cdf_R["stat"] == stat].to_numpy()
        }

        vals_era5 = data_month[var].isel(Q_contour_gauges_contour=stat).values

        task = delayed(_station_diff_numpy)(cdf_H_stat, cdf_F_stat, cdf_R_stat, vals_era5)
        tasks.append(task)

    # Compute all stations in parallel once
    results = compute(*tasks)

    # Pack back into arrays
    for stat, (diff_stat, quants_stat) in enumerate(results):
        diff[:, stat] = diff_stat
        quants[:, stat] = quants_stat


    return diff, quants

def output_yearly(data, dir_out, fname):
    year_out = np.unique(data.time.dt.year)

    for year in year_out:
        print(f"Outputting {year} Chunk")

        out = data.isel(time=data.time.dt.year.isin(year))

        out.to_netcdf(os.path.join(dir_out, fname.format(year=year)), engine="netcdf4")


def main():
    # ===============================================================================
    # User Defined inputs
    # ===============================================================================
    # Directory where the WFLOW data resides
    # dir_in = r'D:\DFM'
    dir_in = r"C:\Users\kai\Documents\KaiDownloads\WFLOW"
    dir_out = r"C:\Users\kai\Documents\KaiDownloads\WFLOW"

    # Model to process
    Mod_list = ["CNRM", "EcEarth", "GFDL", "HadGemHH", "HadGemHM", "HadGemHMsst"]

    # model grid to process (county)
    cnty = "snohomish"

    n_workers = 10

    # -----------------------------
    # Dask cluster
    # -----------------------------
    print("starting Dask Cluster")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,  # good when using SciPy/NumPy
        processes=True,
        silence_logs=False,
    )
    client = Client(cluster)
    client.wait_for_workers(n_workers)
    print("Dashboard:", cluster.dashboard_link)
    print("Workers:", list(client.scheduler_info().get("workers", {}).keys()))

    dask.config.set({"dataframe.shuffle.method": "tasks"})

    # ===============================================================================
    # Load the ERA5 data and calc quantiles
    # ===============================================================================
    print("loading ERA5 Data")

    if (cnty == "pierce") or (cnty == "king"):
        dir_runs = "20240801_discharges" 
    
    elif (cnty == "snohomish") or (cnty == "kitsap"):
        dir_runs = "11_20_2025_Discharges_SnohomishKitsap"
    
    files = os.path.join(
        dir_in,
        dir_runs,
        cnty,
        "era5_3hourly",
        "output_scalar.nc",
    )

    ds_era5 = xr.open_mfdataset(files, engine="h5netcdf", parallel=True)
    ds_era5 = ds_era5.chunk({"time": 52560, "Q_contour_gauges_contour": 1})
    ds_era5 = ds_era5.persist()

    # ===============================================================================
    # Calculate correction for ERA5 based on CMIP6 projections
    # ===============================================================================
    ds_save1 = [i for i in range(len(Mod_list))]
    ds_save2 = [i for i in range(len(Mod_list))]
    for cnt, Mod in enumerate(Mod_list):
        print(f"Processing CMIP6 Difference for {Mod}")

        # split by month
        # Final numpy array that will be filled in month by month
        diff_full = np.full(ds_era5["Q_contour"].shape, np.nan, dtype="float32")
        quants_full = np.full(ds_era5["Q_contour"].shape, np.nan, dtype="float32")
        for month in np.arange(1, 13, 1, dtype=int):
            print(f"Processing Month {month:02d}")

            # Load the CMIP6 historic data
            with open(
                os.path.join(
                    dir_in,
                    dir_runs,
                    cnty,
                    f"cmip6_{Mod}_historic_bc",
                    "CDFmonthly_{0:02d}.pkl".format(month),
                ),
                "rb",
            ) as f:
                cdf_cmipH = pickle.load(f)

            # Load the CMIP6 future data
            with open(
                os.path.join(
                    dir_in,
                    dir_runs,
                    cnty,
                    f"cmip6_{Mod}_future_bc",
                    "CDFmonthly_{0:02d}.pkl".format(month),
                ),
                "rb",
            ) as f:
                cdf_cmipF = pickle.load(f)

            # Load the ERA5 data
            with open(
                os.path.join(
                    dir_in,
                    dir_runs,
                    cnty,
                    "era5_3hourly",
                    "CDFmonthly_{0:02d}.pkl".format(month),
                ),
                "rb",
            ) as f:
                cdf_cmipR = pickle.load(f)

            # subset ERA5 to the month
            # Index for the specific month we are processing (used to fill in Diff_full later)
            ind_month = ds_era5.time.dt.month.isin(month)
            data_month = ds_era5.isel(time=ind_month)

            
            # Calculate the difference between the historic and future.
            diff, quants = calc_diff_numpy(cdf_cmipH, cdf_cmipF, cdf_cmipR, data_month, "Q_contour")


            # Add this month chunk into the full set
            diff_full[ind_month, :] = diff
            quants_full[ind_month, :] = quants

        # Save into the original Xarray dataset
        ds_era5_diff = xr.DataArray(
            data=diff_full,  # enter data here
            dims=["time", "station"],
            coords={
                "time": ds_era5["time"],
                "station": ds_era5["Q_contour_gauges_contour"].values.astype(int),
            },
            attrs={"_FillValue": -9999, "units": "meters"},
        )

        ds_era5_quants = xr.DataArray(
            data=quants_full,  # enter data here
            dims=["time", "station"],
            coords={
                "time": ds_era5["time"],
                "station": ds_era5["Q_contour_gauges_contour"].values.astype(int),
            },
            attrs={"_FillValue": -9999, "units": "meters"},
        )

        # Save for concatenating later
        ds_save1[cnt] = ds_era5_diff
        ds_save2[cnt] = ds_era5_quants

    # Concat
    ds_diff = xr.concat(ds_save1, dim="cmip6")
    ds_diff = ds_diff.assign_coords({"cmip6": Mod_list})
    ds_diff = ds_diff.chunk({"station": 1, "time": 52560, "cmip6": 1})

    ds_quants = xr.concat(ds_save2, dim="cmip6")
    ds_quants = ds_quants.assign_coords({"cmip6": Mod_list})
    ds_quants = ds_quants.chunk({"station": 1, "time": 52560, "cmip6": 1})

    Q = xr.DataArray(
        data=ds_era5["Q_contour"].values,  # enter data here
        dims=["time", "station"],
        coords={
            "time": ds_era5["time"],
            "station": ds_era5["Q_contour_gauges_contour"].values.astype(int),
        },
        attrs={"_FillValue": -9999, "units": "meters"},
    )

    ds_full = xr.Dataset(
        {"Q": Q, "Q_quants": ds_quants, "cmip_diff": ds_diff},
        attrs={
            "DataSource": r"Y:\WFLOW",
            "ProducedBy": "Wflow team (Joost Buitink & Brendan Dalmijn) and Kai Parker",
            "General": "Extracted from WFLOW model version 20240419",
        },
    )

    output_yearly(
        ds_full,
        os.path.join(
            dir_out, dir_runs, cnty, "cdf_diff"
        ),
        "WFLOW_ERA5Diff_{year}.nc",
    )

        
    client.close()
    cluster.close()



if __name__ == "__main__":
    main()
