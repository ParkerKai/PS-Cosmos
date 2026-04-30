# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:29:35 2024

This script calculates the cdf  for each model.
CDF is calculated monthly
This can then be applied to the reanalysis period.  

For WFLOW model runs

@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"


# ===============================================================================
# ECDF helpers
# ===============================================================================

import os
import numpy as np
import xarray as xr
import pandas as pd
from dask.distributed import Client, LocalCluster
import dask


# ===============================================================================
# ECDF helpers
# ===============================================================================

@dask.delayed()
def emp_cdf_xr(arr, stat):
    """
    Compute ECDF for a 1D array (one station), returning a DataFrame with
    columns: values, cdf, stat.
    """
    try:
        from scipy import stats
        HAS_SCIPY_ECDF = hasattr(stats, "ecdf")
    except Exception:
        HAS_SCIPY_ECDF = False

    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return pd.DataFrame({"values": [], "cdf": [], "stat": []})

    if HAS_SCIPY_ECDF:
        res = stats.ecdf(arr)
        q = res.cdf.quantiles
        p = res.cdf.probabilities
    else:
        # Fallback to statsmodels ECDF; if unavailable, use numpy step ECDF.
        try:
            from statsmodels.distributions.empirical_distribution import ECDF
            x_sorted = np.sort(arr)
            ecdf = ECDF(arr)
            q = x_sorted
            p = ecdf(x_sorted)
        except Exception:
            x_sorted = np.sort(arr)
            p = np.arange(1, x_sorted.size + 1) / x_sorted.size
            q = x_sorted

    return pd.DataFrame({"values": q, "cdf": p, "stat": stat})


def emp_cdf(data, var, station_dim):
    """
    Compute ECDF per station for the given DataArray (subset as needed).
    Returns a single pandas.DataFrame concatenated across stations.
    """
    import dask

    nstations = data.dims[station_dim]
    tasks = []

    for stat in range(nstations):
        print(f'processing Station: {stat}')
        # Keep it lazy; conversion to numpy happens inside emp_cdf_xr
        vals = data[var].isel({station_dim: stat}).data
        tasks.append(emp_cdf_xr(vals, stat))

    delayed_results = dask.delayed(pd.concat)(tasks, ignore_index=True)
    out = dask.compute(delayed_results)  # returns (DataFrame,)
    return out


def monthly_CDF(data, var, station_dim, month):
    """
    Filter dataset to a month (int) or list of months, then compute per-station ECDFs.
    """
    if isinstance(month, (list, tuple, set, np.ndarray)):
        data_month = data.sel(time=data.time.dt.month.isin(month))
    else:
        data_month = data.sel(time=data.time.dt.month == int(month))
    return emp_cdf(data_month, var, station_dim)


# ===============================================================================
# Main
# ===============================================================================

def main():
    # -----------------------------
    # User-defined inputs
    # -----------------------------
    dir_in = r'C:\Users\kai\Documents\KaiDownloads\WFLOW\11_20_2025_Discharges_SnohomishKitsap'
    dir_out = r'C:\Users\kai\Documents\KaiDownloads\WFLOW\11_20_2025_Discharges_SnohomishKitsap'

    cnty = 'kitsap'
    Mod_list = ['EcEarth','GFDL','HadGemHH','HadGemHM','HadGemHMsst']   # 'CNRM',
    Per = 'future'

    var = 'Q_contour'                   # variable to process
    station_dim = 'Q_contour_gauges_contour'  # station dimension name in the file
    save_as_parquet = False             # True for Parquet; False keeps Pickle

    # Number of workers
    n = 8

    # -----------------------------
    # Dask cluster
    # -----------------------------
    print("starting Dask Cluster")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    cluster = LocalCluster(
        n_workers=n,
        threads_per_worker=1,
        processes=True,
        silence_logs=False,
    )
    client = Client(cluster)

    print("Dashboard:", cluster.dashboard_link)
    print("Workers:", client.scheduler_info().get("workers", {}).keys())

    # -----------------------------
    # Process each model
    # -----------------------------
    for Mod in Mod_list:
        print(f'Processing {Mod} for {Per} Period')
        files = os.path.join(dir_in, cnty, f'cmip6_{Mod}_{Per}_bc', 'output_scalar.nc')

        ds_cmip = xr.open_mfdataset(
            files,
            engine='h5netcdf',
            parallel=True,
            chunks={'time': 1000}
        )

        # Loop months 1..12
        for month in range(1, 13):
            print(f'Processing Month: {month:02d}')
            cdf_month = monthly_CDF(ds_cmip, var, station_dim, month)[0]  # [0] to unwrap dask.compute result

            out_dir = os.path.join(dir_out, cnty, f'cmip6_{Mod}_{Per}_bc')
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(out_dir, f'CDFmonthly_{month:02d}.{"parquet" if save_as_parquet else "pkl"}')
            if save_as_parquet:
                # Requires `pyarrow`
                cdf_month.to_parquet(out_path, index=False)
            else:
                cdf_month.to_pickle(out_path)

        # Close dataset to free resources
        ds_cmip.close()

    print('Done!')

if __name__ == '__main__':
    main()