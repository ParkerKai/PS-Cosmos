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


# ------------------------------------------------------------------------------
# Import libraries
# ------------------------------------------------------------------------------

import os
import numpy as np
import xarray as xr
import pandas as pd
from dask.distributed import Client, LocalCluster
import dask

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------

def _ensure_numpy(arr):
    """Return a NumPy array from a NumPy or Dask array without loading early."""
    # If this is a Dask collection, compute here (inside the worker)
    if hasattr(arr, "compute"):
        arr = arr.compute()
    return np.asarray(arr)

@dask.delayed  # <- no parentheses
def emp_cdf_xr(values, stat):
    """
    Compute empirical CDF for a 1D array of values for a single station.

    Returns a Pandas DataFrame with columns: values, cdf, stat.
    """
    # Convert to NumPy on the worker (if dask array)
    data = _ensure_numpy(values)

    # Drop NaNs
    data = data[~np.isnan(data)]
    if data.size == 0:
        return pd.DataFrame({"values": [], "cdf": [], "stat": []})

    # SciPy ECDF (SciPy >= 1.11). If not available, fallback to statsmodels.
    try:
        import scipy
        from scipy import stats
        # sanity check version
        try:
            from packaging.version import parse as V
            if V(scipy.__version__) < V("1.11"):
                raise ImportError("scipy.stats.ecdf requires SciPy >= 1.11")
        except Exception:
            # If packaging not installed, just attempt and let ImportError bubble
            pass

        res = stats.ecdf(data)  # returns ECDFResult with cdf/sf
        df = pd.DataFrame(
            {
                "values": res.cdf.quantiles,
                "cdf": res.cdf.probabilities,
                "stat": np.full(len(res.cdf.quantiles), stat, dtype=int),
            }
        )
        return df
    except Exception:
        # Fallback: statsmodels ECDF for older SciPy environments
        from statsmodels.distributions.empirical_distribution import ECDF
        ecdf = ECDF(data)
        # ECDF exposes step locations via ecdf.x and ECDF values via ecdf.y
        df = pd.DataFrame(
            {
                "values": ecdf.x,
                "cdf": ecdf.y,
                "stat": np.full(len(ecdf.x), stat, dtype=int),
            }
        )
        return df

def emp_cdf(ds, var, station_dim):
    """
    Compute ECDF for each station in `station_dim` for DataArray `var` in Dataset `ds`.

    Returns a single Pandas DataFrame (concatenated over stations).
    """
    delayed_frames = []
    nstations = ds[var].sizes[station_dim]

    for stat in range(nstations):
        print(f"processing Station: {stat}")
        # Grab the values lazily (keep as Dask-backed if chunked)
        values = ds[var].isel({station_dim: stat}).data  # don't call .values here
        delayed_frames.append(emp_cdf_xr(values, stat))

    # Concatenate results (still delayed)
    delayed_concat = dask.delayed(pd.concat)(delayed_frames, ignore_index=True)

    # Compute once (returns the concrete DataFrame)
    return dask.compute(delayed_concat)[0]

def monthly_CDF(ds, var, station_dim, month):
    """
    Filter dataset by calendar month and compute ECDFs.
    """
    # `.isin(month)` expects list-like; equality is simpler for a single month.
    ds_month = ds.sel(time=ds.time.dt.month == month)
    return emp_cdf(ds_month, var, station_dim)

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    # -----------------------------
    # User inputs
    # -----------------------------
    dir_in = r'D:\wflow\11_20_2025_Discharges_SnohomishKitsap'
    dir_out = r'D:\wflow\11_20_2025_Discharges_SnohomishKitsap'
    cnty = 'snohomish'
    var = 'Q_contour'
    station_dim = 'Q_contour_gauges_contour'  # adjust if your variable uses a different name
    n_workers = 6

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

    # -----------------------------
    # Load data (single file)
    # -----------------------------
    fn = os.path.join(dir_in, cnty, 'era5_3hourly', 'output_scalar.nc')

    # Use open_dataset for a single file; give chunks to benefit from Dask laziness
    ds = xr.open_dataset(fn, engine='h5netcdf', chunks={'time': 1000})

    # -----------------------------
    # Compute monthly ECDFs
    # -----------------------------
    os.makedirs(os.path.join(dir_out, cnty, 'era5_3hourly'), exist_ok=True)

    for month in range(1, 13):
        print(f'Processing Month: {month:02d}')
        cdf_month = monthly_CDF(ds, var, station_dim, month)
        out_fn = os.path.join(dir_out, cnty, 'era5_3hourly', f'CDFmonthly_{month:02d}.pkl')
        cdf_month.to_pickle(out_fn)
        print(f'Wrote: {out_fn}')

if __name__ == '__main__':
    main()