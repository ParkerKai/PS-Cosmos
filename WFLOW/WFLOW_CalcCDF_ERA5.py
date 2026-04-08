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

#===============================================================================
# Import Modules
#===============================================================================
import os
import numpy as np
import xarray as xr
import pandas as pd
from dask.distributed import Client, LocalCluster
import dask

#===============================================================================
# Define some functions
#===============================================================================

@dask.delayed()
def emp_cdf_xr(data,stat):
    import scipy
    
    data = data[~np.isnan(data)]

    # Calculate the ecdf
    res = scipy.stats.ecdf(data)
    
    data_out = pd.DataFrame(data = {'values': res.cdf.quantiles,
                               'cdf': res.cdf.probabilities,
                               'stat': stat*np.ones((len(res.cdf.quantiles),), dtype=int)})
    
    return data_out

def emp_cdf(data,var):
    import pandas as pd
    import dask 
    
    # data: Xarray dataset
    # var: variable to calculate the cdf for 
    
    # Load the xarray data into memory
    cdf  = []
    
    for stat in range(data.dims['Q_contour_gauges_contour']):   
        print('processing Station: {}'.format(stat))
        # pull data at this station
        vals = dask.delayed(data[var].isel(Q_contour_gauges_contour=stat).values)
        
        # Calculate the cdf 
        
        cdf.append(emp_cdf_xr(vals,stat))
        
        
    delayed_results = dask.delayed(pd.concat)(cdf)
    # if stat == 0:
    #     delayed_results.visualize(filename=os.path.join(dir_out,'TaskGraph.svg'), optimize_graph=True)
    
    out = dask.compute(delayed_results)
    
    return out 


def monthly_CDF(data,var,month):
    data_month = data.sel(time=data.time.dt.month.isin(month)) 
    
    cdf_month = emp_cdf(data_month,var)

    return cdf_month 


def main():    
   

    #===============================================================================
    #  User Defined inputs
    #===============================================================================
    
    # Directory where the WFLOW data resides
    #dir_in = r'D:\DFM'
    dir_in = r'Y:\PS_Cosmos\02_models\WFLOW\11_20_2025_Discharges_SnohomishKitsap'
    dir_out = r'Y:\PS_Cosmos\02_models\WFLOW\11_20_2025_Discharges_SnohomishKitsap'

    # model grid to process (county)
    cnty = 'kitsap'

    # Number of workers
    n = 6

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

    #===============================================================================
    # Load the data 
    #===============================================================================
    # ERA5 Forced data 
    files = os.path.join(dir_in,cnty,'era5_3hourly','output_scalar.nc')

    ds_cmip = xr.open_mfdataset(files, engine='h5netcdf', parallel=True)

    # split by month
    for month in np.arange(1, 13, 1, dtype=int):
        print('Processing Month: {}'.format(month))
        
        cdf_month = monthly_CDF(ds_cmip,'Q_contour',month)[0]
        
        cdf_month.to_pickle(os.path.join(dir_out,cnty,'era5_3hourly','CDFmonthly_{0:02d}.pkl'.format(month)))

if __name__ == '__main__':
    main()