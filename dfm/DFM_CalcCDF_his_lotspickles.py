
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:29:35 2024

This script calculates the cdf  for each model.
CDF is calculated monthly
This can then be applied to the reanalysis period.  


@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"

#===============================================================================
# %% Import Modules
#===============================================================================
import os
from glob import glob
import numpy as np
import xarray as xr
import pandas as pd
import dask.distributed

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'C:\Users\kai\Documents\KaiRuns\DFM'
dir_out = r'C:\Users\kai\Documents\KaiRuns\DFM'

# Period
Per = 'historic'

# Model to process
Mod = 'CNRM-CM6-1-HR'

#===============================================================================
# %% Define some functions
#===============================================================================

@dask.delayed()
def emp_cdf(data):
    import scipy
    
    data = data[~np.isnan(data)]

    # Calculate the ecdf
    res = scipy.stats.ecdf(data)
    
    data_out = pd.DataFrame(data = {'values': res.cdf.quantiles,
                               'cdf': res.cdf.probabilities})
    
    return data_out    

def cdf_month_save(data,var,month):
        
    # data: Xarray dataset
    # var: variable to calculate the cdf for 
    
    # output directory
    out_dir = os.path.join(dir_out,Mod,Per,'Results_Combined','CDF')
    
    # Create CDF directory if doesn't exist 
    if not os.path.exists(out_dir):
        # Create the directory
        os.makedirs(out_dir)
    
    # Save easch station as a pickle 
    for stat in range(10):   #data.dims['station']
        print('processing Station: {}'.format(stat))
        # pull data at this station
        vals = dask.delayed(data[var].isel(station=stat).values)
        
        # Calculate the cdf 
        cdf = emp_cdf(vals)
        
        file_out = os.path.join(out_dir,f'CDF_Mon{month:02d}_Stat{stat:04d}.pkl')
        dask.compute(cdf.to_pickle(file_out))
    
def monthly_CDF(data,var,month):
    data_month = data.sel(time=data.time.dt.month.isin(month)) 
    
    cdf_month_save(data_month,var,month)


def main():    
        
    cl= dask.distributed.LocalCluster()
    client=dask.distributed.Client(cl)
    
    print(client.dashboard_link)
    
    #===============================================================================
    # %% Load the data 
    #===============================================================================
    # CMIP6 His
    print('Processing {} for {} Period'.format(Mod,Per))
    files = glob(os.path.join(dir_in,Mod,Per,'Results_Combined','*.nc'))
    ds_cmip = xr.open_mfdataset(files, engine='h5netcdf', parallel=True,
                                 chunks = {'station':1, 'time':52560})
    
    
    # Subset the historic run to have equal timeperiods
    ds_cmip = ds_cmip.sel(time=slice("1980-01-01", "2014-01-01"))
    
    # split by month
    for month in np.arange(1, 13, 1, dtype=int):
        print('Processing Month: {}'.format(month))
        
        monthly_CDF(ds_cmip,'waterlevel',month)
        

if __name__ == '__main__':
    main()