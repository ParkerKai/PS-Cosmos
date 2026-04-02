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

#===============================================================================
# %% Import Modules
#===============================================================================
import os
from glob import glob
import numpy as np
import xarray as xr
import dask.distributed
from datetime import datetime

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'Y:\WFLOW\20240419_diff'
dir_out = r'Y:\WFLOW\20240419_discharge_wflow_CMIP6_Combined'

# model grid to process (county)
cnty = 'king'

#===============================================================================
# %% Define some functions
#===============================================================================

def output_yearly(data,dir_out,fname):
    
    year_out = np.unique(data.time.dt.year)
    
    for year in year_out:
        
        print(f'Outputting {year} Chunk')

        out = data.isel(time=data.time.dt.year.isin(year))
        
        out.to_netcdf(os.path.join(dir_out,fname.format(year=year)),engine = 'netcdf4')
 

def main():    
        
    cl= dask.distributed.LocalCluster()
    client=dask.distributed.Client(cl)
    
    print(client.dashboard_link)
    

    #===============================================================================
    # %% Calculate correction for ERA5 based on CMIP6 projections
    #===============================================================================    
    
    #===============================================================================
    # %% Load the diff data
    #===============================================================================
    print(f'loading ERA5 Data {cnty}')
    
    files = glob(os.path.join(dir_in,cnty,'*.nc'))
    ds = xr.open_mfdataset(files, engine='h5netcdf', parallel=True)
    
    #===============================================================================
    # %% Compute the CMIP6 applied Hs and save
    #===============================================================================
    
    # Comput new Hs
    Q = (ds['Q'] + ds['cmip_diff'].mean(dim='cmip6')).compute()
    
    Q.attrs = {
        'Long_name':"Dicharge w/ CMIP6 shifting",
        'Description': 'Discharge from ERA5 period shifted with Ensemble average CMIP6 derived quantile corrections',
        'units'     : 'meters^3'}
    
    ds_out = Q
    
    ds_out.attrs = {
        'Author':"KaiParker:kaparker@usgs.gov",
        'CreateDate': datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
        'Description': 'Discharge determined from WFLOW and ERA5 forcing. HQ modified by CMIP6 ensemble predicted changes to quantiles'}
    
    
    #===============================================================================
    # %% Output netcdf
    #===============================================================================
    
    output_yearly(ds_out,os.path.join(dir_out,cnty),'ERA5_Wflow_{year}_Cmip6Applied.nc')

            
    client.shutdown()

if __name__ == '__main__':
    main()