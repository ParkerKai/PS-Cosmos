# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:50:56 2024

This script loads in the calculates the cdf for the ERA5 model and then saves 
as a single netcdf
 
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
import dask
import scipy 

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in_era5 = r'D:\Kai\DFM\ERA5'
dir_out = r'D:\Kai\DFM\ERA5\cdf'

#===============================================================================
# %% Define some functions
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
    import dask 
    
    # data: Xarray dataset
    # var: variable to calculate the cdf for 
    
    # Load the xarray data into memory
    #quants = np.full(shape=data['waterlevel'].shape, fill_value=np.nan, dtype = 'float64')
    quants = np.full(shape=data[var].shape, fill_value=np.nan, dtype = 'float64')
    for stat in range(data.dims['station']):   #data.dims['station']
        print('processing Station: {}'.format(stat))
        # pull data at this station
        vals_era5     = dask.delayed(data[var].isel(station=stat).values)

        # Calculate the cdf 
        # cdf.append(emp_cdf_xr(vals,stat))
        cdf = emp_cdf_xr(vals_era5,stat)
        
        
        # Break into values and quantiles
        era5_cdf_vals = dask.delayed(cdf['values'].to_numpy())
        era5_cdf_quants = dask.delayed(cdf['cdf'].to_numpy())
        
        # Check if cdf is empty (no data for this station)
        if era5_cdf_vals.size.compute() == 0:
            print('No Data Station')
        else:            
            #quants[: ,stat] = interp2quant(era5_cdf_vals,era5_cdf_quants,vals_era5).compute()
            delayedResult = (interp2quant(era5_cdf_vals,era5_cdf_quants,vals_era5))
            quants[:,stat] = delayedResult.compute()
        
    out = quants
        
    return out 

@dask.delayed()
def interp2quant(cdf_vals,cdf_quant,data_vals):
    
    # Determine CDF based on the pre-calculated ERA5 cdf 
    interp_era5 = scipy.interpolate.interp1d(cdf_vals,
                                          cdf_quant,
                                          fill_value = (0,1),
                                          copy=False,
                                          assume_sorted=True,
                                          bounds_error=False)
     
    quants = interp_era5(data_vals)
    

    return quants 


def output_yearly(data,dir_out,fname):

    year_out = np.unique(data.time.dt.year)

    for year in year_out:

        print(f'Outputting {year} Chunk')
        out = data.isel(time=data.time.dt.year.isin(year))
        out.to_netcdf(os.path.join(dir_out,fname.format(year=year)),engine='netcdf4')


# def main():    
        
#     cl= dask.distributed.LocalCluster()
#     client=dask.distributed.Client(cl)
    
#     print(client.dashboard_link)
 
 
#===============================================================================
# %% Load the ERA5 data and calc quantiles
#===============================================================================
print('loading ERA5 Data')    
files = glob(os.path.join(dir_in_era5,'*.nc'))
files = files[8:]
ds_era5 = xr.open_mfdataset(files, engine='h5netcdf', parallel=True)
 
# ds_era5 = ds_era5.sel(time=slice("1980-01-01", "2024-01-01"))
ds_era5 = ds_era5.chunk(chunks = {'station':1, 'time':52560})

# Drop un-necssary dimentions
ds_era5['lat'] = ds_era5['lat'].isel(time=0, drop=True)
ds_era5['lon'] = ds_era5['lon'].isel(time=0, drop=True)
ds_era5['bedlevel'] = ds_era5['bedlevel'].isel(time=0, drop=True)

print('Calculating CDF')
# split by month for correction
# Final numpy array that will be filled in month by month
quants_full = np.full(ds_era5['waterlevel'].shape, np.nan, dtype = 'float32')
for month in np.arange(1, 13, 1, dtype=int):
    
    print(f'processing month {month:02d}')
    # Index for the specific month we are processing
    ind_month =  ds_era5.time.dt.month.isin(month)
    data_month = ds_era5.isel(time=ind_month)
    
    cdf_month = emp_cdf(data_month,'waterlevel') 
    quants_full[ind_month,:] = cdf_month
 
 
# Save into the original Xarray dataset
ds_era5["wl_quants"]=(['time', 'station'],  quants_full) 
ds_era5['wl_quants'] = ds_era5['wl_quants'].chunk({'station':1, 'time':52560})
 
output_yearly(ds_era5,dir_out,'ERA5_cdf_{year}.nc')
         
    # client.shutdown()
    
# if __name__ == '__main__':
#     main()