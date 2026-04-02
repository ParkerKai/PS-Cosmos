# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:50:56 2024

This script loads in the calculates the cdf for each cmip6 model and then applies 
to the ERA5 period
 
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
import pickle
import scipy 

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in_era5 = r'D:\Kai\DFM\ERA5'
dir_out = r'D:\Kai\DFM\ERA5'

# Model to process
Mod_list = ['CNRM-CM6-1-HR','GFDL']

#SLR_list =['000','025','050','100','150','200','300']
SLR = '000'


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
    import pandas as pd
    import dask 
    
    # data: Xarray dataset
    # var: variable to calculate the cdf for 
    
    # Load the xarray data into memory
    cdf  = []
    
    for stat in range(data.dims['station']):   #data.dims['station']
        print('processing Station: {}'.format(stat))
        # pull data at this station
        vals = dask.delayed(data[var].isel(station=stat).values)
        
        # Calculate the cdf 
        # cdf.append(emp_cdf_xr(vals,stat))
        cdf = 
        
    delayed_results = dask.delayed(pd.concat)(cdf)
    if stat == 0:
        delayed_results.visualize(filename=os.path.join(dir_out,'TaskGraph.svg'), optimize_graph=True)
    
    out = dask.compute(delayed_results)
    
    return out 


def monthly_CDF(data,var,month):
    data_month = data.sel(time=data.time.dt.month.isin(month)) 
    
    cdf_month = emp_cdf(data_month,var)

    return cdf_month 


#@dask.delayed()
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

def main():    
        
    cl= dask.distributed.LocalCluster()
    client=dask.distributed.Client(cl)
    
    print(client.dashboard_link)
    
    
    #===============================================================================
    # %% Load the ERA5 data and calc quantiles
    #===============================================================================
    print('loading ERA5 Data')    
    files = glob(os.path.join(dir_in_era5,'*.nc'))
    ds_era5 = xr.open_mfdataset(files, engine='h5netcdf', parallel=True,
                                 chunks = {'station':1, 'time':52560})
    
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
        
        cdf_month = emp_cdf(data_month,'waterlevel')[0]

            
        # Calc CDF
        #quants = dask.array.from_array(np.full(shape=data_month['waterlevel'].shape, fill_value=np.nan, dtype = 'float64'),
        #                               chunks=(data_month['waterlevel'].shape[0],1))
        quants = np.full(shape=data_month['waterlevel'].shape, fill_value=np.nan, dtype = 'float64')
        for stat in range(data_month.dims['station']):   # data_month.dims['station']
            print(f'processing station {stat:04d}')

            # pull data at this station                      
            era5_cdf = cdf_month.loc[cdf_month['stat'] == stat]
              
            # out_type is 'values' or 'cdf'
            # Subset to station
            
            # Break into values and quantiles
            era5_cdf_vals = era5_cdf['values'].to_numpy()
            era5_cdf_quants = era5_cdf['cdf'].to_numpy()
            vals_era5     = (data_month['waterlevel'].isel(station=stat).values)
        
            quants[: ,stat] = interp2quant(era5_cdf_vals,era5_cdf_quants,vals_era5)
        
            # Determine CDF based on the pre-calculated ERA5 cdf 
            # interp_era5 = dask.delayed(scipy.interpolate.interp1d(vals_era5_cdf['values'],
            #                                          vals_era5_cdf['cdf'],
            #                                          fill_value = (0,1),
            #                                          copy=False,
            #                                          assume_sorted=True,
            #                                          bounds_error=False))
            
            # quants[:,stat] = dask.compute(interp_era5(vals_era5['waterlevel']))
       
        # Add this month chunk into the full set
        #delayed_results = dask.delayed(pd.concat)(quants,axis=1)
        #quants_full[ind_month,0:10] = delayed_results.compute()
        quants_full[ind_month,:] = quants
    
    # Save into the original Xarray dataset
    ds_era5["wl_quants"]=(['time', 'station'],  quants_full) 
    ds_era5['wl_quants'] = ds_era5['wl_quants'].chunk({'station':1, 'time':52560})
        
    ds_era5.to_netcdf(os.path.join(dir_out,'ERA5_wl_Full.nc'),engine='h5netcdf')
            
    client.shutdown()
    
if __name__ == '__main__':
    main()