# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:50:56 2024

This script loads in the calculatee cdf for each cmip6 model and then applies 
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
import pandas as pd
import dask.distributed
import pickle
import scipy 

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'C:\Users\kai\Documents\KaiRuns\DFM'
dir_in_era5 = r'D:\Kai\DFM\ERA5'
dir_out = r'D:\Kai\DFM\ERA5_CMIP6'

# Model to process
Mod_list = ['CMCC-CM2-VHR4','CNRM-CM6-1-HR','EC-Earth_HR','GFDL','HadGEM_GC31_HH','HadGEM_GC31_HM_highRes','HadGEM_GC31_HM_highResSST']

#SLR_list =['000','025','050','100','150','200','300']
SLR = '000'


#===============================================================================
# %% Define some functions
#===============================================================================

@dask.delayed()
def interp2quant(cdf_vals,cdf_quant,data_vals):
    
    # Determine CDF based on the pre-calculated ERA5 cdf 
    interp_era5 = scipy.interpolate.interp1d(cdf_vals,
                                          cdf_quant,
                                          fill_value=(cdf_vals.min(),cdf_vals.max()),
                                          copy=False,
                                          assume_sorted=True,
                                          bounds_error=False)
     
    quants = interp_era5(data_vals)

    return quants 

def calc_diff(cdf_H,cdf_F,data_month):
    # Calc CDF correction.
    diff = np.full(data_month['wl_quants'].shape, -9999, dtype = 'int32')  # data_month['wl_quants'].shape
    
        
    for stat in range(data_month.dims['station']):   #data_month.dims['station']
        print(f'processing Station: {stat}')
                        
        # PUll data for the station and unwrap pandas dataframe to numpy
        cdf_H_stat_cdf = dask.delayed(cdf_H['cdf'].loc[cdf_H['stat'] == stat].to_numpy())
        cdf_H_stat_val = dask.delayed(cdf_H['values'].loc[cdf_H['stat'] == stat].to_numpy())
        cdf_F_stat_cdf = dask.delayed(cdf_F['cdf'].loc[cdf_F['stat'] == stat].to_numpy())
        cdf_F_stat_val = dask.delayed(cdf_F['values'].loc[cdf_F['stat'] == stat].to_numpy())
        quant_era5     = dask.delayed(data_month['wl_quants'].isel(station=stat).values)
        
        if (quant_era5.size.compute() == 0) or (cdf_H_stat_cdf.size.compute() == 0) or (cdf_F_stat_cdf.size.compute() == 0):
            print('No Data Station')

        else:
            delayedResult = interp2quant(cdf_F_stat_val,cdf_F_stat_cdf,quant_era5) - interp2quant(cdf_H_stat_val,cdf_H_stat_cdf,quant_era5)
            delayedResult = np.round(delayedResult).astype(dtype='int32')
            diff[:,stat] = delayedResult.compute()
        
    return diff


def output_yearly(data,dir_out,fname):
    
    year_out = np.unique(data.time.dt.year)
    
    for year in year_out:
        
        print(f'Outputting {year} Chunk')

        out = data.isel(time=data.time.dt.year.isin(year))
        
        out.to_netcdf(os.path.join(dir_out,fname.format(year=year)),engine = 'h5netcdf')
 

def main():    
        
    cl= dask.distributed.LocalCluster()
    client=dask.distributed.Client(cl)
    
    print(client.dashboard_link)
    
    #===============================================================================
    # %% Load the ERA5 data and calc quantiles
    #===============================================================================
    print('loading ERA5 Data')    
    files = glob(os.path.join(dir_in_era5,'cdf','ERA5_cdf*'))
    
    for file in files:
        
        ds_era5 = xr.open_mfdataset(file, engine='h5netcdf', parallel=True,
                                      chunks = {'station':1, 'time':52560})
        
        year = np.unique(ds_era5.time.dt.year)[0]
        print(f'Processing Year {year}')
        
        #===============================================================================
        # %% Calculate correction for ERA5 based on CMIP6 projections
        #===============================================================================    
        ds_save  = [i for i in range(len(Mod_list))]
        for cnt,Mod in enumerate(Mod_list):
            
            print(f'Processing CMIP6 Difference for {Mod}')
        
            # split by month
            # Final numpy array that will be filled in month by month
            diff_full = np.full(ds_era5['waterlevel'].shape, -9999, dtype = 'int32')
            
            for month in np.arange(1, 13, 1, dtype=int):
                print(f'Processing Month {month:02d}')
                
                 # Load the CMIP6 historic data
                with open(os.path.join(dir_in,'CMIP6',Mod,'historic','Results_Combined',
                                       'CDFmonthly_{0:02d}.pkl'.format(month)), 'rb') as f: 
                    cdf_cmipH = pickle.load(f)
        
                 # Load the CMIP6 future data
                with open(os.path.join(dir_in,'CMIP6',Mod,'future','Results_Combined',SLR,
                                       'CDFmonthly_{0:02d}.pkl'.format(month)), 'rb') as f: 
                    cdf_cmipF = pickle.load(f)
                
                # subset ERA5 to the month
                # Index for the specific month we are processing (used to fill in Diff_full later)
                ind_month =  ds_era5.time.dt.month.isin(month)
                data_month = ds_era5.isel(time=ind_month) 
                
                # Calculate the difference between the historic and future. 
                diff = calc_diff(cdf_cmipH,cdf_cmipF,data_month)    
                        
                # Add this month chunk into the full set
                diff_full[ind_month,:] = diff
                
            # Save into the original Xarray dataset
            ds_era5_diff = xr.DataArray(
                     data   = diff_full,    # enter data here
                     dims   = ['time','station'],
                     coords = {'time': ds_era5['time'],
                               'station':ds_era5['station']},
                     attrs  = {
                         '_FillValue': -9999,
                         'units'     : 'meters'
                         })
            
            ds_era5_diff = ds_era5_diff.chunk({'station':1, 'time':52560})
            # Save for concatenating later
            ds_save[cnt] = ds_era5_diff
        
        # Concat
        ds_diff = xr.concat(ds_save, dim='cmip6')
        ds_diff = ds_diff.assign_coords({'cmip6': Mod_list})
        ds_diff = ds_diff.chunk({'station':1, 'time':4380, 'cmip6':1})
        
        ds_full =  xr.Dataset({
            'wl':ds_era5['waterlevel'],
            'wl_quant':ds_era5['wl_quants'],
            'cmip_diff':ds_diff,
            'lon':ds_era5['lon'],
            'Lat':ds_era5['lat']})
        
        
        ds_full =  xr.Dataset({'cmip_diff':ds_diff})
        ds_full = ds_full.chunk({'cmip6':1,'time':-1, 'station':1})
        
        #output_yearly(ds_diff,dir_out,'ERA5_{year}_Diff.nc')

        
        ds_full.to_netcdf(os.path.join(dir_out,f'ERA5_{year}_Diff.nc'), engine = 'h5netcdf',
                          encoding={'cmip_diff':{'dtype': 'int32'}})

                
    client.shutdown()

if __name__ == '__main__':
    main()