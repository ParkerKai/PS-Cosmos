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

def interpAtQuant(cdf_vals,cdf_quant,data_quant):
    
    # Determine CDF based on the pre-calculated ERA5 cdf 
    interp_era5 = scipy.interpolate.interp1d(cdf_quant,
                                          cdf_vals,
                                          fill_value=(cdf_vals.min(),cdf_vals.max()),
                                          copy=False,
                                          assume_sorted=True,
                                          bounds_error=False)
     
    vals = interp_era5(data_quant)

    return vals 


def calc_diff(cdf_H,cdf_F,data):    
    # PUll data for the station and unwrap pandas dataframe to numpy
    cdf_H_stat_cdf = cdf_H['cdf'].to_numpy()
    cdf_H_stat_val = cdf_H['values'].to_numpy()
    cdf_F_stat_cdf = cdf_F['cdf'].to_numpy()
    cdf_F_stat_val = cdf_F['values'].to_numpy()
    quant_era5     = data['wl_quants'].values
    
    diff  = interpAtQuant(cdf_F_stat_val,cdf_F_stat_cdf,quant_era5) - interpAtQuant(cdf_H_stat_val,cdf_H_stat_cdf,quant_era5)
                
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
       
        # INitialize dataset
        ds_era5_diff = xr.DataArray(
            data = np.full([ds_era5['time'].size, ds_era5['station'].size, len(Mod_list)],-9999, dtype='int32'),
                 dims   = ['time','station','cmip6'],
                 coords = {'time': ds_era5['time'],
                           'station':ds_era5['station'],
                           'cmip6': Mod_list},
                 attrs  = {
                     'units' : 'meters/10000'
                     })
                                  
        for cnt,Mod in enumerate(Mod_list):
            
            print(f'Processing CMIP6 Difference for {Mod} {SLR}',flush=True)
            
            # split by month
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
                ds_era5_diff[ind_month,:,cnt] = diff
                
        
        # ds_full =  xr.Dataset({
        #     'wl':ds_era5['waterlevel'],
        #     'wl_quant':ds_era5['wl_quants'],
        #     'cmip_diff':ds_diff,
        #     'lon':ds_era5['lon'],
        #     'Lat':ds_era5['lat']})
        
        
        ds_full =  xr.Dataset({'cmip_diff':ds_era5_diff})
        ds_full = ds_full.chunk({'cmip6':1,'time':-1, 'station':1})
        
        #output_yearly(ds_diff,dir_out,'ERA5_{year}_Diff.nc')

        
        ds_full.to_netcdf(os.path.join(dir_out,f'ERA5_{year}_Diff.nc'), engine = 'netcdf4',
                          encoding={'cmip_diff':{'dtype': 'int32'}})

                
    client.shutdown()

if __name__ == '__main__':
    main()