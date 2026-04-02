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
from dask.distributed import Client
import pickle
import scipy 
import traceback

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'D:\DFM_Regional\CMIP6_Results'
dir_in_era5 = r'D:\DFM_Regional\ERA5'
dir_out = r'Y:\PS_Cosmos\02_models\DFM_Regional\cdf_diff\localTest'

# Model to process
Mod_list = ['CMCC-CM2-VHR4','CNRM-CM6-1-HR','GFDL','EC-Earth_HR','HadGEM_GC31_HH','HadGEM_GC31_HM_highRes','HadGEM_GC31_HM_highResSST']

#SLR_list =['000','025','050','100','150','200','300']
#SLR = '000'


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

def calc_diff(cdf_H,cdf_F,data_month):
    # Calc CDF correction.
    diff = np.full(data_month['wl_quants'].shape, -9999, dtype = 'int32')  # data_month['wl_quants'].shape
    
    for stat in range(data_month.dims['station']):   #data.dims['station']
        #print(f'processing Station: {stat}')
                        
        # PUll data for the station and unwrap pandas dataframe to numpy
        cdf_H_stat_cdf = cdf_H['cdf'].loc[cdf_H['stat'] == stat].to_numpy()
        cdf_H_stat_val = cdf_H['values'].loc[cdf_H['stat'] == stat].to_numpy()
        cdf_F_stat_cdf = cdf_F['cdf'].loc[cdf_F['stat'] == stat].to_numpy()
        cdf_F_stat_val = cdf_F['values'].loc[cdf_F['stat'] == stat].to_numpy()
        quant_era5     = data_month['wl_quants'].isel(station=stat).values
        
        if (quant_era5.size == 0) or (cdf_H_stat_cdf.size == 0) or (cdf_F_stat_cdf.size == 0):
            print(f'No Data for Station {stat}')

        else:
            diff_stat = interpAtQuant(cdf_F_stat_val,cdf_F_stat_cdf,quant_era5) - interpAtQuant(cdf_H_stat_val,cdf_H_stat_cdf,quant_era5)
            diff_stat  = np.nan_to_num(diff_stat,nan=-9999, posinf=-9999, neginf=-9999)
            diff_stat = np.round(diff_stat).astype(dtype='int32')
            diff[:,stat] = diff_stat
        
    return diff



SLR =  '100'
year = 1941

#===============================================================================
# %% Load the ERA5 data and calc quantiles
#===============================================================================
print('loading ERA5 Data')    
#files = glob(os.path.join(dir_in_era5,'ERA5_cdf*'))
files = os.path.join(dir_in_era5,f'ERA5_{SLR}',f'ERA5_cdf_{year}.nc')

ds_era5 = xr.open_mfdataset(files, engine='netcdf4', parallel=True,
                              chunks = {'station':1, 'time':-1})

    
#===============================================================================
# %% Calculate correction for ERA5 based on CMIP6 projections
#===============================================================================   

# split by month
for month in np.arange(9, 13, 1, dtype=int):
    print(f'Processing Month {month:02d}')

    # subset ERA5 to the month
    # Index for the specific month we are processing (used to fill in Diff_full later)
    ind_month =  ds_era5.time.dt.month.isin(month)
    data_month = ds_era5.isel(time=ind_month)

    diff = np.full([data_month['time'].size, data_month['station'].size, len(Mod_list)],-9999, dtype='int32')
    for cnt,Mod in enumerate(Mod_list):
        print(f'Processing CMIP6 Difference for {Mod} {SLR}',flush=True)

         # Load the CMIP6 historic data
        with open(os.path.join(dir_in,Mod,'historic',
                               'CDFmonthly_{0:02d}.pkl'.format(month)), 'rb') as f:
            cdf_cmipH = pickle.load(f)

         # Load the CMIP6 future data
        with open(os.path.join(dir_in,Mod,'future',SLR,
                               'CDFmonthly_{0:02d}.pkl'.format(month)), 'rb') as f:
            cdf_cmipF = pickle.load(f)


        # Calculate the difference between the historic and future.
        diff[:,:,cnt] = calc_diff(cdf_cmipH,cdf_cmipF,data_month)
        
        if np.sum(diff[:,:,cnt] <  -200000000) >0:
            print('Issue with diff calc')
            asdf
    ds_diff = xr.DataArray(
        data = diff,
             dims   = ['time','station','cmip6'],
             coords = {'time': data_month['time'],
                       'station':data_month['station'],
                       'cmip6': Mod_list},
             attrs  = {'long_name':'Cmip6 Difference in water levels',
                       'Date': f'year {year}, month {month}',
                       'units' : 'meters/10000'})

    ds_diff = ds_diff.chunk({'cmip6':1,'time':-1, 'station':1})

    # Post process some variables for output
    #wl_out = data_month['waterlevel'].fillna(-9999)
    #wl_out = wl_out.round(0).astype(dtype='int32')
    #wl_out = wl_out.chunk({'time':-1, 'station':1})
    station_id = data_month['station'].to_numpy().astype('unicode')

    #ds_full =  xr.Dataset({
    #    'wl':wl_out,
    #    'wl_quant':data_month['wl_quants'],
    #    'cmip_diff':ds_diff,
    #    'lon':data_month['lon'],
    #    'Lat':data_month['lat']})
    
    ds_full =  xr.Dataset({'cmip_diff':ds_diff})
    ds_full = ds_full.assign_coords({'station':station_id})

    ds_full.to_netcdf(os.path.join(dir_out,SLR,f'ERA5wl_Diff_{year}_{month:02d}.nc'), engine = 'netcdf4',
                          encoding={'cmip_diff':{'dtype': 'int32'}})

    #ds_full.to_netcdf(os.path.join(dir_out,SLR,f'ERA5wl_Diff_{year}_{month:02d}.nc'), engine = 'netcdf4',
    #                  encoding={'wl':{'dtype': 'int32'},
    #                            'wl_quant':{'dtype':'float32'},
    #                            'cmip_diff':{'dtype': 'int32'},
    #                            'Lon':{'dtype': 'float64'},
    #                            'Lat':{'dtype':'float64'}})


