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
import pickle
import scipy 
import matplotlib 
import pandas as pd 

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'C:\Users\kai\Documents\KaiRuns\DFM\cdf_diff_combined'
#dir_out = r'D:\Kai\DFM\ERA5_CMIP6'


# Station
Stat = 562

#SLR_list =['000','025','050','100','150','200','300']
SLR = '050'

year = 1943


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

#===============================================================================
# %% Load the ERA5 data and calc quantiles
#===============================================================================
print('loading ERA5 Data')    
files = os.path.join(dir_in,f'ERA5wl_Diff_{year}.nc')
ds_era5 = xr.open_mfdataset(files, engine='h5netcdf', parallel=True)
ds_era5 = ds_era5.isel(station=Stat)

# Unscale water lelvel and cmip_diff
scale_factor = 100000000

ds_era5['waterlevel'] = ds_era5['waterlevel']/scale_factor
ds_era5['cmip_diff'] = ds_era5['cmip_diff']/scale_factor 


#===============================================================================
# %% Plots
#===============================================================================    

fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 6)

l1 = ax.plot(ds_era5['time'], ds_era5['waterlevel'],color = 'k',
              label = 'WaterLevel')

for ii in range(ds_era5['cmip6'].size):
    
    ax.plot(ds_era5['time'], (ds_era5['waterlevel']+ds_era5['cmip_diff'].isel(cmip6=ii)-np.asarray(SLR, dtype='int32')/100))


ax.set_xlim(pd.Timestamp(f'{year}-10-01'),pd.Timestamp(f'{year}-10-14'))
#ax.set_xlim(100,200)

ax.legend(['ERA5','CMCC','CNRM','EcEarth','GFDL','HadGemHH','HadGemHM','HadGemHMsst'])


ax.grid()
ax.set_title('Water Level Comparison ')
ax.set_ylabel('WL (m)')
ax.set_xlabel('Date')


##############################################################################


fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 6)

l2 = ax.plot(ds_era5['time'],(ds_era5['waterlevel']+ds_era5['cmip_diff'].mean(dim='cmip6')-np.asarray(SLR, dtype='int32')/100),
                     color = 'b')

l = ax.fill_between(ds_era5['time'],(ds_era5['waterlevel']+ds_era5['cmip_diff'].max(dim='cmip6')-np.asarray(SLR, dtype='int32')/100),
                     (ds_era5['waterlevel']+ds_era5['cmip_diff'].min(dim='cmip6')-np.asarray(SLR, dtype='int32')/100),
                     color = 'b',alpha = 0.8)


l1 = ax.plot(ds_era5['time'], ds_era5['waterlevel'],color = 'k',
              label = 'WaterLevel')

ax.set_xlim(pd.Timestamp(f'{year}-10-01'),pd.Timestamp(f'{year}-10-14'))
#ax.set_xlim(100,200)
ax.grid()
ax.set_title('Water Level ')
ax.set_ylabel('WL (m)')
ax.set_xlabel('Date')

##############################################################################






