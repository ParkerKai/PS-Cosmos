# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:00:12 2024

Created on Thu May 16 13:50:56 2024

This script Plots the monthly cdf corrections
 
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
import matplotlib.pyplot as plt 

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'Y:\PS_Cosmos\02_models\DFM_Regional'
dir_out = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\figures\CDF_Diff\DFM'

# Model to process
Mod_list = ['CNRM-CM6-1-HR','EC-Earth_HR','GFDL','HadGEM_GC31_HH',
            'HadGEM_GC31_HM_highRes','HadGEM_GC31_HM_highResSST',
            'CMCC-CM2-VHR4']

# Station
Stat = 552

#SLR_list =['000','025','050','100','150','200','300']
SLR = '100'

Month_list = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# Plot time 
t_lims = ['2020-01-01','2021-01-01']

#===============================================================================
# %% Define some functions
#===============================================================================


#===============================================================================
# %% Load the  data
#===============================================================================

# Load the CDF difference dataset
files = glob(os.path.join(dir_in,'cdf_diff',SLR,'ERA5wl_Diff_*'))
ds = xr.open_mfdataset(files, engine='h5netcdf', parallel=True,
                       chunks={"time": -1, "station": 1, "cmip6": 1})

# pull out the data for the variable of interest
data = ds['cmip_diff']
data = data.sel(time=slice(t_lims[0], t_lims[1]))
data = data.isel(station=Stat)

data_mean = data.mean(dim='cmip6').compute() - float(SLR)

# Load the Water Level difference dataset
files = glob(os.path.join(dir_in,'ERA5','ERA5',f'ERA5_{SLR}','Results_Combined','DFM_wl_*'))
ds = xr.open_mfdataset(files, engine='h5netcdf', parallel=True,
                       chunks={"time": -1, "station": 1})

# pull out the data for the variable of interest
data_era5 = ds['waterlevel']
data_era5 = data_era5.sel(time=slice(t_lims[0], t_lims[1]))
data_era5 = data_era5.isel(station=Stat)




#===============================================================================
# %% Load the  data
#===============================================================================

fig = plt.subplots(2, 1,sharex=True,)
fig[0].set_size_inches(8, 6)
ax = fig[0].get_axes()

l1 = ax[0].plot(data_mean['time'],data_mean/100,color = 'k')
l2 = ax[0].plot([0,1],[0,0],'k--')

ax[0].set_xlim(pd.Timestamp('2020-01-01'),pd.Timestamp('2020-01-14'))

ax[0].set_title('CMIP6 Average Difference at each timestep ')


ax[0].grid()
ax[0].set_ylabel('Water Level Difference (cm)')


l1 = ax[1].plot(data_era5['time'],data_era5/10000,color = 'k')
l2 = ax[1].plot(data_era5['time'],((data_era5/10000) + (data_mean/10000) ),'r--')

ax[1].set_title('ERA5 Water Levels ')

ax[1].grid()
ax[1].set_xlabel('Water Level Difference (cm)')
ax[1].set_ylabel('Date')


fig[0].savefig(os.path.join(dir_out,f'Cmip6Diff_{Stat}.tiff'), dpi=600)

