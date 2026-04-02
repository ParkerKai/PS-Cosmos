# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:33:01 2025

This script checks input forcing to see why flooding is so high in the duwamish basin.
 

 
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
import numpy as np
import xarray as xr 
import scipy 
from glob import glob 
#from hydromt_sfincs import SfincsModel
import matplotlib 
import pandas as pd
#===============================================================================
# %% User Defined
#===============================================================================
dir_in = r'Y:\PS_Cosmos\02_models\SFINCS\20250122_synthetic_future_meanchange_100yr_Intel\01_King'

Stat_plot = 50 
SLR_list = ['000','025','050','100','150','200','250']
SLR_list = ['000','025','100','200']

#===============================================================================
# %% Functions
#===============================================================================

#===============================================================================
# %% Read in forcing
#===============================================================================
# ds_000 = xr.open_mfdataset(os.path.join(dir_in,'000','SY000','sfincs_bndbzs.nc'),parallel=True,decode_times=False)
# ds_025 = xr.open_mfdataset(os.path.join(dir_in,'025','SY000','sfincs_bndbzs.nc'),parallel=True,decode_times=False)
# ds_100 = xr.open_mfdataset(os.path.join(dir_in,'100','SY000','sfincs_bndbzs.nc'),parallel=True,decode_times=False)
# ds_200 = xr.open_mfdataset(os.path.join(dir_in,'200','SY000','sfincs_bndbzs.nc'),parallel=True, decode_times=False)


# pull_000 = ds_000.isel(stations=100)
# pull_025 = ds_025.isel(stations=100)
# pull_100 = ds_100.isel(stations=100)
# pull_200 = ds_200.isel(stations=100)


# fig, ax = matplotlib.pyplot.subplots(1, 1)
# fig.set_size_inches(12, 7)


# # ax.plot(pull_000['time'],pull_000['zs'], label = '000')
# # ax.plot(pull_100['time'],pull_100['zs'], label = '100')
# # ax.legend()
# # ax.grid()

# diff_025 = (pull_025['zs'].mean() - pull_000['zs'].mean()).values
# diff_100 = (pull_100['zs'].mean() - pull_000['zs'].mean()).values
# diff_200 = (pull_200['zs'].mean() - pull_000['zs'].mean()).values


# print(f'diff for 25cm {diff_025}')
# print(f'diff for 100cm {diff_100}')
# print(f'diff for 200cm {diff_200}')





#===============================================================================
# %% Read in forcing
#===============================================================================
# files = glob(r'Y:\PS_Cosmos\02_models\DFM_Regional\cdf_diff\000\ERA5wl_Diff_1941*')
# dfm_000 = xr.open_mfdataset(files,parallel=True)
# dfm_000 = dfm_000.sel(station = 'NOAA_9444090_Port_Angeles')

for SLR in SLR_list:
    dfm_slr = []
    files = glob(os.path.join(r'Y:\PS_Cosmos\02_models\DFM_Regional\cdf_diff',SLR,'ERA5wl_Diff_1942*'))
    dfm_slr = xr.open_mfdataset(files,parallel=True, engine ='h5netcdf')
    dfm_slr['cmip_diff'] = dfm_slr['cmip_diff'].astype('float64')
    dfm_slr['cmip_diff'] = dfm_slr['cmip_diff'].where(dfm_slr.cmip_diff > -200000000, np.nan)
    dfm_slr = dfm_slr.sel(station = 'NOAA_9444090_Port_Angeles')
    
    mean_cmip_diff = dfm_slr['cmip_diff'].mean(dim=['cmip6','time']).values/10000
    print(f'Mean CMIPdiff for {SLR}: {mean_cmip_diff}')



# fig, [ax1,ax2] = matplotlib.pyplot.subplots(2, 1)
# fig.set_size_inches(12, 7)

# for ii in range(len(dfm_000['cmip6'])):
#     ax1.plot(dfm_000.isel(cmip6=ii)['time'],dfm_000.isel(cmip6=ii)['cmip_diff']/10000,label = '000')
#     ax2.plot(dfm_100.isel(cmip6=ii)['time'],dfm_100.isel(cmip6=ii)['cmip_diff']/10000,label = '000')
# ax.legend()
# ax.grid()
