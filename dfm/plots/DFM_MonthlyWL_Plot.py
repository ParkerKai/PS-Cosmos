# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:17:26 2024

This script Plots the monthly average water levels at Seattle
 
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
import xarray as xr
import matplotlib 

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'Y:\PS_Cosmos\02_models\DFM_Regional'
dir_out = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\figures\DFM'

# Station
Stat = 562

#SLR_list =['000','025','050','100','150','200','300']
SLR = '000'


#===============================================================================
# %% Define some functions
#===============================================================================

#===============================================================================
# %% Open the Data
#===============================================================================    

# Open all the data
files = glob(os.path.join(dir_in,'ERA5','ERA5',f'ERA5_{SLR}','Results_Combined','DFM_wl*'))
ds = xr.open_mfdataset(files,parallel=True, chunks={"time": -1, "station": 1})

# scale and select for data
ds = ds['waterlevel']
ds = ds.isel(station=Stat)/10000
ds = ds.resample(time="1MS").mean(dim="time")

#===============================================================================
# %%Plot
#===============================================================================    

fig = matplotlib.pyplot.subplots(1, 1)
fig[0].set_size_inches(8, 6)
ax = fig[0].get_axes()
ax = ax[0]
                
l1 = ax.plot(ds['time'],ds,'k')

ax.set_title('DFM Water Levels')
ax.set_ylabel('Water Level (NAVD88,m)')
ax.set_xlabel('Date')
ax.grid()


#fig[0].savefig(os.path.join(dir_out,'DFM_MonthlyWL_Seattle.tiff'), dpi=600)

