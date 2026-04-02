# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:07:15 2023

@author: kaparker
"""

#===============================================================================
# %% Import Modules
#===============================================================================
import os
from glob import glob
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
import matplotlib
import datetime
import sys
import pandas as pd

#===============================================================================
# %% Define some functions
#===============================================================================

def get_files_nc(directory):
    listing = os.listdir(directory)
    
    # Create a list of all netcdfs in the directory
    # Number of files
    num_files = 0
    for cnt,file in enumerate(listing):
        if file.endswith('.nc'):
            num_files = num_files +1 
    
    cnt = 0
    files = ['empty']*num_files
    for file in listing:
        if file.endswith('.nc'):
            files[cnt] = os.path.join(directory,file) 
            cnt = cnt +1
    # or use the glob strategy
    # from glob import glob
    # files = glob(os.path.join(directory,'*.nc'))

    return files


#===============================================================================
# %% User Defined inputs
#===============================================================================

#### Directories #####
dir_in_dfm = r'D:\DFM'
dir_in_sim = r'Y:\PS_Cosmos\DFM\ERA5'
dir_era5 = r'Y:\PS_Cosmos\ERA5\Download'
dir_gauge   =r'Y:\PS_Cosmos\Crosby_Archive\NOAA\observations\noaa_coop_erdapp_complete\NewDownload'

dir_out = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\figures'
#===============================================================================
# %% Read the Data 
#===============================================================================
# REad the output wl data
ds_wl = xr.open_mfdataset(os.path.join(dir_in_dfm,'ERA5_wl.nc'),
                       engine='netcdf4', parallel=False)

# Read the meteo data 
ds_meteo = xr.open_mfdataset(os.path.join(dir_in_sim,'Meteo','WY_2011','ERA5_meteo_WY2011.nc'),
                        engine='netcdf4', parallel=False)     

#
#ds_meteo = xr.open_mfdataset(os.path.join(dir_in_sim,'ERA5_meteo_WY2008.nc'),
#                        engine='netcdf4', parallel=False)   

# Read the raw meteo data 
files = get_files_nc(dir_era5)
ds_era5 = xr.open_mfdataset(files, engine='netcdf4', parallel=True)


# Read the pressure
Pres_Stat = pd.read_csv(os.path.join(dir_gauge,'NOAA_TG_9443090_Pressure.csv'),
                   delimiter=',',parse_dates=['Date'],index_col=['Date'])
Pres_Stat['Pressure'].replace(-9999, np.nan, inplace=True)


#===============================================================================
# %% Plots 
#===============================================================================


########### Station Pressure ############
fig, [ax1,ax2] = matplotlib.pyplot.subplots(2, 1)
pres_stat = ds_meteo.sel(dict(latitude = 48.25, longitude = -127))['msl']
pres_stat2= ds_era5.sel(dict(latitude = 48.25, longitude = -127))['msl']


fig.set_size_inches(6,6)

ax1.plot(pres_stat['time'].values, pres_stat.values/100,'k',label='ERA5')
#ax1.plot(Pres_Stat.index, Pres_Stat['Pressure'].values,'r',label='NOAA')
ax2.plot(pres_stat2['time'].values, pres_stat2.values/100,'k',label='ERA5_2')

ax1.set_xlim(pd.Timestamp('2011-09-01'),pd.Timestamp('2010-11-01'))
ax2.set_xlim(pd.Timestamp('2011-09-01'),pd.Timestamp('2010-11-01'))

ax1.set_ylim(980,1040)
ax2.set_ylim(980,1040)
ax1.grid()
ax2.grid()
ax1.set_title('Pressure near Neah Bay')
ax1.set_ylabel('Pressure (HPa)')


########### Pressure Fields ############

# Pressure field
pres = ds_meteo['msl'].sel(time='2011-01-04T00:00:00')
u = ds_meteo['u10'].sel(time=   '2011-01-04T00:00:00')
v = ds_meteo['v10'].sel(time=   '2011-01-04T00:00:00')

fig, ax1 = matplotlib.pyplot.subplots(1, 1)

fig.set_size_inches(7,6)
c = ax1.pcolormesh(pres['longitude'],pres['latitude'],pres.values/100,shading = 'nearest')
cbar = fig.colorbar(c, ax=ax1)
cbar.ax.set_ylabel('Pressure (HPa)')
ax1.set_title('2010-01-04T00:00:00')





fig, [ax1,ax2,ax3] = matplotlib.pyplot.subplots(3, 1)

fig.set_size_inches(5,8)
c = ax1.pcolormesh(pres['longitude'],pres['latitude'],pres.values/100,shading = 'nearest')
cbar = fig.colorbar(c, ax=ax1)
cbar.ax.set_ylabel('Pressure (HPa)')
ax1.set_title('2018-01-16T07:00:00')

c = ax2.pcolormesh(u['longitude'],u['latitude'],u.values,shading = 'nearest')
cbar = fig.colorbar(c, ax=ax2)
cbar.ax.set_ylabel('U10 (m/s)')

c = ax3.pcolormesh(v['longitude'],v['latitude'],v.values,shading = 'nearest')
cbar = fig.colorbar(c, ax=ax3)
cbar.ax.set_ylabel('V10 (m/s)')
fig.savefig(os.path.join(dir_out,'Meteo.png'), dpi=600)





