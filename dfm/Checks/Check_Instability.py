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
ds_wl = xr.open_mfdataset(os.path.join(dir_in_dfm,'ERA5_wl_1970.nc'),
                       engine='netcdf4', parallel=False)

# Read the meteo data 
ds_meteo = xr.open_mfdataset(os.path.join(dir_in_sim,'Meteo','WY_1971','ERA5_meteo_WY1971.nc'),
                        engine='netcdf4', parallel=False)     

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

########### Instability ############
# Find station of interest
for cnt,stat in enumerate(ds_wl['station'].values):
    if 'NOAA_9443090_Neah_Bay' in str(stat): 
        ind_keep = cnt

wl_stat = ds_wl.isel(dict(station = ind_keep))['waterlevel']/10000

fig, ax = matplotlib.pyplot.subplots(1, 1)
ax.plot(wl_stat['time'].values, wl_stat.values,'k')
ax.set_xlim(pd.Timestamp('1971-03-17'),pd.Timestamp('1971-03-18'))
ax.set_ylim(-10,20)
ax.grid()
ax.set_title('WL {}'.format(str(wl_stat['station'].values)))
ax.set_ylabel('Water level (NAVD88,m)')

ind_max = np.nanargmax(wl_stat.values)
print(wl_stat['time'][ind_max].values)


########### Station Pressure ############
pres_stat = ds_meteo.sel(dict(latitude = 48.25, longitude = -127))['msl']
pres_stat2= ds_era5.sel(dict(latitude = 48.25, longitude = -127))['msl']

u_stat    = ds_meteo.sel(dict(latitude = 48.25, longitude = -127))['u10']
v_stat    = ds_meteo.sel(dict(latitude = 48.25, longitude = -127))['v10']


fig, [ax1,ax2,ax3] = matplotlib.pyplot.subplots(3, 1)
fig.set_size_inches(8,6)

ax1.plot(pres_stat['time'].values, pres_stat.values/100,'k',label='ERA5')
#ax1.plot(Pres_Stat.index, Pres_Stat['Pressure'].values,'r',label='NOAA')
ax1.plot(pres_stat2['time'].values, pres_stat2.values/100,'b',label='ERA5_2')

ax1.set_xlim(pd.Timestamp('1971-03-17'),pd.Timestamp('1971-03-18'))
#ax.set_xlim(100,200)
ax1.grid()
ax1.set_title('Atmospheric Forcing Near Neah Bay')
ax1.set_ylabel('Pressure (HPa)')
ax1.legend()

ax2.plot(u_stat['time'].values, u_stat.values,'k',label='ERA5')
ax2.set_xlim(pd.Timestamp('1971-03-17'),pd.Timestamp('1971-03-18'))
#ax.set_xlim(100,200)
ax2.grid()
ax2.set_ylabel('Wind U (m/s)')


ax3.plot(v_stat['time'].values, v_stat.values,'k',label='ERA5')
ax3.set_xlim(pd.Timestamp('1971-03-17'),pd.Timestamp('1971-03-18'))
#ax.set_xlim(100,200)
ax3.grid()
ax3.set_ylabel('Wind V (m/s)')


ind_min = np.nanargmin(pres_stat.values)
print(pres_stat['time'][ind_min].values)

########### Pressure Fields ############

# Pressure field
pres = ds_meteo['msl'].sel(time='1971-03-17T12:00:00') #,method = 'nearest'
u = ds_meteo['u10'].sel(   time='1971-03-17T12:00:00')
v = ds_meteo['v10'].sel(   time='1971-03-17T12:00:00')

fig, ax1 = matplotlib.pyplot.subplots(1, 1)

fig.set_size_inches(7,6)
c = ax1.pcolormesh(pres['longitude'],pres['latitude'],pres.values/100,shading = 'nearest')
cbar = fig.colorbar(c, ax=ax1)
cbar.ax.set_ylabel('Pressure (HPa)')
ax1.set_title('2018-01-15T00:00:00')



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





