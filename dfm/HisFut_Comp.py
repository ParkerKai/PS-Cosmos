# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:22:20 2024

This script plots the waterlevels for CMIP6 historic-future runs 


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
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib
import pandas as pd
from scipy import interpolate

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
dir_in = r'D:\DFM'
dir_out = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\figures\DFM'

# Model to process
Mod = 'EC-Earth_HR'

# Station to plot
station = 'NOAA_9447130_Seattle'
#station = 'NOAA_9443090_Neah'
#station  = r'ndbc_46087'

#===============================================================================
# %% Define some functions
#===============================================================================
# def CheckArrayFor(data,string):
#     out = []
#     for cnt,row in enumerate(data):
#         string = str(row)
#         # check if string present on a current line
#         if string.find(station) != -1:
#             if len(out) == 0:
#                 out = cnt
#             else:
#                 out = out.append(cnt)

#     return out

def CheckArrayFor(data,station):

    # Single string for station (turn into a list)
    if isinstance(station,str):
        station = [station]
    
    out = np.full(len(station),0,dtype='int32')
    
    for cnt1,station_pull in enumerate(station):
        
        out_station = np.full(1,0,dtype='int32')
        for cnt2,row in enumerate(data):
            string_row = str(row)
            # check if string present on a current line
            if string_row.find(station_pull) != -1:
                out_station = np.array(cnt2,dtype='int32')

        out[cnt1] = out_station
            
        
    return out

def Load_Station(files,station_string):
    # Test open
    test_open = xr.open_mfdataset(files[0], engine='netcdf4', parallel=False)
    names = test_open['station'].values
    
    ind_grab = CheckArrayFor(names,station)
    
    if (len(ind_grab) == 0):
        print('Station Not Found!!!')
    
    for cnt,file in enumerate(files):
        data = xr.open_mfdataset(file, engine='netcdf4', parallel=False)
        
        #Subset to station of interest
        data = data.isel(station=ind_grab)    
        
        if cnt == 0 :
            data_save = data
        else:
            data_save = xr.concat([data_save,data],dim='time')
    
    # Convert to millimeters 
    data_save['waterlevel'] = data_save['waterlevel']/10
    return data_save

#===============================================================================
# %% Load the data 
#===============================================================================

# ERA5
files = glob(os.path.join(dir_in,'ERA5','*.nc'))
ds_era5 = Load_Station(files,station)


# CMIP6 His
files = glob(os.path.join(dir_in,'Cmip6',Mod,'historic','*.nc'))
ds_cmipH = Load_Station(files,station)

# Subset the historic run to have equal timeperiods
ds_cmipH = ds_cmipH.sel(time=slice("1980-01-01", "2014-01-01"))

# CMIP6 Fut
files = glob(os.path.join(dir_in,'Cmip6',Mod,'future','050','*.nc'))
ds_cmipF = Load_Station(files,station)
        

#===============================================================================
# %% Remove some bad events
#===============================================================================
# A couple of bad events for the ec-Earth runs
# ind_bad = (ds_cmipF['time'] >= pd.Timestamp('2021-03-15')) & (ds_cmipF['time'] <= pd.Timestamp('2021-03-20'))
# ds_cmipF['waterlevel'][ind_bad] = np.nan

ind_bad = (ds_cmipF['time'] >= pd.Timestamp('2016-12-25')) & (ds_cmipF['time'] <= pd.Timestamp('2017-01-01'))
ds_cmipF['waterlevel'][ind_bad] = np.nan


#===============================================================================
# %% Calculate the statistics
#===============================================================================
cdf_era5  = ECDF(ds_era5['waterlevel'].squeeze())
ind_keep = np.isfinite(cdf_era5.x)
cdf_era5.x = cdf_era5.x[ind_keep]
cdf_era5.y = cdf_era5.y[ind_keep]

cdf_cmipH = ECDF(ds_cmipH['waterlevel'].squeeze())
ind_keep = np.isfinite(cdf_cmipH.x)
cdf_cmipH.x = cdf_cmipH.x[ind_keep]
cdf_cmipH.y = cdf_cmipH.y[ind_keep]

cdf_cmipF = ECDF(ds_cmipF['waterlevel'].squeeze())
ind_keep = np.isfinite(cdf_cmipF.x)
cdf_cmipF.x = cdf_cmipF.x[ind_keep]
cdf_cmipF.y = cdf_cmipF.y[ind_keep]


# Difference in Water Levels at all quantiles 
quants = np.arange(0.01, 1, 0.01)
fH = interpolate.interp1d(cdf_cmipH.y, cdf_cmipH.x)
fF = interpolate.interp1d(cdf_cmipF.y, cdf_cmipF.x)

q_diff = fF(quants) - fH(quants) 

#===============================================================================
# %% Plot
#===============================================================================

fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 6)

ax.plot(ds_cmipH['time'],ds_cmipH['waterlevel'], color = 'red',label = 'CmipH')
ax.plot(ds_cmipF['time'],ds_cmipF['waterlevel'], color = 'blue',label = 'CmipF')


#ax.set_xlim(pd.Timestamp('2016-12-25'),pd.Timestamp('2017-01-01'))
#ax.set_xlim(100,200)
ax.grid()
ax.set_title('Water Levels')
ax.set_ylabel('WL (NAVD88,mm)')
ax.set_xlabel('date')
ax.legend()
#ax.set_xlim([pd.Timestamp('2016-12-01'),pd.Timestamp('2017-01-01')])
matplotlib.pyplot.show()
#fig.savefig(os.path.join(dir_out,'DFM_Cmip6_TS_EcEarth_Neah'),  dpi=800,
#        bbox_inches='tight')          

fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 8)

ax.plot(cdf_era5.x,cdf_era5.y, color = 'black',label = 'ERA5')
ax.plot(cdf_cmipH.x,cdf_cmipH.y, color = 'red',label = 'CmipH')
ax.plot(cdf_cmipF.x,cdf_cmipF.y, color = 'blue',label = 'CmipF')

#ax.set_xlim(pd.Timestamp('2000-08-01'),pd.Timestamp('2001-08-01'))
#ax.set_xlim(100,200)
ax.grid()
ax.set_title('Water Level CDF')
ax.set_xlabel('WL (NAVD88,mm)')
ax.set_ylabel('CDF (emp.)')
ax.legend()


matplotlib.pyplot.show()
fig.savefig(os.path.join(dir_out,'DFM_Cmip6_cdf_EcEarth_Neah'),  dpi=800,
        bbox_inches='tight')  



fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 8)

ax.plot(cdf_era5.x,cdf_era5.y, color = 'black',label = 'ERA5')
ax.plot(cdf_cmipH.x,cdf_cmipH.y, color = 'red',label = 'CmipH')
ax.plot(cdf_cmipF.x,cdf_cmipF.y, color = 'blue',label = 'CmipF')

ax.set_ylim(0.9,1)
ax.set_xlim(2250,3750)

ax.grid()
ax.set_title('Water Level CDF')
ax.set_xlabel('WL (NAVD88,mm)')
ax.set_ylabel('CDF (emp.)')
ax.legend()

matplotlib.pyplot.show()
fig.savefig(os.path.join(dir_out,'DFM_Cmip6_cdfZoom_EcEarth_Neah'),  dpi=800,
        bbox_inches='tight')  


fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 8)

ax.plot(q_diff,quants, color = 'black',label = 'ERA5')

ax.grid()
ax.set_title('Difference in CMIP6 CDF')
ax.set_xlabel('Diff in WL (mm)')
ax.set_ylabel('Quantile')

matplotlib.pyplot.show()
fig.savefig(os.path.join(dir_out,'DFM_Cmip6_cdfDiff_EcEarth_Neah'),  dpi=800,
        bbox_inches='tight')  
