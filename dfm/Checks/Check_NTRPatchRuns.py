# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:40:02 2024

This script loads in DFM outputs broken up by water year and combines them into
a single file

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
import matplotlib
import numpy as np
import xarray as xr
import datetime
import pandas as pd
import math 

#===============================================================================
# %% Define some functions
#===============================================================================

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

#===============================================================================
# %% User Defined inputs
#===============================================================================

#### Directories #####
dir_data1 = r'Y:\PS_Cosmos\02_models\DFM_Regional\CMIP6\EC-Earth_HR\future\Results\000'
dir_data2 = r'Y:\PS_Cosmos\02_models\DFM_Regional\CMIP6\EC-Earth_HR\future\Results\000_Rerun';

year = 2021

station = 'NOAA_9447130_Seattle'

#===============================================================================
# %% Load the data 
#===============================================================================

file1 = os.path.join(dir_data1,'WY_{}'.format(year),'WY_{}_000_0000_his.nc'.format(year))
file2 = os.path.join(dir_data2,'WY_{}'.format(year),'WY_{}_000_0000_his.nc'.format(year))

ds1 = xr.open_mfdataset(file1, engine='netcdf4', parallel=False)
ds2 = xr.open_mfdataset(file2, engine='netcdf4', parallel=False)


# Slice by time to remove spinup and downsample variables
start = "{}-10-01".format(year-1)
end   = "{}-10-01".format(year)

ds1 = ds1.sel(time=slice(start, end))
ds1 = ds1[{'waterlevel','station_id','station_name',
                 'station_x_coordinate','station_y_coordinate','time'}]

ds2 = ds2.sel(time=slice(start, end))
ds2 = ds2[{'waterlevel','station_id','station_name',
                 'station_x_coordinate','station_y_coordinate','time'}]

# Find location to grab 
names = ds1['station_id'].values
ind_grab = CheckArrayFor(names,station)

ds1 = ds1.isel(stations = ind_grab)
ds2 = ds2.isel(stations = ind_grab)

# Remove the bad data from the first run
# A couple of bad events for the ec-Earth runs
ind_bad = (ds1['time'] >= pd.Timestamp('2021-03-15')) & (ds1['time'] <= pd.Timestamp('2021-03-20'))
ds1['waterlevel'][ind_bad] = np.nan

ind_bad = (ds1['time'] >= pd.Timestamp('2016-12-25')) & (ds1['time'] <= pd.Timestamp('2017-01-01'))
ds1['waterlevel'][ind_bad] = np.nan

#===============================================================================
# %% plots 
#===============================================================================

fig, ax = matplotlib.pyplot.subplots(1, 1)
ax.plot(ds1['time'],ds1['waterlevel'],'k',label = 'BadNTR')
ax.plot(ds2['time'],ds2['waterlevel'],'r',label = 'GoodNTR')
ax.set_xlim(pd.Timestamp('{}-01-01'.format(year)),pd.Timestamp('{}-03-01'.format(year)))
#ax.set_xlim(100,200)
ax.grid()
ax.set_title('Water Level Comparison')
ax.set_ylabel('Elevation (NAVD88,m)')
ax.set_xlabel('Time')
ax.legend()

matplotlib.pyplot.show()

print(ds1['waterlevel'])


d_val = (ds1['waterlevel'].mean() - ds2['waterlevel'].mean()).compute()
print('Water Level difference is {} millimeters'.format(np.round(d_val.values*1000)))
