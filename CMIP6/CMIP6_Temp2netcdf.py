# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:05:24 2023

This script converts CMIP6 data into DFM netcdf files.
Splits the file by water year and also outputs the file unsplit

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
from scipy.interpolate import griddata
import matplotlib
import datetime
import sys
import pandas as pd

#===============================================================================
# %% Define some functions
#===============================================================================

def warpTo360(lon):
    lon_360 = lon % 360
    return lon_360

def to_datetime(d):
    import datetime as dt
    import cftime
    import pandas as pd
    
    if isinstance(d, dt.datetime):
        return d
    if isinstance(d, cftime.DatetimeNoLeap):
        return dt.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    elif isinstance(d, cftime.DatetimeGregorian):
        return dt.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    elif isinstance(d, cftime.DatetimeJulian):
        return dt.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    elif isinstance(d, cftime.Datetime360Day):
        return dt.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    elif isinstance(d, str):
        errors = []
        for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return dt.datetime.strptime(d, fmt)
            except ValueError as e:
                errors.append(e)
                continue
        raise Exception(errors)
    elif isinstance(d, np.datetime64):
        temp = d.astype(dt.datetime)
        if isinstance(temp,dt.datetime):
            out = temp
        elif isinstance(temp,int):
            out = pd.Timestamp(d).to_pydatetime()
        return out
    else:
        raise Exception("Unknown value: {} type: {}".format(d, type(d)))


def interp_datetime(date_q,date,data):
    import calendar
   
    def toTimestamp(t):
        out = [calendar.timegm(val.timetuple()) for val in t]
        out = np.array(out)
        return out
  

    result = np.interp(toTimestamp(date_q),toTimestamp(date),data)
    return result
    

#===============================================================================
# %% User Defined inputs
#===============================================================================

#### Directories #####
dir_data = r'Z:\CMIP6'
#dir_out  = r'C:\Users\kaparker\Documents\Data\Temp\Example_DFMfiles'
dir_out = r'Y:\PS_Cosmos\CMIP6\Temperature';

# Model to process
# model = 'CMCC-CM2-VHR4' 'CNRM-CM6-1-HR' 'EC-Earth_HR' 'GFDL' 'HadGEM_GC31_HH'
#         'HadGEM_GC31_HM_highRes' 'HadGEM_GC31_HM_highResSST'
model = 'CNRM-CM6-1-HR'

# Model Version to process
# ver = 'historic' 'future'
ver   = 'future'

# Model Variable
# var = 'uas' 'vas' 'psl'
# var   = 'pr'
var = 'tas'

# Limits for grid
# Here determined from Babaks dfm netcdfs
lat_lim = np.array([46.95,51.6347])
lon_lim = warpTo360(np.array([-129.2000,-121.9703]))

#===============================================================================
# %% Figure out name shortening
#===============================================================================

if model == 'CMCC-CM2-VHR4':
    mod_short = 'CMCC'
    
elif model == 'CNRM-CM6-1-HR':
    mod_short = 'CNRM'
    
elif model == 'EC-Earth_HR':
    mod_short = 'EcEarth'

elif model == 'GFDL':
    mod_short = 'GFDL'

elif model == 'HadGEM_GC31_HH':
    mod_short = 'HadGemHH'

elif model == 'HadGEM_GC31_HM_highRes':
    mod_short = 'HadGemHM'

elif model == 'HadGEM_GC31_HM_highResSST':
    mod_short = 'HadGemHMsst'

else:
     print('Model Choice not found')   
     sys.exit()


#===============================================================================
# %% Load the Data
#===============================================================================

# Loop thorugh each variable 
print('processing {} for {}'.format(var,model))

files = glob(os.path.join(dir_data,model,ver,'{}*'.format(var)))
ds = xr.open_mfdataset(files, engine='netcdf4', parallel=False)

# Find indexes for location we want
ds2 = ds[var].sel(lat=slice(lat_lim[0]-0.75,lat_lim[1]+0.75),
                    lon =slice(lon_lim[0]-0.75,lon_lim[1]+0.75))

# modify the calendar if 360
if (ds2['time'].dt.calendar == '360_day'):
    ds2 = ds2.convert_calendar('noleap', dim='time', align_on='year')

# Pull out the time variable
time = ds2['time'].values

#ds2 = ds2.interp(time = time)

# Convert to a datetime variable
date = [to_datetime(val) for val in time]
time = np.array(date)

# Lat/Lon reduced to region of interest
ds_lat = ds2.lat.values
ds_lon = ds2.lon.values

# Convert to grid and extract variable
ds_longrid, ds_latgrid = np.meshgrid(ds_lon, ds_lat, indexing='xy')

Var_extract = ds2.values

#===============================================================================
# %% Plot to make sure things are working
#===============================================================================
# cfig, ax = matplotlib.pyplot.subplots(1, 1)
#cf = ax.pcolormesh(xv,yv,var_intrp_ts[t_plot,:,:],
#              shading = 'nearest')
#ax.scatter(ds_longrid.flatten(),ds_latgrid.flatten(), s=20, c=Var_extract[t_plot,:,:].flatten(),
#           edgecolors='k', label = 'CMIP6')
#
# ax.set_title('Interpolated {} grid'.format(var))
# fig.colorbar(cf, ax=ax, label = var)
# matplotlib.pyplot.show()
    

#===============================================================================
# %% Output of combined final file
#===============================================================================
# define data with variable attributes

data_vars = {'tas':(['time','lat','lon'], Var_extract, 
                         ds2.attrs)}


coords = {'lon':(['lon'], ds2['lon'].values, ds2['lon'].attrs),
          'lat':(['lat'], ds2['lat'].values, ds2['lat'].attrs),
          'time':(['time'],ds2['time'].values,ds2['time'].attrs)}

# define coordinates
# coords=dict(
#     lon  =(["x"], ds2['lon']),
#     lat  =(["y"],ds2['lat'],
#     time = time_wnt)

# define global attributes
attrs = dict(
    creation_date=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), 
    author='Kai Parker', 
    email='kaparker@usgs.gov',
    Cmip6 = model,
    timeperiod = ver,
    variant_label= ds.attrs['variant_label'])

# create dataset
ds_out = xr.Dataset(data_vars=data_vars, 
                coords=coords, 
                attrs=attrs)


# Output the file  
ds_out.to_netcdf(os.path.join(dir_out,ver,
                              'psfm_{}_{}_temp.nc'.format(mod_short,ver)),
                 encoding={'tas': {'dtype': 'float32', '_FillValue': -9999},
                           'time': {'dtype': 'int64', '_FillValue': -9999},
                           'lon': {'dtype': 'float64', '_FillValue': -9999},
                           'lat': {'dtype': 'float64', '_FillValue': -9999}})
    
    