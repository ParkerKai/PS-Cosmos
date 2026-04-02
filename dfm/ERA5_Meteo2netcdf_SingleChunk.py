# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:05:24 2023

This script converts ERA5 data into DFM netcdf files.
THis outputs for a user chosen period 
So does not splits the file by water year

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
dir_data = r'Y:\PS_Cosmos\ERA5\Download'
#dir_out  = r'C:\Users\kaparker\Documents\Data\Temp\Example_DFMfiles'
#dir_out = r'Y:\PS_Cosmos\DFM\ERA5_test\Meteo\WY_1997'
dir_out = r'Y:\PS_Cosmos\ERA5\FullPeriod_DFM'

# Model Variable
# var = 'uas' 'vas' 'psl'
var_list   = ['vas','uas','psl']

# Limits for grid
# Here determined from Babaks dfm netcdfs
lat_lim = np.array([46.95,51.6347])
lon_lim = warpTo360(np.array([-129.2000,-121.9703]))
lat_d   = 0.1   # grid spacing
lon_d   = 0.1

# Time slice to point (index)
t_start = datetime.datetime(1997, 8, 1, 0, 0) 
t_end   = datetime.datetime(1997, 12, 1, 0, 0) 

#vas_6hrPlevPt_CMCC-CM2-VHR4_hist-1950_r1i1p1f1_gn_199312010000-199312311800.nc 

#===============================================================================
# %% Load the Data
#===============================================================================

# ERA5 files to read and Open all
files = get_files_nc(dir_data)
ds = xr.open_mfdataset(files, engine='netcdf4', parallel=True)

# Pull out the time variable
time = ds['time'].values
    
# Convert to a datetime variable
date = [to_datetime(val) for val in time]
time = np.array(date)

t_start = time[0]
t_end   = time[-1]

#===============================================================================
# %% Output as  netcdf (as user chosen )
#===============================================================================

# Extract water year of data    
ind_wy =np.logical_and((time >= t_start),(time <= t_end))
    
# Extract
ds_wy = ds.sel(time = ind_wy)

# Drop Extra Variables and rename
#ds_wy['slp'] = ds_wy['msl']
ds_wy = ds_wy.drop_vars(["t2m", "tp"])

# Add standard name attribute that DFM wants
ds_wy['msl'].attrs['standard_name'] = 'air_pressure_fixed_height'
ds_wy['u10'].attrs['standard_name'] = 'eastward_wind'
ds_wy['v10'].attrs['standard_name'] = 'northward_wind'

# define global attributes
ds_wy.attrs = dict(
    Conventions = ds_wy.attrs['Conventions'],
    history      = ds_wy.attrs['history'],
    creation_date=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), 
    author='Kai Parker', 
    email='kaparker@usgs.gov',
    Model = 'ERA5',
    download = 'https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form',
    TemporalCoverage = '1940 to present',
    Sampling         = 'hourly',
    PostProcessing   = 'Extracted to PugetSound Region',
    )
    
# Output the file  
ds_wy.to_netcdf(os.path.join(dir_out,'ERA5_meteo.nc'),
    encoding={'u10': {'dtype': 'float32', '_FillValue': -9999},
              'v10': {'dtype': 'float32', '_FillValue': -9999},
              'msl': {'dtype': 'float32', '_FillValue': -9999},
              'time': {'dtype': 'int64', '_FillValue': -9999},
              'longitude': {'dtype': 'float32', '_FillValue': -9999},
              'latitude': {'dtype': 'float32', '_FillValue': -9999}})

    