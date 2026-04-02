# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:30:24 2023

This script converts CMIP6 data into netcdf files.
It is specifically pulling for the WFLOW hydrology team
It leaves model resolution the same and just trims to regional limits


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
import dask

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

#===============================================================================
# %% User Defined inputs
#===============================================================================

#### Directories #####
dir_data = r'Z:\CMIP6\Orography'
#dir_out  = r'C:\Users\kaparker\Documents\Data\Temp\Example_DFMfiles'
dir_out = r'Y:\PS_Cosmos\01_data\Climate\CMIP6\Wflow_Extract\orog';

# Model to process
# model = 'CNRM-CM6-1-HR' 'EC-Earth_HR' 'GFDL' 'HadGEM_GC31_HH'
#         'HadGEM_GC31_HM_highRes' 'HadGEM_GC31_HM_highResSST'
model = 'HadGEM_GC31_HM_highResSST'

# Model Variable
var = 'orog'

# Limits for geographical clipping
lat_lim = np.array([45.5,49.5])
lon_lim = warpTo360(np.array([-125.0,-120.5]))

#===============================================================================
# %% Figure out name shortening
#===============================================================================

if model == 'CMCC-CM2-VHR4':
    mod_short = 'CMCC'
    mod_find  = 'blehbleh'
    
elif model == 'CNRM-CM6-1-HR':
    mod_short = 'CNRM'
    mod_find  = 'CNRM-CM6-1-HR'

elif model == 'EC-Earth_HR':
    mod_short = 'EcEarth'
    mod_find  = 'EC-Earth3P-HR'

elif model == 'GFDL':
    mod_short = 'GFDL'
    mod_find  = 'BlehBleh'

elif model == 'HadGEM_GC31_HH':
    mod_short = 'HadGemHH'
    mod_find  = 'HadGEM3-GC31-HH'

elif model == 'HadGEM_GC31_HM_highRes':
    mod_short = 'HadGemHM'
    mod_find  = 'HadGEM3-GC31-HM'

elif model == 'HadGEM_GC31_HM_highResSST':
    mod_short = 'HadGemHMsst'
    mod_find  = 'HadGEM3-GC31-HM'

else:
     print('Model Choice not found')   
     sys.exit()

#===============================================================================
# %% Load the Data
#===============================================================================

# Loop thorugh each variable 
print('Processing: {} {}'.format(var, mod_short))

files = glob(os.path.join(dir_data,'*{}*'.format(mod_find)))
if len(files) != 1:
    raise Exception('Incorect number of files found!')
    

ds = xr.open_mfdataset(files, engine='netcdf4', parallel=False)

# Find indexes for location we want
ds2 = ds[var].sel(lat=slice(lat_lim[0]-0.75,lat_lim[1]+0.75),
                    lon =slice(lon_lim[0]-0.75,lon_lim[1]+0.75))


#===============================================================================
# %% Export the Data
#===============================================================================
print('Exporting: {} {}'.format(var, mod_short))


# Output the file  
ds2.to_netcdf(os.path.join(dir_out,'cmip6_{}_{}.nc'.format(mod_short,var)))

    
    