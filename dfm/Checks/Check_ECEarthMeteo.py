# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:52:15 2024

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
import dask

#===============================================================================
# %% Define some functions
#===============================================================================

def warpTo360(lon):
    lon_360 = lon % 360
    return lon_360


# The term U.S.Geological Survey "water year"  is defined as
# the 12-month period October 1, for any given year through September 30, of the following year.
# The water year is designated by the calendar year in which it ends and
# which includes 9 of the 12 months.
# Thus, the year ending September 30, 1999 is called the "1999" water year.
def get_water_year(dates):
    # Dates is a vector
    if hasattr(dates,'__iter__'):
        water_year = np.empty(len(dates),dtype = 'int64')
        for cnt,date in enumerate(dates):
            if date.month < 10:
                #adjust the year for the second half of the water year
                water_year[cnt]  = date.year
            else:
                water_year[cnt]  = date.year + 1
                    
    # Single date        
    else:
        if dates.month < 10:
            #adjust the year for the second half of the water year
            water_year = dates.year
        else:
            water_year = dates.year + 1
                
                
    return water_year

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
dir_data = r'Y:\PS_Cosmos\01_data\Climate\CMIP6\Meteo'


# Model to process
# model = 'CMCC-CM2-VHR4' 'CNRM-CM6-1-HR' 'EC-Earth_HR' 'GFDL' 'HadGEM_GC31_HH'
#         'HadGEM_GC31_HM_highRes' 'HadGEM_GC31_HM_highResSST'
model = 'EC-Earth_HR'

# Model Version to process
# ver = 'historic' 'future'
ver   = 'future'


# Time slice to point (index)
t_plot = 1000

dask.config.set(**{'array.slicing.split_large_chunks': True})


#vas_6hrPlevPt_CMCC-CM2-VHR4_hist-1950_r1i1p1f1_gn_199312010000-199312311800.nc 
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
# %% Figure out the date vector
#===============================================================================
#data = xr.open_mfdataset(os.path.join(dir_data,model,ver, f'psfm_{mod_short}_meteo.nc'))

#data = xr.open_mfdataset(os.path.join(dir_data,model,ver,f'psfm_{mod_short}_meteo_WY2040.nc'))
#data = xr.open_mfdataset(os.path.join(r'Y:\PS_Cosmos\02_models\DFM_Regional\CMIP6\EC-Earth_HR\future\Meteo\WY_2040'
#                                      ,'psfm_cmip6_meteo_WY2040.nc'))
data = xr.open_mfdataset(os.path.join(r'Y:\PS_Cosmos\05_xfer',
                                      'psfm_cmip6_meteo_WY2040.nc'))



data['u10'].isel(time=t_plot).transpose().plot()