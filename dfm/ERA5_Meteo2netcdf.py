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
dir_data = r'Y:\PS_Cosmos\01_data\Climate\ERA5\Download'
#dir_out  = r'C:\Users\kaparker\Documents\Data\Temp\Example_DFMfiles'
dir_out = r'Y:\PS_Cosmos\01_data\Climate\ERA5\WY_DFM';


# Model Variable
# var = 'uas' 'vas' 'psl'
var_list   = ['vas','uas','psl']

# buffer
# Add the is to the front and end of the timeseries as a spinup buffer and
# to make sure the timeseries covers the full simulation time.
buff = 7;  # in days

# Limits for grid
# Here determined from Babaks dfm netcdfs
lat_lim = np.array([46.95,51.6347])
lon_lim = warpTo360(np.array([-129.2000,-121.9703]))
lat_d   = 0.1   # grid spacing
lon_d   = 0.1

# Time slice to point (index)
t_plot = 102

#vas_6hrPlevPt_CMCC-CM2-VHR4_hist-1950_r1i1p1f1_gn_199312010000-199312311800.nc 

#===============================================================================
# %% Figure out the date vector
#===============================================================================
# only want data for a timeperiod we have all the dat
# Also want a consistent time sampling
# start_time = np.empty(len(var_list), dtype = datetime.datetime)
# end_time   = np.empty(len(var_list), dtype = datetime.datetime)
# dt         = np.empty(len(var_list), dtype = datetime.timedelta)
# for cnt,var in enumerate(var_list):
#     files = glob(os.path.join(dir_data,model,ver,'{}*'.format(var)))
#     ds = xr.open_mfdataset(files, engine='netcdf4', parallel=False)
    
#     # modify the calendar if 360
#     if (ds['time'].dt.calendar == '360_day'):
#         ds = ds.convert_calendar('noleap', dim='time', align_on='year')
    
#     # Pull out the time varibale
#     time = ds['time'].values
    
#     # Convert to a datetime variable
#     # temp = ds.indexes['time'].to_datetimeindex() # possibly better solution using pandas timestamp
#     date = [to_datetime(val) for val in time]
#     time = np.array(date)
    
#     start_time[cnt] = time[0]
#     end_time[cnt]   = time[-1]
#     dt[cnt]         = time[1]-time[0]    
    
# # Create a new (and consistent) timeseries
# # Only do overlapping time.
# t = np.arange(np.max(start_time), np.min(end_time), np.median(dt))
# date = [to_datetime(val) for val in t]
# time_wnt = np.array(date)

#===============================================================================
# %% Load the Data
#===============================================================================

# ERA5 files to read and Open all
files = get_files_nc(dir_data)
ds = xr.open_mfdataset(files, engine='netcdf4', parallel=True)

# Fix for the early downloaded last year which has 2 experiments (early reporting of data).
ds = ds.loc[dict(expver=1)]
ds = ds.drop_vars('expver')

# Pull out the time variable
time = ds['time'].values
    
# Convert to a datetime variable
date = [to_datetime(val) for val in time]
time = np.array(date)

#===============================================================================
# %% Output as yearly netcdf (water year)
#===============================================================================
# Unique years in timeseries
year_vec = [int(time[i].strftime("%Y")) for i in range(len(time))]
year_vec = np.array(year_vec)
month_vec = [int(time[i].strftime("%m")) for i in range(len(time))]
month_vec = np.array(month_vec)

# Find first and last water years
inds = np.argwhere(month_vec == 10)  # Start of first water year
yr1 = year_vec[inds[0]]+1               # Water year is defined by ending year (so +1 from starting year)
inds = np.argwhere(month_vec == 9)       # End of water year is last september date
yr2  = year_vec[inds[-1]]

years = np.arange(yr1,yr2+1, 1)

print(years)
for yr in years:
    print('Outputting Year {}'.format(yr))

    # Extract water year of data    
    ind_wy = np.logical_and((time >= (datetime.datetime(yr-1, 10, 1, 0, 0) - datetime.timedelta(days=buff))), 
                                    (time <= (datetime.datetime(yr, 9, 30, 0, 0)   +datetime.timedelta(days=buff))))
    
    # add another datapoint to make sure covers the whole period
    ind_last = np.max(np.where(ind_wy)[0])
    if (ind_last < ind_wy.size):
        ind_wy[ind_last+1] = True
        
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
        PostProcessing   = 'Trimmed to Water Year',
        )
        
    # Output the file  
    ds_wy.to_netcdf(os.path.join(dir_out,'ERA5_meteo_WY{}.nc'.format(yr)),
        encoding={'u10': {'dtype': 'float32', '_FillValue': -9999},
                  'v10': {'dtype': 'float32', '_FillValue': -9999},
                  'msl': {'dtype': 'float32', '_FillValue': -9999},
                  'time': {'dtype': 'int64', '_FillValue': -9999},
                  'longitude': {'dtype': 'float32', '_FillValue': -9999},
                  'latitude': {'dtype': 'float32', '_FillValue': -9999}})

 
    #if yr == 2010:
    #    
    #    # Pressure field
    #    pres = ds_wy['msl'].sel(time='2010-01-04T00:00:00')

    #    fig, ax1 = matplotlib.pyplot.subplots(1, 1)
    #    
    #    fig.set_size_inches(7,6)
    #    c = ax1.pcolormesh(pres['longitude'],pres['latitude'],pres.values/100,shading = 'nearest')
    #    cbar = fig.colorbar(c, ax=ax1)
    #    cbar.ax.set_ylabel('Pressure (HPa)')
    #    ax1.set_title('2010-01-04T00:00:00')
        
    #    print('figure created')
    