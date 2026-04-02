# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:05:24 2023

This script looks at CMIP6 data as extracted for dfm and tries to see if there
is any bias in the meteo forcing

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
from scipy import stats
import matplotlib
import datetime
import sys
import pandas as pd

#===============================================================================
# %% Define some functions
#===============================================================================

def wrapTo360(lon):
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
    
def UV2WndSpeedDir(U,V,Conv):
    # U is a vector of East-West Wind speed
    # v is a vector of Nothr-South Wind Speeds
    # Conv is the convention you would like the wind in.
    # Conv = 1 means the direction the wind is going to 
    # Conv = 2 is the meteorological convention of the direction the wind is coming from
    wind_mag = np.sqrt(U**2 + V**2)
    wind_dir = np.arctan2(U/wind_mag, V/wind_mag) 
    wind_dir = wind_dir * 180/np.pi ## -111.6 degrees
    #Then you must convert this wind vector to the meteorological convention of the direction the wind is coming from:
    
    if Conv == 2:    
        wind_dir = wind_dir + 180 ## 68.38 degrees
    
    return wind_mag,wind_dir


#===============================================================================
# %% User Defined inputs
#===============================================================================

#### Directories #####
dir_data = r'Y:\PS_Cosmos\01_data\Climate\CMIP6\Meteo'
#dir_out  = r'C:\Users\kaparker\Documents\Data\Temp\Example_DFMfiles'
dir_out = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\figures\MeteoCheck';

# Models to process
model_list = ['CMCC-CM2-VHR4', 'CNRM-CM6-1-HR', 'EC-Earth_HR', 'GFDL', 'HadGEM_GC31_HH',
         'HadGEM_GC31_HM_highRes', 'HadGEM_GC31_HM_highResSST']

# Model Version to process
# ver = 'historic' 'future'
ver   = 'historic'


# Point to pull 
pull_latlon = [47.6033, -122.34]    # Seattle Tide gauge

#vas_6hrPlevPt_CMCC-CM2-VHR4_hist-1950_r1i1p1f1_gn_199312010000-199312311800.nc 
#===============================================================================
# %% Figure out name shortening
#===============================================================================
mod_short = [i for i in range(len(model_list))]

for cnt,model in enumerate(model_list):
    if model == 'CMCC-CM2-VHR4':
        mod_short[cnt] = 'CMCC'
        
    elif model == 'CNRM-CM6-1-HR':
        mod_short[cnt] = 'CNRM'
        
    elif model == 'EC-Earth_HR':
        mod_short[cnt] = 'EcEarth'
    
    elif model == 'GFDL':
        mod_short[cnt] = 'GFDL'
    
    elif model == 'HadGEM_GC31_HH':
        mod_short[cnt] = 'HadGemHH'
    
    elif model == 'HadGEM_GC31_HM_highRes':
        mod_short[cnt] = 'HadGemHM'
    
    elif model == 'HadGEM_GC31_HM_highResSST':
        mod_short[cnt] = 'HadGemHMsst'
    
    else:
         print('Model Choice not found')   
         sys.exit()

#===============================================================================
# %% Load the Data and plot 
#===============================================================================

fig1, [ax1,ax2,ax3] = matplotlib.pyplot.subplots(1, 3)
ax1.grid()
ax2.grid()
ax3.grid()


# Loop thorugh each model 
for cnt, model in enumerate(model_list):
    print('processing {}'.format(model))
    
    files = glob(os.path.join(dir_data,model,ver,'*meteo.nc'))
    
    if (len(files) != 1):
        print('More than one Meteo file found')
        sys.exit()

    ds = xr.open_mfdataset(files, engine='netcdf4')
    
    # Initialize if it's the first index
    if cnt == 0:
        pnt_TS_U = np.empty([ds['time'].shape[0],len(model_list)])
        pnt_TS_U.fill(np.nan)
        pnt_TS_V = np.empty([ds['time'].shape[0],len(model_list)])
        pnt_TS_V.fill(np.nan)
        pnt_TS_P = np.empty([ds['time'].shape[0],len(model_list)])
        pnt_TS_P.fill(np.nan)

    # modify the calendar if 360
    if (ds['time'].dt.calendar == '360_day'):
        ds = ds.convert_calendar('noleap', dim='time', align_on='year')
    
    # Pull out the time variable and Convert to a datetime variable
    time = ds['time'].values
    date = [to_datetime(val) for val in time]
    time = np.array(date)

    # Find indexes for location we want (from pull_latlon)
    ds_pnt = ds.sel(latitude=pull_latlon[0], longitude =wrapTo360(pull_latlon[1]),
                      method='nearest')
    
    # Save timeseries for model timeseries. 
    pnt_TS_U = ds_pnt['u10'].values
    pnt_TS_V = ds_pnt['v10'].values
    pnt_TS_P = ds_pnt['msl'].values
    
    # Calculate the ECDF and plot
    res = stats.ecdf(pnt_TS_U)
    ax1.plot(res.cdf.quantiles,res.cdf.probabilities)
    
    res = stats.ecdf(pnt_TS_V)
    ax2.plot(res.cdf.quantiles,res.cdf.probabilities)
    
    res = stats.ecdf(pnt_TS_P)
    ax3.plot(res.cdf.quantiles,res.cdf.probabilities)
    
    fig1, ax1 = matplotlib.pyplot.subplots(1, 1)
    ax1.plot(ds_pnt['time'],pnt_TS_P)

fig1.set_size_inches(8,3)

ax1.set_title('U Wind 10')
ax2.set_title('V Wind 10')
ax3.set_title('Sea Level Pressure')
ax1.set_ylim(0,1)
ax2.set_ylim(0,1)
ax3.set_ylim(0,1)

ax3.legend(mod_short)
fig1.savefig(os.path.join(dir_out,'CMIP6_cdf.png'), dpi=600)



## # Loop thorugh each model and plot some meteo files 
model = model_list[1]
ind_plot = np.arange(1000,1050, 1, dtype='int')

print('processing {}'.format(model))

files = glob(os.path.join(dir_data,model,ver,'*meteo.nc'))

ds = xr.open_mfdataset(files, engine='netcdf4')

for ii in ind_plot:
    t_plot = ds['time'][ii].values

    psl = np.transpose(ds['msl'].sel(time=t_plot,method='nearest').values)
    U   = np.transpose(ds['u10'].sel(time=t_plot,method='nearest').values)
    V   = np.transpose(ds['v10'].sel(time=t_plot,method='nearest').values)
    
    [Umag,Udir] = UV2WndSpeedDir(U,V,Conv = 2)  
    X, Y = np.meshgrid(ds['longitude'].values, ds['latitude'].values)
    
    fig2, [ax1,ax2] = matplotlib.pyplot.subplots(2, 1)
    fig2.set_size_inches(4,8)
    ax1.grid()
    ax2.grid()
    ax3.grid()
    
    c = ax1.pcolormesh(ds['longitude'],ds['latitude'],psl,shading = 'nearest')
    cbar = fig2.colorbar(c, ax=ax1)
    c.set_clim(101600,103000)
    cbar.ax.set_ylabel('Pressure (HPa)')
    ax1.set_title('{}'.format(t_plot))
    
    c = ax2.pcolormesh(ds['longitude'],ds['latitude'],Umag,shading = 'nearest')
    cbar = fig2.colorbar(c, ax=ax2)
    c.set_clim(0,15)
    
    ax2.quiver(X,Y,U,V, scale = 75)
    cbar.ax.set_ylabel('Umag (m/s)')
    
    #c = ax3.pcolormesh(ds['longitude'],ds['latitude'],Udir,shading = 'nearest')
    #cbar = fig2.colorbar(c, ax=ax3)
    #c.set_clim(0,360)
    #cbar.ax.set_ylabel('Udir (deg.)')
    
    fig2.savefig(os.path.join(dir_out,'Meteo_{}.png'.format(ii)), dpi=600)


    