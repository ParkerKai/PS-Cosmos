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
dir_data = r'Z:\CMIP6'
#dir_out  = r'C:\Users\kaparker\Documents\Data\Temp\Example_DFMfiles'
dir_out = r'Y:\PS_Cosmos\CMIP6';

# Model to process
# model = 'CMCC-CM2-VHR4' 'CNRM-CM6-1-HR' 'EC-Earth_HR' 'GFDL' 'HadGEM_GC31_HH'
#         'HadGEM_GC31_HM_highRes' 'HadGEM_GC31_HM_highResSST'
model = 'EC-Earth_HR'

# Model Version to process
# ver = 'historic' 'future'
ver   = 'future'

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
# only want data for a timeperiod we have all the dat
# Also want a consistent time sampling
start_time = np.empty(len(var_list), dtype = datetime.datetime)
end_time   = np.empty(len(var_list), dtype = datetime.datetime)
dt         = np.empty(len(var_list), dtype = datetime.timedelta)
for cnt,var in enumerate(var_list):
    files = glob(os.path.join(dir_data,model,ver,'{}*'.format(var)))
    ds = xr.open_mfdataset(files, engine='netcdf4', parallel=False)
    
    # modify the calendar if 360
    if (ds['time'].dt.calendar == '360_day'):
        ds = ds.convert_calendar('noleap', dim='time', align_on='year')
    
    # Pull out the time varibale
    time = ds['time'].values
    
    # Convert to a datetime variable
    # temp = ds.indexes['time'].to_datetimeindex() # possibly better solution using pandas timestamp
    date = [to_datetime(val) for val in time]
    time = np.array(date)
    
    start_time[cnt] = time[0]
    end_time[cnt]   = time[-1]
    dt[cnt]         = time[1]-time[0]    
    
# Create a new (and consistent) timeseries
# Only do overlapping time.
t = np.arange(np.max(start_time), np.min(end_time), np.median(dt))
date = [to_datetime(val) for val in t]
time_wnt = np.array(date)

#===============================================================================
# %% Load the Data
#===============================================================================

# Loop thorugh each variable 
for var in var_list:
    print('processing {}'.format(var))
    
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
    # %% Interpolate to timeperiod of interest
    #===============================================================================
    print('Interpolating in Time')
    var_intrp_t = np.empty((len(time_wnt),len(ds_lat),len(ds_lon)),dtype = 'float32')
    for ll in range(len(ds_lat)): 
        for jj in range(len(ds_lon)):
            
            var_intrp_t[:,ll,jj] = interp_datetime(time_wnt,time,Var_extract[:,ll,jj])
    
    #===============================================================================
    # %% Extract down to just the region of interest
    #===============================================================================
    # Create a Grid.
    lat_vec = np.arange(lat_lim[0],lat_lim[1],lat_d, dtype='float64')
    lon_vec = np.arange(lon_lim[0],lon_lim[1],lon_d, dtype='float64')
    
    xv, yv = np.meshgrid(lon_vec, lat_vec, indexing='ij')
    
    print('Interpolating in Space')
    var_intrp_ts = np.empty((len(time_wnt),len(lon_vec),len(lat_vec)),dtype = 'float64')
    for tt in range(len(time_wnt)):
        # Interpolate 
        var_intrp_ts[tt,:,:] = griddata((ds_longrid.flatten(),ds_latgrid.flatten()),var_intrp_t[tt,:,:].flatten(),
                           (xv, yv), method='linear')
        
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
    
    years = np.arange(yr1,yr2, 1)

    for yr in years:
        print('Outputting Year {}'.format(yr))

        # Extract water year of data
        ind_wy = (time_wnt >= (datetime.datetime(yr-1, 10, 1, 0, 0)- datetime.timedelta(days=buff))) & (time_wnt <= (datetime.datetime(yr, 9, 30, 0, 0)+datetime.timedelta(days=buff)))
        
        var_intrp_yr = var_intrp_ts[ind_wy,:,:]
        time_yr      = time_wnt[ind_wy]
        
        # add another datapoint to make sure covers the whole period
        ind_last = np.max(np.where(ind_wy)[0])
        if (ind_last < ind_wy.size):
            ind_wy[ind_last+1] = True
        
        # define data with variable attributes
        if var == 'uas':
            data_vars = {'u10':(['time','x','y'], var_intrp_yr, 
                                     {'units': 'm/s', 
                                      'standard_name':'eastward_wind'})}
        elif var == 'vas':
            data_vars = {'v10':(['time','x','y'], var_intrp_yr, 
                                     {'units': 'm/s', 
                                      'standard_name':'northward_wind'})}
        elif var == 'psl':
                data_vars = {'slp':(['time','x','y'], var_intrp_yr, 
                                         {'units': 'Pa', 
                                          'standard_name':'air_pressure_fixed_height'})}
            
        # define coordinates
        coords=dict(
            lon  =(["x"], lon_vec),
            lat  =(["y"], lat_vec),
            time = time_yr)
        
        # define global attributes
        attrs = dict(
            creation_date=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), 
            author='Kai Parker', 
            email='kaparker@usgs.gov',
            Cmip6 = model,
            timeperiod = ver)
        
        # create dataset
        ds_out = xr.Dataset(data_vars=data_vars, 
                        coords=coords, 
                        attrs=attrs)
        
        # Reset some attributes 
        ds_out['lon'].attrs = {"units": 'degrees_north',
                           'standard_name': 'longitude'}
        ds_out['lat'].attrs = {"units": 'degrees_east',
                           'standard_name': 'latitude'}
        
        
        # Output the file  
        if var == 'uas':   
            ds_out.to_netcdf(os.path.join(dir_out,model,ver,
                                          'psfm_{}_meteo_u_WY{}.nc'.format(mod_short,yr)),
                             encoding={'u10': {'dtype': 'float32', '_FillValue': -9999},
                                       'time': {'dtype': 'int64', '_FillValue': -9999},
                                       'lon': {'dtype': 'float32', '_FillValue': -9999},
                                       'lat': {'dtype': 'float32', '_FillValue': -9999}})
        elif var == 'vas':
            ds_out.to_netcdf(os.path.join(dir_out,model,ver,
                                          'psfm_{}_meteo_v_WY{}.nc'.format(mod_short,yr)),
                             encoding={'v10': {'dtype': 'float32', '_FillValue': -9999},
                                       'time': {'dtype': 'int64', '_FillValue': -9999},
                                       'lon': {'dtype': 'float32', '_FillValue': -9999},
                                       'lat': {'dtype': 'float32', '_FillValue': -9999}})
        elif var == 'psl':
            ds_out.to_netcdf(os.path.join(dir_out,model,ver,
                                          'psfm_{}_meteo_psl_WY{}.nc'.format(mod_short,yr)),
                             encoding={'slp': {'dtype': 'float32', '_FillValue': -9999},
                                       'time': {'dtype': 'int64', '_FillValue': -9999},
                                       'lon': {'dtype': 'float32', '_FillValue': -9999},
                                       'lat': {'dtype': 'float32', '_FillValue': -9999}})
            
    #===============================================================================
    # %% Save data for combining later
    #===============================================================================
    if var == 'uas':   
        uas = var_intrp_ts
        
    elif var == 'vas':
        vas = var_intrp_ts

    elif var == 'psl':
        psl = var_intrp_ts

#===============================================================================
# %% Output of combined final file
#===============================================================================
# define data with variable attributes

data_vars = {'u':(['time','x','y'], uas, 
                         {'units': 'm/s', 
                          'standard_name':'eastward_wind'}),
             'v':(['time','x','y'], vas, 
                         {'units': 'm/s', 
                          'standard_name':'eastward_wind'}),
             'slp':(['time','x','y'], psl, 
                             {'units': 'Pa', 
                              'standard_name':'air_pressure'})}

# define coordinates
coords=dict(
    lon  =(["x"], lon_vec),
    lat  =(["y"], lat_vec),
    time = time_wnt)

# define global attributes
attrs = dict(
    creation_date=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), 
    author='Kai Parker', 
    email='kaparker@usgs.gov',
    Cmip6 = model,
    timeperiod = ver)

# create dataset
ds_out = xr.Dataset(data_vars=data_vars, 
                coords=coords, 
                attrs=attrs)

# Reset some attributes 
ds_out['lon'].attrs = {"units": 'degrees_north',
                   'standard_name': 'longitude'}
ds_out['lat'].attrs = {"units": 'degrees_east',
                   'standard_name': 'latitude'}

# Output the file  
ds_out.to_netcdf(os.path.join(dir_out,model,ver,
                              'psfm_{}_meteo.nc'.format(mod_short)),
                 encoding={'u': {'dtype': 'float32', '_FillValue': -9999},
                           'v': {'dtype': 'float32', '_FillValue': -9999},
                           'slp': {'dtype': 'float32', '_FillValue': -9999},
                           'time': {'dtype': 'int64', '_FillValue': -9999},
                           'lon': {'dtype': 'float32', '_FillValue': -9999},
                           'lat': {'dtype': 'float32', '_FillValue': -9999}})
    
    