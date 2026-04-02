# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:19:49 2023

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



#===============================================================================
# %% User Defined inputs
#===============================================================================

#### Directories #####
dir_data = r'Y:\PS_Cosmos\DFM\ERA5_Results\Results'
#dir_out  = r'C:\Users\kaparker\Documents\Data\Temp\Example_DFMfiles'
dir_out = r'Y:\PS_Cosmos\DFM\ERA5_Results\Results_Combined';

# buffer
# Add the is to the front and end of the timeseries as a spinup buffer and
# to make sure the timeseries covers the full simulation time.
buff = 5;  # in days

# Years to combine
yr_start  = 1996
yr_end    = 1997

#===============================================================================
# %% Load the data 
#===============================================================================
# Water Years
yrs = np.arange(yr_start,yr_end+1,1,dtype= np.int32)

files = [os.path.join(dir_data,'WY_{}'.format(year),'WY_{}_0000_his.nc'.format(year))
         for year in yrs] 

for cnt,file in enumerate(files):

    ds = xr.open_mfdataset(file, engine='netcdf4', parallel=False)
    
    print('processing {}'.format(file))
    
    # Slice by time to remove spinup and downsample variables
    start = "{}-10-01".format(yrs[cnt]-1)
    end   = "{}-10-06".format(yrs[cnt])
    ds_trm = ds.sel(time=slice(start, end))
    ds_trm = ds_trm[{'waterlevel','station_id','station_name',
                     'station_x_coordinate','station_y_coordinate',
                     'bedlevel','x_velocity','y_velocity','time'}]
    
    wl_pull = ds_trm['waterlevel'].values
    time_pull = ds_trm['time'].values
    
    # Index of nans
    ind_nan = np.isnan(wl_pull)
    
    # Turn to an integer for space savings
    wl_pull = np.round(wl_pull*10000,decimals=0).astype(int)
    wl_pull[ind_nan] = -999999
    
    
    # Combine with the next year
    if cnt == 0:
        wl_save   = wl_pull
        time_save = time_pull
        
    else:
        
        
        # Previous chunck
        ind = time_save < (time_save[-1]- np.timedelta64(buff, 'D'))
        part1_wl = wl_save[ind,:]
        part1_t  = time_save[ind]   
        part1_nan = (part1_wl == -999999)
    
        # Average Overlap chunk between previous and this file
        inda = (time_save >= (time_save[-1]- np.timedelta64(buff, 'D')))
        indb = time_pull <= time_pull[0]+ np.timedelta64(buff, 'D')
        
        part2a_wl = wl_save[inda,:]
        part2a_t  = time_save[inda]
        part2a_nan= (part2a_wl == -999999)
        
        part2b_wl = wl_pull[indb,:]
        part2b_t  = time_pull[indb]
        part2b_nan= (part2b_wl == -999999)
        
        part2_wl = (part2a_wl + part2b_wl) / 2
        part2_wl = np.round(part2_wl,decimals=0).astype(int)
        part2_wl[np.logical_or(part2a_nan, part2b_nan)] = -999999
        part2_t  = part2a_t
        
        
        # this file
        ind = time_pull > time_pull[0]+ np.timedelta64(buff, 'D')
        
        part3_wl = wl_pull[ind,:]
        part3_t  = time_pull[ind]
        
        fig, [ax1,ax2,ax3] = matplotlib.pyplot.subplots(3, 1)
        ax1.plot(part2a_t,part2a_wl[:,1], c = 'black')
        ax2.plot(part2b_t,part2b_wl[:,1], c = 'black')
        ax3.plot(part2_t,part2_wl[:,1], c = 'black')
        
        #ax.set_ylim(np.multiply([0,2],(10**10)))
        ax2.set_ylabel('Water Level (m)')
        ax3.set_xlabel('Date (days)')
        
        ax1.set_xlim([datetime.datetime(1996, 10, 1), datetime.datetime(1996, 10, 6)])
        ax2.set_xlim([datetime.datetime(1996, 10, 1), datetime.datetime(1996, 10, 6)])
        ax3.set_xlim([datetime.datetime(1996, 10, 1), datetime.datetime(1996, 10, 6)])
        
        ax1.set_ylim(np.multiply([-5, 5],10000))
        ax2.set_ylim(np.multiply([-5, 5],10000))
        ax3.set_ylim(np.multiply([-5, 5],10000))
        
        ax1.grid()
        ax2.grid()
        ax3.grid()
        
        
            
asdf
#===============================================================================
# %% Output as yearly netc
#===============================================================================

# Unique years in timeseries
years = np.unique(pd.to_datetime(time_save).year.astype(int))

for yr in years:
    print('Outputting Year {}'.format(yr))

    # Extract water year of data    
    ind_pull =np.logical_and((time_save >= (pd.Timestamp(yr, 1, 1, 0, 0).to_datetime64())), 
                                    (time_save < (pd.Timestamp(yr+1, 1, 1, 0, 0).to_datetime64())))
    
    # Extract
    wl_pull   = wl_save[ind_pull,:]
    time_pull = time_save[ind_pull]   
    
    # define data with variable attributes
    data_vars = {'waterlevel':(['time','station'], wl_pull, 
                             {'units': 'm',
                              'ScaleFactor': 10000,
                              'standard_name':'sea_surface_height',
                              'long_name':'water level',
                              'reference':'NAVD88'}),
                'lon':(['station'], ds_trm['station_x_coordinate'].values, 
                             {'units': 'degrees_east', 
                              'standard_name':'longitude',
                              'long_name':'water original x-coodinate of station (non-snapped)',
                              'reference':'WGS84'}),
                'lat':(['station'],  ds_trm['station_y_coordinate'].values, 
                             {'units': 'degrees_north', 
                              'standard_name':'latitude',
                              'long_name':'water original y-coodinate of station (non-snapped)',
                              'reference':'WGS84'}),
                 'bedlevel':(['station'],
                             np.round(ds_trm['bedlevel'].values*10000,decimals=0).astype(int), 
                                 {'units': 'm',
                                  'ScaleFactor': 10000,
                                  'long_name':'bottom level',
                                  'reference':'NAVD88'})}
    
    
    # define coordinates
    coords= { 'station':(['station'], ds_trm['station_name'].values, 
                             {'cf_role': 'timeseries_id', 
                              'long_name':'observation station name'}),
                 'time':(['time'],  time_pull, 
                                 {'standard_name':'time'})}
    
    # define global attributes
    attrs = dict(
        processing_date=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
        source_dir = dir_data,
        source_date = ds.attrs['date_created'],
        sourc = ds.attrs['source'],
        author='Kai Parker', 
        email='kaparker@usgs.gov',
        forcing = 'ERA5')
    
    # create dataset
    ds_out = xr.Dataset(data_vars=data_vars, 
                    coords=coords, 
                    attrs=attrs)
    
    
    # Output the file  
    ds_out.to_netcdf(os.path.join(dir_out,'ERA5_wl_{}.nc'.format(yr)),
                     encoding={'waterlevel': {'dtype': 'int32', '_FillValue': -9999},
                               'station': {'dtype': 'S64', '_FillValue': -9999},
                               'bedlevel': {'dtype': 'int32', '_FillValue': -9999},
                               'time': {'dtype': 'int64', '_FillValue': -9999},
                               'lon': {'dtype': 'float64', '_FillValue': -9999},
                               'lat': {'dtype': 'float64', '_FillValue': -9999}})
        
    