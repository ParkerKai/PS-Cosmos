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
import datetime
import pandas as pd
import math 

#===============================================================================
# %% Define some functions
#===============================================================================



#===============================================================================
# %% User Defined inputs
#===============================================================================

#### Directories #####
dir_data = r'Y:\PS_Cosmos\02_models\DFM_Regional\ERA5_Results\Results'
#dir_out  = r'C:\Users\kaparker\Documents\Data\Temp\Example_DFMfiles'
dir_out = r'Y:\PS_Cosmos\02_models\DFM_Regional\ERA5_Results\Results_Combined';

# buffer
# Add the is to the front and end of the timeseries as a spinup buffer and
# to make sure the timeseries covers the full simulation time.
buff = 5;  # in days

# Years to combine
#yr_start  = 1951
#yr_end    = 2013

# Chunk length
# Size in years of output netcdf chunks
chunk_yr = 5


#===============================================================================
# %% Figure out the chunks
#===============================================================================
# Break into $chunk_yr sections for outputting and loading. This was done for
# memory reasons (can't load all the data). Tricky because the WY sims have a 
# $buff overlap that we want to average. So if an overlap period is available 
# (all years other than the start and end), need to load an additional year to
# cover the overlap and allow an averaging.

dirs = glob(os.path.join(dir_data,'WY_*'))

year = np.empty(len(dirs),dtype=int)
for cnt,direc in enumerate(dirs):
    temp = direc.split('WY_')
    year[cnt] = np.asarray(temp[1],dtype=int)
 
year = np.sort(year)

yr_start = math.floor(year[0] / 5.0) * 5
yr_end  = year[-1]

yr_breaks = np.arange(yr_start, yr_end, chunk_yr, dtype=int)
# if (yr_breaks[-1] != yr_end):
#     yr_breaks = np.append(yr_breaks,yr_end)

break_start = np.empty(yr_breaks.shape[0]-1,dtype=int)
break_end   = np.empty(yr_breaks.shape[0]-1,dtype=int)
break2_start = np.empty(yr_breaks.shape[0]-1,dtype=int)
break2_end   = np.empty(yr_breaks.shape[0]-1,dtype=int)
for ii in range(yr_breaks.shape[0]-1):
    # First chunk must include the start year (no overlap needed)
    if (ii == 0):
        break_start[ii] = yr_breaks[ii]
        break_end[ii]   = yr_breaks[ii+1]+1
        
        break2_start[ii] = yr_breaks[ii]
        break2_end[ii]   = yr_breaks[ii+1]
        
    # End chunk must include the end year (no overlap)
    elif (ii == yr_breaks.shape[0]-1):
        break_start[ii] = yr_breaks[ii]-1
        break_end[ii]   = yr_breaks[ii+1]
        
        break2_start[ii] = yr_breaks[ii]
        break2_end[ii]   = yr_breaks[ii+1]
        
    # Intermediate years include one year on either side to account for overlap
    else:
        break_start[ii] = yr_breaks[ii]-1
        break_end[ii]   = yr_breaks[ii+1]+1
    
        break2_start[ii] = yr_breaks[ii]
        break2_end[ii]   = yr_breaks[ii+1]


# Hard coded end (pain in the butt)
#break_start = np.append(break_start,2019)
#break_end   = np.append(break_end,2021)

#break2_start = np.append(break2_start,2020)
#break2_end   = np.append(break2_end,2021)


# break_start[0] = 1941
# break2_start[0] = 1941

#break_start = np.append(break_start,2009)
#break_end   = np.append(break_end,2013)

#break2_start = np.append(break2_start,2010) 
#break2_end   = np.append(break2_end,2013)

if (break_start[0] != year[0]):
    break_start[0]  = year[0]
    break2_start[0] = year[0]

if (break_end[0] != year[-1]):
    break_end   = np.append(break_end,year[-1])
    break2_end   = np.append(break2_end,year[-1])

    break_start = np.append(break_start,yr_breaks[-1]-1)
    break2_start = np.append(break2_start,yr_breaks[-1]) 


#===============================================================================
# %% Load the data 
#===============================================================================
if (os.path.exists(dir_out) == False):
    os.mkdir(dir_out)

for ii in range(yr_breaks.shape[0]):
    
    # Water Years
    yrs = np.arange(break_start[ii],break_end[ii]+1,1,dtype= np.int32)
    
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
        wl_pull[ind_nan] = -999999/10000

        # Turn to an integer for space savings
        wl_pull = np.round(wl_pull*10000,decimals=0).astype(int)
        
        # Combine with the next year
        if cnt == 0:
            wl_save   = wl_pull
            time_save = time_pull
            
        else:
                    
            
            # Previous chunk
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
            
            # Combine for saving
            wl_save = np.vstack((part1_wl,part2_wl,part3_wl))
            time_save = np.hstack((part1_t,part2_t,part3_t))
            
            

#===============================================================================
# %% Output as  netc
#===============================================================================

    print('Outputting Chunk {}'.format(yr_breaks[ii]))
    
    # For last timestep grab all the data
    if ii == yr_breaks.shape[0]-1:
        # Extract calendar year of data    
        ind_pull =(time_save >= (pd.Timestamp(break2_start[ii], 1, 1, 0, 0).to_datetime64()))

    # Otherwise trim to the calendar year
    else:
        # Extract calendar year of data    
        ind_pull =np.logical_and((time_save >= (pd.Timestamp(break2_start[ii], 1, 1, 0, 0).to_datetime64())), 
                                        (time_save < (pd.Timestamp(break2_end[ii], 1, 1, 0, 0).to_datetime64())))
        
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
    ds_out.to_netcdf(os.path.join(dir_out,'ERA5_wl_{}.nc'.format(yr_breaks[ii])),
                     encoding={'waterlevel': {'dtype': 'int32', '_FillValue': -999999},
                               'station': {'dtype': 'S64', '_FillValue': -999999},
                               'bedlevel': {'dtype': 'int32', '_FillValue': -999999},
                               'time': {'dtype': 'int64', '_FillValue': -999999},
                               'lon': {'dtype': 'float64', '_FillValue': -999999},
                               'lat': {'dtype': 'float64', '_FillValue': -999999}})
        
    