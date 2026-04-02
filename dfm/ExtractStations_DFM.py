# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:22:20 2024

This script extracts waterlevels for ERA5. It concatonates the 5 year chunks
and subsets the data to just stations you select. 


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

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
dir_in = r'D:\DFM'
dir_out = r'D:\DFM'

# Station to plot
station = ['SFINCS_1458_Snohomish',
           'SFINCS_1459_Snohomish',
           'SFINCS_1460_Snohomish',
           'stillaguamish_delta_00016',
           '1d2d_50',
           'SFINCS_1477_Snohomish',
           'fm_add_00058',
           'skagit_delta_00021',
           'SFINCS_1434_Skagit',
           '1d2d_77',
           'SFINCS_1746_Whatcom',
           'nooksack_delta_00012',
           'lummi_bay_00010',
           'birch_bay']

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

def Load_Station(files,station_string):
    # Test open
    test_open = xr.open_mfdataset(files[0], engine='netcdf4', parallel=False)
    names = test_open['station'].values
    
    ind_grab = CheckArrayFor(names,station)
    
    
    for cnt,file in enumerate(files):
        data = xr.open_mfdataset(file, engine='netcdf4', parallel=False)
        
        #Subset to station of interest
        data = data.isel(station=ind_grab)    
        
        if cnt == 0 :
            data_save = data
        else:
            data_save = xr.concat([data_save,data],dim='time')
    
    # Convert to millimeters 
    data_save['waterlevel'] = data_save['waterlevel']
    return data_save

#===============================================================================
# %% Load the data 
#===============================================================================

# ERA5
files = glob(os.path.join(dir_in,'ERA5','*.nc'))
ds_era5 = Load_Station(files,station)

asdf
ds_era5.to_netcdf(os.path.join(dir_out,'ERA5_Eric_StationPull.nc'))

