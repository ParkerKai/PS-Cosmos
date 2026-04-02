# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:46:24 2023

This script loads WW3 data and extracts station output 

Authors:
-------
Kai Parker
    USGS: PCMSC
   kaparker@usgs.gov

Log of edits:
-------------
April 23, 2021 - Created script
    Kai Parker

Dependencies:
-------------
Environment: analysis 
"""


__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"

# Import Modules
import xarray as xr
import os
#import mpi4py

#===============================================================================
# %% User Defined inputs
#===============================================================================

# Directory where ERA5 Data is stored
dir_era5 = r"D:\WaveData\ERA5\netcdf"

# Output directory
dir_out = r"C:\Users\kaparker\Documents\Data\Temp\ww3"


#===============================================================================
# %% Define functions
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


#===============================================================================
# %% Load the  data
#===============================================================================

########################## Load the GTSM data ########################
# Load the storm surge Netcdfs
files_in = get_files_nc(dir_era5)
files_in = files_in[0:3]
wave = xr.open_mfdataset(files_in, parallel=True, engine='netcdf4')




