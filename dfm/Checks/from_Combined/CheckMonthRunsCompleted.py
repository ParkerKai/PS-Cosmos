# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:47:28 2024

This script sees if all the monthly cmip6 diff files were created 
 
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


#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_out = r'C:\Users\kai\Documents\KaiRuns\DFM\cdf_diff_combined'
dir_in_era5 = r'C:\Users\kai\Documents\KaiRuns\DFM\ERA5_050'
dir_in_diff = r'C:\Users\kai\Documents\KaiRuns\DFM\cdf_diff\050'

#===============================================================================
# %% Define some functions
#===============================================================================

def getfile_date(files):
    year = np.full(len(files),-9999,dtype='int32')
    mnth = np.full(len(files),-9999,dtype='int32')
    for cnt,file in enumerate(files):
        file = os.path.basename(file)
        year[cnt] = np.asarray(file[-10:-6],dtype='int32')
        mnth[cnt] = np.asarray(file[-5:-3],dtype='int32')
        
    return year,mnth

def get_MissingRuns(files):
    
    # Return date/month for runs that completed
    file_yr,file_mn = getfile_date(files)
    file_date = np.transpose(np.vstack((file_yr,file_mn)))
    
    # Compile runs that should have completed
    yr_strt = file_yr.min()
    yr_end  = file_yr.max()
    years = np.arange(yr_strt,yr_end+1,dtype='int32')
    
    want_yr = np.full(years.size*12,-9999,dtype='int32')
    for cnt,yr in enumerate(years):    
        want_yr[cnt*12:cnt*12+12] = np.full(12,yr,dtype='int32')
        
    want_mn = np.full(years.size*12,-9999,dtype='int32')
    for cnt in range(years.size):    
        want_mn[cnt*12:cnt*12+12] = np.arange(1,12+1,dtype='int32')    
    
    want_date = np.transpose(np.vstack((want_yr,want_mn)))
    
    # Check if year,month runs are in the wanted array
    run_exists = np.full(want_date.shape[0],False)
    for ii in range(want_date.shape[0]):
        run_exists[ii] = any(np.all(file_date == want_date[ii,:], axis=1))
    
    # We also don't want any files that are too small
    for file in files:
        size = os.path.getsize(file)
        if size < 3e+8:
            run_exists[ii] = False
        
    runs_want = want_date[run_exists==False,:]
    
    #num_runs = run_exists.shape[0]
    #num_failed = runs_want.shape[0]
    #print(f'{num_failed} Runs out of {num_runs} need to be re-run')
    
    return runs_want



#===============================================================================
# %% Main
#===============================================================================
    

#===============================================================================
# %% Figure out missing months and throw warning if insufficent
#===============================================================================

# get file list
files = glob(os.path.join(dir_in_diff,'*.nc'))
    
# Get files that need to be run
runs_want = get_MissingRuns(files)
print('Missing RUns: ')
print(runs_want)

