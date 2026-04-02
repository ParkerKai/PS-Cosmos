# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:24:25 2024

This script plots wave roses for the ERA5 period 

 
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
import pandas as pd
import scipy 
import geopandas as gpd
import windrose
import sys

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'X:\PS_Cosmos\LUT\LUT_output'
dir_out = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\figures\CDF_Diff\Waves'

SLR ='000'
Mod = 'CMCC'

stat_want = 338

# Return Periods wanted 
RPs_want = [1,10,25,50,84]

#===============================================================================
# %% Define some functions
#===============================================================================
sys.path.append(r'C:\Users\kaparker\Documents\GitHub\Python\Kai_Python\General_Functions')
from Kai_MatlabTools import read_mat, read_mat_geo
from Kai_MapTools import distance_ll, Get_StationOrder

#===============================================================================
# %% Plots
#===============================================================================    
print(f'Processing: {SLR} Station: {stat_want}')

# Load the data 
# files = glob(os.path.join(dir_in,'LUt_KingPierce_CMIP6_Diff',SLR,'*.nc'))
# ds = xr.open_mfdataset(files, engine='h5netcdf', parallel=True)

# LatLon_ds = pd.DataFrame(np.transpose(np.vstack((ds['Lat'].values[0,:], ds['lon'].values[0,:]))),columns = ['Lat','Lon'])
file_in = os.path.join(dir_in, 'LUT_output_KingPierce_ERA5', 'LUT_output_KingPierce_ERA5_10mIso_1941_2023.mat')
LatLon_era5 = read_mat_geo(file_in,'lat_10mIso','lon_10mIso')


file_in = os.path.join(dir_in, 'LUT_output_KingPierce_CMIP6_historical', f'LUT_wave_timeSeries_{Mod}_his.mat')
#hs = read_mat(file_in,'Hs',stat_want)
#dm = read_mat(file_in,'Dm',stat_want)
#data_h = pd.concat([hs,dm],axis=1)
LatLon_cmip = read_mat_geo(file_in,'lat','lon')


Ind_reorder = Get_StationOrder(LatLon_era5, LatLon_cmip,dist_min = 0.3)
LatLon_cmip = LatLon_cmip.iloc[Ind_reorder,:]


