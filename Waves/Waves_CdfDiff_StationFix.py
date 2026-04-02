# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:05:01 2024

This script Fixes the station order in the CDF_Diff files.
The station order was different between the CMIP6 files and the ERA5 files. So
Thngs were inconsistent. This fixes that issue.  
 
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
import numpy as np
import xarray as xr
import pickle
import sys
import glob

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'X:\PS_Cosmos\LUT\LUT_output'
dir_out = r'X:\PS_Cosmos\LUT\LUT_output\LUt_KingPierce_CMIP6_Diff'

# Model to process

SLR_list =['000','025','050','100','150','200','300']

#===============================================================================
# %% Define some functions
#===============================================================================
sys.path.append(r'C:\Users\kaparker\Documents\GitHub\Python\Kai_Python\General_Functions')
from Kai_MatlabTools import read_mat_geo
from Kai_MapTools import Get_StationOrder

#===============================================================================
# %% Load the data real LatLon locations 
#===============================================================================

# Load the data  ERA5 Lat Lon Data
# Location that is similar to the CMIP6 files is the "lat_10mIso Lat and lon"
file_in = os.path.join(dir_in, 'LUT_output_KingPierce_ERA5', 'LUT_output_KingPierce_ERA5_10mIso_1941_2023.mat')
LatLon_era5 = read_mat_geo(file_in,'lat_10mIso','lon_10mIso')

# load the cmip6 actual LatLon
file_in = os.path.join(dir_in, 'LUT_output_KingPierce_CMIP6_historical', 'LUT_wave_timeSeries_CMCC_his.mat')
#hs = read_mat(file_in,'Hs',stat_want)
#dm = read_mat(file_in,'Dm',stat_want)
#data_h = pd.concat([hs,dm],axis=1)
LatLon_cmip = read_mat_geo(file_in,'lat','lon')

# Get the station order for CMIP6 (used to reorganize )
Ind_reorder = Get_StationOrder(LatLon_era5, LatLon_cmip,dist_min = 0.3)

# Load the actual lat/lon output point
file_in = os.path.join(dir_in, 'LUT_output_KingPierce_ERA5', 'LUT_output_KingPierce_ERA5_10mIso_1941_2023.mat')
LatLon_ncPoint = read_mat_geo(file_in,'lat_ncPoint','lon_ncPoint')


#===============================================================================
# %% Calculate correction for ERA5 based on CMIP6 projections
#===============================================================================    
for SLR in SLR_list:
    files = glob.glob(os.path.join(dir_in,'LUt_KingPierce_CMIP6_Diff_DONTUSE',SLR,'*.nc'))
    
    for file in files:
        
        print(f'Processing: {file}')
        
        ds = xr.open_mfdataset(file, engine='h5netcdf', parallel=True)
        
        # Reorder and recreate
        ds['cmip_diff'].values = ds['cmip_diff'].isel(station=Ind_reorder).values
        
        # Rename Latitude to get rid of random capitol 
        ds = ds.rename({'Lat':'lat'})
        
        #Output
        ds.to_netcdf(file.replace('LUt_KingPierce_CMIP6_Diff','LUt_KingPierce_CMIP6_Diff_OrderFix'),
                     engine = 'h5netcdf')
