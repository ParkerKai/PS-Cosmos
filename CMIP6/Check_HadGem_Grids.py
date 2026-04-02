# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:19:38 2024

This script checks the model grids for two cmip6 models to make
sure they are the same

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
import xarray as xr
import numpy as np
import matplotlib

#===============================================================================
# %% User Defined inputs
#===============================================================================

#### Directories #####
dir_cmip6 = r'Z:\CMIP6'


#===============================================================================
# %% Import the two grid files 
#===============================================================================


ds_meteo1 = xr.open_mfdataset(os.path.join(dir_cmip6,'HadGEM_GC31_HM_highResSST',
                                           'historic',
                                           'pr_3hr_HadGEM3-GC31-HM_highresSST-present_r1i1p1f1_gn_195001010130-195006302230.nc'),
                                           engine='netcdf4', parallel=False)


ds_meteo2 = xr.open_mfdataset(os.path.join(dir_cmip6,'HadGEM_GC31_HM_highRes',
                                           'historic',
                                           'pr_3hr_HadGEM3-GC31-HM_hist-1950_r1i1p1f1_gn_195001010130-195006302230.nc'),
                                           engine='netcdf4', parallel=False)



ds_orog = xr.open_mfdataset(os.path.join(dir_cmip6,'Dump','orog_fx_HadGEM3-GC31-HM_hist-1950_r1i1p1f1_gn.nc'),
                            engine='netcdf4', parallel=False)


# Pull the Lat Lon information 
Lat1 = ds_meteo1['lat'].values
Lon1 = ds_meteo1['lon'].values

Lat2 = ds_meteo2['lat'].values
Lon2 = ds_meteo2['lon'].values

Lat3 = ds_orog['lat'].values
Lon3 = ds_orog['lon'].values

np.array_equal(Lat1,Lat2)
np.array_equal(Lat1,Lat3)

np.array_equal(Lon1,Lon2)
np.array_equal(Lon1,Lon3)



