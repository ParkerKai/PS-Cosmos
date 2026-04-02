# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:36:23 2024

This script checks on DFM Water Year files

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
dir_data = r'D:\Kai\cmip6\EC-Earth_HR\future\Results_Combined\150'
dir_out = r'D:\Kai\cmip6\figs';


#===============================================================================
# %% Load the data  
#===============================================================================

dirs = glob(os.path.join(dir_data,'ERA5_*'))

ds = xr.open_mfdataset(dirs, engine='netcdf4', parallel=False)


ind_nan = 