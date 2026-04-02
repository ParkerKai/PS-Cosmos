# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 21:32:14 2025

This script exports the SFINCS stations as lat / lon so that they can be
converted to MHW with VDATUM.



@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"





#===============================================================================
# %% Import Modules
#===============================================================================
import sys
import os
import scipy
import xarray as xr
import numpy as np
import pandas as pd 
# from sklearn.metrics import root_mean_squared_error,r2_score,mean_absolute_error
# from sklearn.linear_model import LinearRegression
# import matplotlib
from glob import glob 
import geopandas as gpd
# import pandas as pd
# from scipy.interpolate import interp1d
# import h5py
import h5py

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'Y:\PS_Cosmos\02_models\SFINCS\20250122_synthetic_future_meanchange_100yr_Intel\02_Pierce\000\SY000'
dir_out = r'Y:\PS_Cosmos\02_models\SFINCS\Daily_MHHW\Stats_4Vdatum'


county= '02_Pierce'

#===============================================================================
# %% Define some functions
#===============================================================================



#===============================================================================
# %% Main
#===============================================================================
ds = xr.open_mfdataset(os.path.join(dir_in,'sfincs_bndbzs.nc'), engine='h5netcdf', parallel=True,
                       chunks={"time": -1, "station": 1}, decode_times=False)


x = ds['x'].values
y = ds['y'].values
    
    
geometry = gpd.points_from_xy(x, y, crs="EPSG:32610")

Stats = gpd.GeoDataFrame(geometry=geometry)


# Export as csv
out = pd.DataFrame({
                    'x':Stats['geometry'].x,
                    'y':Stats['geometry'].y,
                    'height':np.full(Stats.shape[0],0)})

out.index.name = 'ID'


out.to_csv(os.path.join(dir_out,f'DFMPnts_{county}_4Vdatum.csv'))



