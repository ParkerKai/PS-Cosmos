# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:52:24 2024
This script plots wave setup and beach slope parameters for King and Pierce counties
Developed as part of the PS-Cosmos project.
Wave parameters came from Anita, based on Sean's LUT
Wave setup calclulated using "Calc_WaveSetup.m"


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
#import hvplot.pandas
#import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import geopandas as gpd
#import holoviews

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
dir_in = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\WaveSetup'
dir_out = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\WaveSetup'

# Model to process
Region = 'Pierce'

#===============================================================================
# %% Define some functions
#===============================================================================
def FillNans(data,Field,FillVal,SetVal):
    out = data.where(data['setup'] != -9999)
    out = out.fillna(SetVal)
    
    return out

def CalcStats(data,axis):
    d = {'Mean': np.mean(data,axis=axis),
         'Max': np.max(data,axis=axis),
         'Std': np.std(data,axis=axis),
         'Q99': np.quantile(data,.99,axis=axis),
         'Q95': np.quantile(data,.95,axis=axis)}
    
    # Convert to pandas dataframe
    out = pd.DataFrame(data=d)
    
    return out
#===============================================================================
# %% Load the data 
#===============================================================================
data = xr.open_mfdataset(os.path.join(dir_in,'WaveSetup_{}.nc'.format(Region)),
                         engine='netcdf4', parallel=False, mask_and_scale=False)


# Some post processing
#data = FillNans(data,'setup',-9999,0)
#data = FillNans(data,'R2',-9999,0)

setup = data['setup'].values
setup[setup == -9999] = 0

setup_stat = CalcStats(setup,0)
#append on Beta 
setup_stat = pd.concat([setup_stat,pd.DataFrame({'Beta':data['Beta'].values})],axis=1)
setup_stat = pd.concat([setup_stat,pd.DataFrame({'StationID':data['station'].values})],axis=1)


geometry = gpd.points_from_xy(x=data['X'], y=data['Y'],crs = 'EPSG:32610')
setup_out = gpd.GeoDataFrame(data =setup_stat, geometry=geometry)



#===============================================================================
# %% Export the data 
#===============================================================================
setup_out.to_file(os.path.join(dir_out,'SetupStats.shp'))



