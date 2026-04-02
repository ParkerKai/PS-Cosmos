# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 10:29:33 2025

Created on Fri Jan  3 11:56:45 2025

This script computes the DFM stations along each county so that they can be inputted into Vdatum.


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
import xarray as xr
import numpy as np
import pandas as pd 
import geopandas as gpd
import rioxarray 

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
dir_in_DEM = r'Y:\PS_Cosmos\01_data\topo_bathymetry\DEM\Pierce_King'
dir_in_wl = r'D:\DFM_Regional'

dir_in_polygon = r'Y:\PS_Cosmos\GIS\Shapefiles\general\CountyClippingPolygons'

dir_out = r'Y:\PS_Cosmos\02_models\MHW_Bathtub\Stations_4Vdatum'

county = '02_Pierce'


#===============================================================================
# %% Define some functions
#===============================================================================
sys.path.append(r'C:\Users\kaparker\Documents\GitHub\Python\Kai_Python\General_Functions')
from Kai_XarrayTools import Get_Station_index, export_pnts


#===============================================================================
# %% Read in the Station Data 
#===============================================================================
    
if county == '02_Pierce':
    ds = rioxarray.open_rasterio(os.path.join(dir_in_DEM,'Pierce_2m_AddWeir.tif'))
elif county == '01_King':
    ds = rioxarray.open_rasterio(os.path.join(dir_in_DEM,'King_2m_AddWeir.tif'))

x = ds['x'].values
y = ds['y'].values

X,Y = np.meshgrid(x,y)
X = X.flatten()
Y = Y.flatten()


geometry = gpd.points_from_xy(X, Y, crs="EPSG:32610")
   
Stats = gpd.GeoDataFrame( geometry=geometry)


# Export as csv
out = pd.DataFrame({
                    'X':Stats['geometry'].x,
                    'Y':Stats['geometry'].y,
                    'height':np.full(Stats.shape[0],0)})

out.index.name = 'ID'


out.to_csv(os.path.join(dir_out,f'DEMPnts_{county}_4Vdatum.csv'))





