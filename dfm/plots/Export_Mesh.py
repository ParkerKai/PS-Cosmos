# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:42:40 2025

@author: kaparker
"""

import xarray as xr
import geopandas as gpd
import pandas as pd

file_in  = r'Y:\PS_Cosmos\02_models\DFM_Regional\ERA5\ERA5\ERA5_000\GRD\salish_sea6_net.nc'

#===============================================================================
# %% Load the  data
#===============================================================================

ds = xr.open_mfdataset(file_in)





df = pd.DataFrame({'NetID':ds['nNetNode'].values})
NetNodes = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(ds['NetNode_x'].values, ds['NetNode_y'].values))


#===============================================================================
# %% Plot 
#===============================================================================



cnty = gpd.read_file(r'Y:\PS_Cosmos\GIS\general\Washington_Counties_with_Natural_Shoreline___washsh_area.shp')
ax1 = cnty.plot(figsize=(10, 10), color='0.8', edgecolor='black', alpha=0.4)

s1 = NetNodes.plot(ax=ax1, markersize=10, color='k',label='Statistically Significant')

ax1.set_xlim([-124,-122])
ax1.set_ylim([47,49.25])
ax1.grid()
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latittude')
ax1.set_title('DFM Model Nodes')
ax1.legend()

fig =ax1.get_figure()