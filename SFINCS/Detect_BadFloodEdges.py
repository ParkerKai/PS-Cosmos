# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:42:10 2024

This script depects hard flood edges.  
 
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
import pandas as pd
import geopandas as gpd
import matplotlib
import rasterio
import xarray as xr 

from hydromt_sfincs import SfincsModel
from hydromt_sfincs import utils

#===============================================================================
# %% User Defined
#===============================================================================
file_in_full = r'Y:\PS_Cosmos\02_models\SFINCS\20240830_historical_new_weirs_ModelBuild\01_King\WY1942\sfincs_map.nc'
file_in_2m = r'Y:\PS_Cosmos\02_models\SFINCS\20240830_historical_new_weirs\01_King\downscaled_2m\connection_RP001_2m.tif'

dir_out = r'Y:\PS_Cosmos\02_models\SFINCS\20240830_historical_new_weirs\01_King\shapefiles'

# Filter Depth
#Fitler out points below MSL (removes all ocean points)
Filter_depth = 1

# Fitler out non-connected points
# This filters out solitary points which are not connected to another found point on the raster.
# Lots of flood polygon points happen to intersect the raster layer. So the idea is to only provide edges (two concurrent points)
# This removes incorrectly tagged solitary points but also removes some points on hard edges.
Filter_adjacent = 1

#===============================================================================
# %% Functions
#===============================================================================

def get_raster_geometry(file,offset):
    # Offset follows Rasterio's convention. Options are  center,ul, ur, ll, lr.
    
    with rasterio.open(file) as src:
        band1 = src.read(1)
        print('Band1 has shape', band1.shape)
        height = band1.shape[0]
        width = band1.shape[1]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset=offset)
        
        
        X = np.array(xs)
        Y = np.array(ys)

        X_out = np.reshape(X,rows.shape)
        Y_out = np.reshape(Y,cols.shape)
        
    return X_out,Y_out 

def get_adjacent_cells(grid_X, grid_Y, x_ind, y_ind):
    # This finds the adjacent indices for a cell in a raster.
    # Output is a np.array of the x,y indices.
    
    # grid_X: grid values (x compoent). in 2D array format 
    # grid_Y: grid values (y compoent). in 2D array format 
    # x_ind:  Index of x coordinate we are getting adjacent cells for.  
    # Y_ind:  Index of y coordinate we are getting adjacent cells for.
    cnt = 0
    for x,y in [(x_ind+i,y_ind+j) for i in (-1,0,1) for j in (-1,0,1) if i != 0 or j != 0]:
        print((x,y))
        if (x >= 0) & (y>=0) & (x <grid_X.shape[0]) & (y <grid_Y.shape[0]): 
            if (cnt == 0):
                result = np.array((x,y))
                cnt = cnt+1
            
            # Concatenate to save 
            else:
                result = np.vstack((result, np.array((x,y))))
            
    if 'result' not in locals():
        print('No adjacent cell found')
        print ('most likely issue is that coordinates not indices used for x_ind and y_ind')
        
    return result


#===============================================================================
# %% Read raster at subgridded resolution
#===============================================================================

# dataset = rasterio.open(file_in_2m)
# band1 = dataset.read(1)

# X_sg,Y_sg = get_raster_geometry(file_in_2m,'center')

# Load the polygon
flood_layer = gpd.read_file(r'Y:\PS_Cosmos\02_models\SFINCS\20240830_historical_new_weirs\01_King\shapefiles\FloodingRP100_connected.shp')
polygon = flood_layer.iloc[1238]['geometry']

#===============================================================================
# %% Hydromet read of model parameters 
#===============================================================================
# Initialize SfincsModel Python class with the artifact data catalog which contains publically available data for North Italy
sfincs_root = r'Y:\PS_Cosmos\02_models\SFINCS\20240830_historical_new_weirs_ModelBuild\01_King'  # (relative) path to sfincs root


mod = SfincsModel(sfincs_root, mode="r")

# we can simply read the model results (sfincs_map.nc and sfincs_his.nc) using the read_results method
mod.read_results(fn_map=r'WY1942\sfincs_map.nc', fn_his=r'WY1942\sfincs_his.nc')

# the following variables have been found
list(mod.results.keys())


h = mod.results["zsmax"].where(mod.results["zsmax"] > 0)
h.attrs.update(long_name="water depth", unit="m")

#===============================================================================
# %% Read SFINCS outputs
#===============================================================================

ds = xr.open_mfdataset(file_in_full,parallel=True)

# For our case grid is cartesian and non-rotated 
X  = ds['x'].isel(n=0).values
Y = ds['y'].isel(m=0).values
Z = mod.grid.dep.values

# Cell corner is upper left
cornerX = ds['corner_x'].values
cornerY = ds['corner_y'].values


#===============================================================================
# %% Find matches between polygon and grid edges 
#===============================================================================
xx, yy = np.array(polygon.exterior.coords.xy)

cnt = 0 
for ind in range(len(xx)):
    polygon_vertice_xx = xx[ind]
    polygon_vertice_yy = yy[ind]
    
    match = np.argwhere((cornerX == polygon_vertice_xx) & (cornerY == polygon_vertice_yy))

    # If there is a match start saving data
    if (match.size != 0):
        # Initialize the array if it's the first one 
        if cnt == 0 :
            match_pnts = match
            cnt = cnt+1
        
        # Concatenate to save 
        else:
            match_pnts = np.vstack((match_pnts, match))

#===============================================================================
# %% Filter
#===============================================================================

# Filter by depth. Points below 0 are offshore. 
if (Filter_depth == 1):
    
    z_match = Z[match_pnts[:,0]-1,match_pnts[:,1]-1]
    
    ind_good = (z_match > 0)
    match_pnts = match_pnts[ind_good,:]


# Filter by connectivity.  Only want points next to another.
#  (we want edges rather than single random intersections.)
if (Filter_adjacent == 1):
    
    ind_good = np.full(match_pnts.shape[0],True)
    for ii in range(len(match_pnts)):
        # Get adjacent cells to this point
        adjacent_cells = get_adjacent_cells(cornerX, cornerY, match_pnts[ii,0], match_pnts[ii,1])
        
        # Are these adjacent cells in the match_points list . 
        # Go through each adjacent cell and see if it is found in the match_pnt list.
        test = np.full(adjacent_cells.shape[0],False)
        for jj in range(len(adjacent_cells)):
            test[jj] = np.sum((np.isin(match_pnts[:,0], adjacent_cells[jj,0])) & (np.isin(match_pnts[:,1], adjacent_cells[jj,1])))
        
        
        # If none of the adjacent cells are in the match_pnts list then say this isn't a connected cell 
        if np.sum(test) == 0 :
            ind_good[ii] = False
        
    
    match_pnts = match_pnts[ind_good,:]



# Get the locations where polygon matches the grid.
x_match = cornerX[match_pnts[:,0],match_pnts[:,1]]
y_match = cornerY[match_pnts[:,0],match_pnts[:,1]]
z_match = Z[match_pnts[:,0]-1,match_pnts[:,1]-1]

#===============================================================================
# %% Export 
#===============================================================================

gdf = gpd.GeoDataFrame( geometry=gpd.points_from_xy(x=x_match, y=y_match))



gdf.to_file(os.path.join(dir_out,'KingRP100_EdgeTag_filtered2.shp'))


# fig = matplotlib.pyplot.subplots(1, 1)
# fig[0].set_size_inches(6, 6)
# ax = fig[0].get_axes()
# ax = ax[0]

# ax.scatter(X,Y,10,'k',label='Cell Center')
# ax.scatter(cornerX,cornerY,10,'r',label='Cell Corner')
# ax.grid()
# ax.legend()



