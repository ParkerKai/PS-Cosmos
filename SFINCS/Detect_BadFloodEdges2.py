# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:42:10 2024

This script detects hard flood edges.  
It does this by first finding any edges over the test_edge length.
It then filters out ocean points (depth below MSL) and only grabs points that 
are located on the compute grid.

 
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
import geopandas as gpd
import rasterio
import xarray as xr 
import scipy 
from glob import glob 

from hydromt_sfincs import SfincsModel

#===============================================================================
# %% User Defined
#===============================================================================
file_in_full = r'Y:\PS_Cosmos\02_models\SFINCS\20240830_historical_new_weirs\02_Pierce\WY1942\sfincs_map.nc'
dir_in_shapefile = r'Y:\PS_Cosmos\02_models\SFINCS\20240830_historical_new_weirs\02_Pierce\shapefiles'

# Test distance
# Any edge over this length is tagged (shorter is not)
test_dist = 20

#===============================================================================
# %% Functions
#===============================================================================

#===============================================================================
# %% Read in full resolution grid
#===============================================================================
ds = xr.open_mfdataset(file_in_full ,parallel=True)

# For our case grid is cartesian and non-rotated 
X  = ds['x'].isel(n=0).values
Y = ds['y'].isel(m=0).values

# Cell corner is upper left
cornerX = ds['corner_x'].values
cornerY = ds['corner_y'].values

#===============================================================================
# %% Hydromet read of model parameters 
#===============================================================================
# Initialize SfincsModel Python class with the artifact data catalog which contains publically available data for North Italy
sfincs_root = r'Y:\PS_Cosmos\02_models\SFINCS\20240830_historical_new_weirs\02_Pierce'  # (relative) path to sfincs root

mod = SfincsModel(sfincs_root, mode="r")

# we can simply read the model results (sfincs_map.nc and sfincs_his.nc) using the read_results method
mod.read_results(fn_map=r'WY1942\sfincs_map.nc', fn_his=r'WY1942\sfincs_his.nc')

#===============================================================================
# %% Read in flood shapefile
#===============================================================================

# Load the polygon
files = glob(os.path.join(dir_in_shapefile,'*_connected.shp'))
for file in files:
    flood_layer = gpd.read_file(file)

    print(f'Processing: {file}')
    
    #===============================================================================
    # %% Find edge points
    #===============================================================================
    
    # Run through each polygon and pull edges 
    for plygn in range(len(flood_layer)):
        
        # Pull out polygon of interest
        polygon = flood_layer.iloc[plygn]['geometry']
        xx, yy = np.array(polygon.exterior.coords.xy)
        
        # Calculate the length of the polygon edge
        # Everything is nice and cartesian in UTM (meters).
        dx = np.diff(xx)
        dy = np.diff(yy)
        
        # First point isn't real because of polygon edge wrapping
        dx[0] = 0
        dy[0] = 0 
        
        # Grab locations where there is a straight line over the test distance
        horz_line = np.argwhere((dx >= test_dist) | (dx <= -test_dist))
        vert_line = np.argwhere((dy >= test_dist) | (dy <= -test_dist))
        
        # Logic to handle annoying numpy concatenating with no data matrices
        if (len(horz_line) > 0) & (len(vert_line) > 0):
            edge_pnts = np.concatenate((horz_line,vert_line),axis=0)
        elif (len(horz_line) > 0) & (len(vert_line) == 0):
            edge_pnts = horz_line
        elif (len(horz_line) == 0) & (len(vert_line) > 0):
            edge_pnts = vert_line
        else:
            edge_pnts = np.array([])
        
        # Only grab points on the compute grid 
        first = 0
        if (len(edge_pnts) > 0):
            # Coordinates of the points of interst
            x_match = np.squeeze(xx[edge_pnts])
            y_match = np.squeeze(yy[edge_pnts])
            
            # See which points are on the compute grid.  
            ind_good = np.full(edge_pnts.shape,False)
            
            for ind in range(len(edge_pnts)):
                if (len(edge_pnts) > 1):
                    polygon_vertice_xx = x_match[ind]
                    polygon_vertice_yy = y_match[ind]
                else:
                    polygon_vertice_xx = x_match
                    polygon_vertice_yy = y_match
                
                match = np.argwhere((cornerX == polygon_vertice_xx) & (cornerY == polygon_vertice_yy))
            
                # If there is a match start saving data
                if (match.size != 0):
                    ind_good[ind] = True
            
            # Update the edge point and x/y match list 
            edge_pnts = edge_pnts[ind_good]
            x_match = np.squeeze(xx[edge_pnts])
            y_match = np.squeeze(yy[edge_pnts])
                
            # Save to a growing matrix of tagged edges 
            if first == 0:
                    
                x_edges = x_match
                y_edges = y_match
                
                first = first+1
            else:
                x_edges = np.concatenate((x_edges,x_match),axis=0)
                y_edges = np.concatenate((y_edges,y_match),axis=0)
    
    
    #===============================================================================
    # %% Filter
    #===============================================================================
    
    # Filter by depth. Points below 0 are offshore. 
        
    # Get Z values for points of interest
    grid_x,grid_y = np.meshgrid(mod.grid.x,mod.grid.y, indexing='ij')
    interp = scipy.interpolate.NearestNDInterpolator(list(zip(grid_x.flatten(), grid_y.flatten())), np.transpose(mod.grid.dep.values).flatten())
    z_match = interp(x_edges,y_edges) 
    
    # Get rid of points below sealevel 
    ind_good = (z_match > 0)
    
    x_edges = x_edges[ind_good]
    y_edges = y_edges[ind_good]
    
    #===============================================================================
    # %% Export 
    #===============================================================================
    # output as a shapefile.
    gdf = gpd.GeoDataFrame( geometry=gpd.points_from_xy(x=x_edges, y=y_edges))
    
    gdf.to_file(file.replace('_connected.shp','EdgeTag.shp'))
    

