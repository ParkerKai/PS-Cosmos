# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:41:03 2023

This function finds the edge of a channel that was hydroflattened.
It needs a "thalweg" shapefile and transects along that thalweg.  

NOte that this does not find the channel edge as defined by a leve or high point
Rather it finds where the side of the channel intersects the hydroflattened bathymetry.

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
import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np
from shapely import geometry
import matplotlib
from scipy.signal import find_peaks
from scipy.interpolate import InterpolatedUnivariateSpline
from shapely.geometry import LineString
from shapely.geometry import Point
import sys

#===============================================================================
# %% User Defined inputs
#===============================================================================

# directory where the GIs Layers reside
#dir_gis = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\GIS\Shapefiles\StreamBathy'
dir_gis = r'Y:\PS_Cosmos\GIS\StreamBathy'

# directory where the Bathymetry resides
#dir_bathy = r'D:\BathyChunks\Puget_Sound_CoNED_Topobathy_DEM_1m'
dir_bathy = r'Y:\PS_Cosmos\01_data\topo_bathymetry\DEM\PugetSound_DEM\Puget_Sound_CoNED_Topobathy_DEM_1m'

name = 'Duwamish'

# Define coordinate systems
epsg_wgs = "EPSG:4326" # World Geodetic System 1984
epsg_utm = "EPSG:26910" # UTM Zone 10N (West coast of the US.)

#Numerics
# Spline Smoothing
# cranking this up will smooth the initial bathymetry before trying to find edges.  
# More smoothing makes it more likely to find the correct edge (not spurious edges)
# But too much smoothing will mean you don't find an edge.
sp_sm = 0.05

# Peak prominence
# Similar to spline smoothing
# higher peak prominence tollerance will mean more likely to find the correct peak
# But too high means you won't be able to find the edge of the channel.
pk_prom = 0.05

# Leaving both sp_sm and pk_prom low leads to the most robust running (code doesn't fail).
# But requires more after edits to the edge line.  

#===============================================================================
# %% Define some functions
#===============================================================================
sys.path.append(r'C:\Users\kaparker\Documents\GitHub\Python\Kai_Python\General_Functions')
from Kai_GeoTools import redistribute_vertices,reverse_geom

#===============================================================================
# %% Load the Data
#===============================================================================

# Gis features of the river 
thalweg = gpd.read_file(os.path.join(dir_gis,name,'{}_thalweg.shp'.format(name)))
trans = gpd.read_file(os.path.join(dir_gis,name,'{}_transects.shp'.format(name)))

# Slope tiff.
bathy = rasterio.open(os.path.join(dir_bathy,'Puget_Sound_CoNED_Topobathy_DEM_1m.tif'))

#===============================================================================
# %% Get bathy information from the CONED raster 
#===============================================================================

for tt in range(len(trans)):
    pull = trans.loc[tt]
    line = pull.geometry

    # Make sure that the transect is the same direction as the previous.
    # Done by finding the distance to the previous leading point. 
    # So transect length must be larger than transect spacing.
    pnt_1 = gpd.GeoSeries(Point(line.coords.xy[0][0],line.coords.xy[1][0]),crs = trans.crs) 
    pnt_2 = gpd.GeoSeries(Point(line.coords.xy[0][1],line.coords.xy[1][1]),crs = trans.crs) 
    
    
    if (tt > 0):
        
        dist1 = pnt_1.distance(pnt_prev)
        dist2 = pnt_2.distance(pnt_prev)
        
        # Reverse the line
        if (dist2[0] < dist1[0]):
            line = reverse_geom(line)
            print('transect {} is reversed'.format(tt))
            trans.geometry[tt] = line
            
    # Save the endpoint
    pnt_prev = gpd.GeoSeries(Point(line.coords.xy[0][0],line.coords.xy[1][0]),crs = trans.crs)

    # Add extra points to the line
    distance_delta = 1
    distances = np.arange(0, np.round_(line.length), distance_delta)

    # Redestribute vertices to every 1 m on a projected MultiLineString
    line = redistribute_vertices(line, 1)

    coord_list = np.stack((line.coords.xy[0],line.coords.xy[1]),axis=1)
    
    trans_z = [x for x in bathy.sample(coord_list)]
    trans_z = np.asarray(trans_z)

    # add another point for plotting
    distances = np.append(distances,distances[-1]+distance_delta)
    
    # smooth the bathy series a bit
    spl = InterpolatedUnivariateSpline(distances, trans_z)
    spl.set_smoothing_factor(s=sp_sm)
    trans_z = spl(distances)
    
    derivative = spl.derivative(n=2)
    trans_diff = derivative(distances)
    
    # Along axis gradient
    # trans_diff = np.asarray(0,dtype='float32')
    # trans_diff = np.append(trans_diff,np.diff(trans_z,n=1, axis=0))
    # trans_diff = np.absolute(trans_diff)
    
    # FInd peaks in elevation change
    peaks,properties  = find_peaks(trans_diff, prominence=pk_prom)
    
    # Points out representing the edge of the channel
    # Reduce if can't find 2
    if len(peaks) != 2:
        # Reduce to 2 peaks
        # Peaks nearest to the center
        mid_pnt = len(distances)/2
        low = peaks[peaks < mid_pnt]
        high = peaks[peaks > mid_pnt]  
        
        if (len(low)==0) or (len(high)==0):
            sys.exit('No edge found for transect {}'.format(tt))

            # ('No edge found for transect {}'.format(tt))
            
        peaks =np.array([low.max(), high.min()])
    
    # Save as a geoseries
    # point 1 is the minimum to maintain side of the channel.
    
    temp1 = gpd.GeoSeries(geometry.Point(coord_list[peaks.min()]),crs = trans.crs)
    temp2 = gpd.GeoSeries(geometry.Point(coord_list[peaks.max()]),crs = trans.crs)
            
    # Save as a geoseries
    if (tt == 0):
        point_1 = temp1
        point_2 = temp2
    else:
        point_1 = pd.concat([point_1,temp1])
        point_2 = pd.concat([point_2,temp2])
    

    fig, [ax,ax2] = matplotlib.pyplot.subplots(2, 1)
    ax.plot(distances,trans_z,'k')
    ax.plot(distances[peaks],trans_z[peaks], "x", color = 'red')
    #ax.set_xlim(pd.Timestamp('2000-08-01'),pd.Timestamp('2001-08-01'))
    #ax.set_xlim(100,200)
    ax.grid()
    ax.set_title('Transect {} Elevation'.format(tt))
    ax.set_ylabel('Elevation (NAVD88,m)')
    ax.set_xticks([])
    
    ax2.plot(distances,trans_diff,'k')
    #ax2.set_xlim(100,200)
    #ax.set_ylim(0,100000)
    ax2.plot(distances[peaks],trans_diff[peaks], "x", color = 'red')
    ax2.grid()
    ax2.set_xlabel('Distance Along Transect (m)')
    ax2.set_ylabel('Slope')
    
    matplotlib.pyplot.show()
    

#===============================================================================
# %% Export to shapefile
#===============================================================================

#point_1.to_file(os.path.join(dir_gis,name, 'edge1_pnts.shp'))
#point_2.to_file(os.path.join(dir_gis,name,'edge2_pnts.shp'))


line_1 = gpd.GeoSeries(LineString(point_1.to_list()),crs = trans.crs)
line_2 = gpd.GeoSeries(LineString(point_2.to_list()),crs = trans.crs)

line_1.to_file(os.path.join(dir_gis,name,'edge1_{}.shp'.format(name)))
line_2.to_file(os.path.join(dir_gis,name,'edge2_{}.shp'.format(name)))


# Save the corrected transect file 
# Shifted so transects all go across the river the same way.
trans.to_file(os.path.join(dir_gis,name,'{}_transects.shp'.format(name)))
