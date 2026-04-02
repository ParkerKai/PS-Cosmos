# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:54:08 2023

This function Pulls height at the edge of the channel and exports as a shapefile

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
import sys
import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np
import matplotlib
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import find_peaks
from shapely.geometry import LineString

#from osgeo import gdal

#===============================================================================
# %% User Defined inputs
#===============================================================================

#### Directories #####
dir_gis = r'Y:\PS_Cosmos\GIS\StreamBathy'
dir_bathy = r'Y:\PS_Cosmos\01_data\topo_bathymetry\DEM\PugetSound_DEM\Puget_Sound_CoNED_Topobathy_DEM_1m'
dir_out  = r'Y:\PS_Cosmos\GIS\StreamBathy'

name = 'Puyallup'

#####  Define coordinate systems #####
epsg_wgs = "EPSG:4326" # World Geodetic System 1984
epsg_utm = "EPSG:26910" # UTM Zone 10N (West coast of the US.)

###  Numerical ###
# Resolution for transects 
distance_delta = 1

# Spline Smoothing
# cranking this up will smooth the initial bathymetry before trying to find Levee elevations
sp_sm = 0.1

# Peak prominence
# Similar to spline smoothing
# higher peak prominence tollerance will mean more likely to find the correct peak
# But too high means you won't be able to find one
pk_prom = 0.02

# Algorithim for choosing levee height
# 1: Choose the nearest peak to the channel edge. Good for not choosing too high of a value when the bank continues upward.
# 2: Choose the highest peak.   
Type = 2

#===============================================================================
# %% Define some functions
#===============================================================================
def redistribute_vertices(geom, distance):
    if geom.geom_type == 'LineString':
        num_vert = int(round(geom.length / distance))
        if num_vert == 0:
            num_vert = 1
        return LineString(
            [geom.interpolate(float(n) / num_vert, normalized=True)
             for n in range(num_vert + 1)])
    elif geom.geom_type == 'MultiLineString':
        parts = [redistribute_vertices(part, distance)
                 for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])
    else:
        raise ValueError('unhandled geometry %s', (geom.geom_type,))


# Using super fast ckdtree search from scipi
# Will return the distance and 'Name' of the nearest neighbor in gpdB from each point in gpdA. 
# It assumes both gdfs have a geometry column (of points).
def ckdnearest(gdA, gdB):
    from scipy.spatial import cKDTree

    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ], 
        axis=1)

    return gdf

# Determine distance along a LineSTring
# Input is a geopandas Geoseries with a single linestring feature
def Dist_LineString(LineIn):
    pnts = gpd.GeoSeries(gpd.points_from_xy(x = LineIn[0].coords.xy[0], y = LineIn[0].coords.xy[1]),
                                            crs = LineIn.crs)
    dist_pnt = np.zeros(len(pnts))
    for ii in range(len(pnts)):
        
        # Dont do first point as it is empty
        if (ii >0):
            
            pnt1 = pnts[ii-1]
            pnt2  = pnts[ii]
            dist_pnt[ii] = pnt2.distance(pnt1)
    
    dist = np.cumsum(dist_pnt)    

    return dist

#===============================================================================
# %% Load the Data
#===============================================================================

# Gis features of the river 
thalweg = gpd.read_file(os.path.join(dir_gis,name,'{}_thalweg.shp'.format(name)))
trans = gpd.read_file(os.path.join(dir_gis,name, '{}_transects.shp'.format(name)))

# Reduce to dataseries
thalweg = thalweg['geometry']
trans   = trans['geometry']

# DEM Tiff
bathy_Coned = rasterio.open(os.path.join(dir_bathy,'Puget_Sound_CoNED_Topobathy_DEM_1m.tif'))

# Edges
Edge1 = gpd.read_file(os.path.join(dir_gis,name,'edge1_{}.shp'.format(name)))
Edge2 = gpd.read_file(os.path.join(dir_gis,name,'edge2_{}.shp'.format(name)))

#===============================================================================
# %% First a couple checks that geometry is going to work. 
#===============================================================================
# Makes sure that all transects and edges have a single intersection.  

flag_break = False 
for tt in range(len(trans)):
    
    # Check Thalweg and Transect intersection
    temp = thalweg.intersection(trans[tt])
    
    if (temp[0].geometryType() == 'MultiPoint'):
        print('Multiple Intersections for Thalweg and Transect {}'.format(tt))
        flag_break = True
    # Check Edges
    line = trans.loc[tt]
    temp1 = Edge1.intersection(line)  # Intersection Edge and transect
    if (temp1[0].geometryType() == 'MultiPoint'):
        print('Multiple Intersections for Edge1 and Transect {}'.format(tt))
        flag_break = True

    temp2 = Edge2.intersection(line)  # Intersection Edge1 and transect
    if (temp2[0].geometryType() == 'MultiPoint'):
        print('Multiple Intersections for Edge2 and Transect {}'.format(tt))
        flag_break = True

if flag_break:
    sys.exit("Fix geometry before Preceeding")
    
#===============================================================================
# %% Get bathy information from the CONED raster 
#===============================================================================

# Initializ trans_edit to keep track of transect edits
# trns_edit = 0  DEM bathy (no edits)
# trsn_edit = 1  Hydrosurvey bathy (transect fixed with observed data)
# trns_edit = 2  Interpolated bathy 
# trns_edit = 3  Assumed bathy.
trns_edit = np.zeros(shape= len(trans),dtype = 'int16')

# Initialize the "thalweg z" variablen to keep track of the channel depth.
thalweg_z  = np.zeros(shape= len(trans))

for tt in range(len(trans)):
    line = trans.loc[tt]
    
    # Redestribute vertices to every 1 m on a projected MultiLineString
    line = redistribute_vertices(line, 1)
    coord_list = np.stack((line.coords.xy[0],line.coords.xy[1]),axis=1)
    trans_pnts = gpd.GeoSeries(gpd.points_from_xy(x = coord_list[:,0], y = coord_list[:,1]),crs = trans.crs)     # Geoseries of line coordinates
    
    # Distances for transects Add extra points to the line
    # Used to increase resolution of transects
    distances = np.arange(0, np.round_(line.length), distance_delta)

    # add another point for plotting and lining things up
    distances = np.append(distances,distances[-1]+distance_delta)

    # Coordinates (as a list and a geoseries of points)
    coord_list = np.stack((line.coords.xy[0],line.coords.xy[1]),axis=1)
    trans_pnt = gpd.GeoSeries(gpd.points_from_xy(x = coord_list[:,0], y = coord_list[:,1]),crs = trans.crs) 
    
    # Pull the bathymetry from the DEM
    trans_z = [x for x in bathy_Coned.sample(coord_list)]
    trans_z = np.asarray(trans_z)

    # smooth the bathy series a bit
    spl = InterpolatedUnivariateSpline(distances, trans_z)
    spl.set_smoothing_factor(s=0.1)
    trans_z = spl(distances)
        
    # Edges of the Region to interpolate
    temp1 = Edge1.intersection(line)  # Intersection Edge1 and transect
    new = ckdnearest(trans_pnts,temp1)
    ind_edge1 = np.argmin(new['dist'])  # Find the index of the nearest point

    temp1 = Edge2.intersection(line)  # Intersection Edge2 and transect
    new = ckdnearest(trans_pnts,temp1)
    ind_edge2 = np.argmin(new['dist'])
    
    # Sort to make sure things go in the right order
    ind_1 = np.minimum(ind_edge1,ind_edge2)
    ind_2 = np.maximum(ind_edge1,ind_edge2)
    
    # Find peaks in elevation change
    peaks,properties  = find_peaks(trans_z, prominence=pk_prom)
    

    if Type == 1:
        # Get rid of peaks within the edges
        peaks_sel = peaks[((peaks < ind_1) | (peaks > ind_2))]
        
        # Find the nearest ind to the edges 
        ind_pk1 = peaks_sel[np.argmin(np.abs(peaks_sel-ind_1))]
        ind_pk2 = peaks_sel[np.argmin(np.abs(peaks_sel-ind_2))]
        
    elif Type == 2:
    
        # Find the largest peaks
        peaks_sel = peaks[(peaks < ind_1)] # outside of channel left
        if peaks_sel.size > 0:
            ind_pk1 = peaks_sel[np.argmax(trans_z[peaks_sel])]
        else:
            ind_pk1 = np.argmax(trans_z[0:ind_1])
            
        peaks_sel = peaks[(peaks > ind_2)] # oustide of channel right
        if peaks_sel.size > 0:
            ind_pk2 = peaks_sel[np.argmax(trans_z[peaks_sel])]
        else:
            ind_pk2 = np.argmax(trans_z[ind_2:-1])+ind_2
        
    # Plot
    fig, ax = matplotlib.pyplot.subplots(1, 1)
    ax.plot(distances,trans_z,'k')
    ax.plot(distances[peaks],trans_z[peaks], ".", color = 'red')
    ax.plot(distances[ind_1],trans_z[ind_edge1], "x", color = 'red')
    ax.plot(distances[ind_2],trans_z[ind_2], "x", color = 'red')
    ax.plot(distances[ind_pk1],trans_z[ind_pk1], "o", color = 'red')
    ax.plot(distances[ind_pk2],trans_z[ind_pk2], "o", color = 'red')
    #ax.plot(distances,trans_z_new,'r')
    #ax.set_xlim(pd.Timestamp('2000-08-01'),pd.Timestamp('2001-08-01'))
    #ax.set_xlim(100,200)
    ax.grid()
    ax.set_title('Transect {} Elevation'.format(tt))
    ax.set_ylabel('Elevation (NAVD88,m)')
    ax.set_xticks([])
    matplotlib.pyplot.show()
    
    if (tt == 0):
        bank1_X = trans_pnts.iloc[ind_pk1].x;
        bank1_Y = trans_pnts.iloc[ind_pk1].y;
        bank1_Z = trans_z[ind_pk1];

        bank2_X = trans_pnts.iloc[ind_pk2].x;
        bank2_Y = trans_pnts.iloc[ind_pk2].y;
        bank2_Z = trans_z[ind_pk2];
    else:
        bank1_X = np.vstack((bank1_X, trans_pnts.iloc[ind_pk1].x))
        bank1_Y = np.vstack((bank1_Y, trans_pnts.iloc[ind_pk1].y))
        bank1_Z = np.vstack((bank1_Z, trans_z[ind_pk1]))

        bank2_X = np.vstack((bank2_X, trans_pnts.iloc[ind_pk2].x))
        bank2_Y = np.vstack((bank2_Y, trans_pnts.iloc[ind_pk2].y))
        bank2_Z = np.vstack((bank2_Z, trans_z[ind_pk2]))

# %% Export as a shapefile

# Squeeze down to a single dimention for creating dataframe
bank1_X = np.squeeze(bank1_X)
bank1_Y = np.squeeze(bank1_Y)
bank1_Z = np.squeeze(bank1_Z)

bank2_X = np.squeeze(bank2_X)
bank2_Y = np.squeeze(bank2_Y)
bank2_Z = np.squeeze(bank2_Z)


bank1_out = gpd.GeoDataFrame({'Zelev': bank1_Z, 
                               'geometry': gpd.points_from_xy(x = bank1_X, y = bank1_Y, z= bank1_Z)},
                              crs = trans.crs) 

bank2_out = gpd.GeoDataFrame({'Zelev': bank2_Z, 
                               'geometry': gpd.points_from_xy(x = bank2_X, y = bank2_Y, z= bank2_Z)},
                              crs = trans.crs) 


bank1_out.to_file(os.path.join(dir_out,name, 'bank1_{}_elev.shp'.format(name)))
bank2_out.to_file(os.path.join(dir_out,name, 'bank2_{}_elev.shp'.format(name)))



