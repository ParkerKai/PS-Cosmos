# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:26:05 2023

This function explores the NHPlus dataset and its viability for looking at thalweg depths.

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
from shapely import ops
import matplotlib
from scipy.interpolate import InterpolatedUnivariateSpline
from shapely.geometry import LineString, Point
import fiona

#from osgeo import gdal

#===============================================================================
# %% User Defined inputs
#===============================================================================

#### Directories #####
dir_gis         = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\GIS\Shapefiles\StreamBathy'
dir_bathy       = r'D:\BathyChunks\Puget_Sound_CoNED_Topobathy_DEM_1m'
dir_hydrosurvey = r'D:\BathyChunks\hydrology'
dir_raster_out  = r'D:\BathyChunks\hydrology\River_FixDems'
dir_NHD         = r'D:\Hydrology\NHDPlus_HR\NHDPLUS_H_1711_HU4_GDB' 

name = 'Puyallup'

###  Numerical ###
# Distance limit at which points are snapped to a transect.
# Points past this are ignored.
dist_lim = 1000

# Resolution for transects 
distance_delta = 1

# Interpolation type
# Type =1 Interpolate min thalweg depth for stream. Then replace all channels with trapezoidal channel.
Type = 1


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

# Convert a geopandas linestring into a geoseries of points.
# Unwraps the lines into a single geosereis if there are multiple in the lines in the originating geoseries.
def linestring_to_points(gpd_line):
    for count,line in enumerate(gpd_line):
        # Coordinates (as a list and a geoseries of points)
        
        pnts = gpd.GeoSeries(gpd.points_from_xy(x = line.coords.xy[0], y = line.coords.xy[1]),
                                            crs = gpd_line.crs)
        pnts = pnts.rename('geometry')
        if (count ==0):
            gpd_pnt = pnts
            
        else:    
            gpd_pnt = pd.concat([gpd_pnt,pnts],axis = 0)
            
    gpd_pnt.reset_index(drop = True)
    return gpd_pnt

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
bathy = rasterio.open(os.path.join(dir_bathy,'Puget_Sound_CoNED_Topobathy_DEM_1m.tif'))

#===============================================================================
# %% Load the Bathymetry data data
#===============================================================================
if name == 'Puyallup':
    HS = gpd.read_file(os.path.join(dir_hydrosurvey,name,'Pierce_HydroSurvery_utm.shp'))
    elev_name = 'Elev_m'  # NAme of the elevation attribute
    
elif  name == 'Duwamish':
    HS = gpd.read_file(os.path.join(dir_hydrosurvey,name,'Duwamish_Bathy_Combined.shp'))
    elev_name = 'NAVD88_m' # NAme of the elevation attribute
  
#===============================================================================
# %% For each point find the nearest transect
#===============================================================================

# 'Trans' is an index corresponding to the transect the points should be mapped to.
HS['Trans'] = np.empty(len(HS),dtype = 'int64')
for tt in range(len(HS)):
    pnt_pull = HS.geometry[tt]
    # Find distance of all transects
    dist =  np.empty(len(trans))
    for dd in range(len(trans)):
        dist[dd] = pnt_pull.distance(trans.geometry[dd])
    
    # Find the minimum 
    if (dist.min() <= dist_lim):
        HS['Trans'][tt] = np.argmin(dist)
        
    else:
        HS['Trans'][tt] = -999

#===============================================================================
# %% Get bathy information from the CONED raster 
#===============================================================================

# Initialize the "thalweg z" variablen to keep track of the channel depth.
thalweg_z      = np.zeros(shape= len(trans))
thalweg_z_data = np.zeros(shape= len(trans)) 
trns_edit = np.zeros(shape= len(trans),dtype = 'int16')

for tt in range(len(trans)):
    line = trans.loc[tt]
    
    # Redestribute vertices to every 1 m on a projected MultiLineString
    line = redistribute_vertices(line, 1)
    
    # Distances for transects Add extra points to the line
    # Used to increase resolution of transects
    distances = np.arange(0, np.round_(line.length), distance_delta)

    # add another point for plotting and lining things up
    distances = np.append(distances,distances[-1]+distance_delta)

    # Coordinates (as a list and a geoseries of points)
    coord_list = np.stack((line.coords.xy[0],line.coords.xy[1]),axis=1)
    trans_pnt = gpd.GeoSeries(gpd.points_from_xy(x = coord_list[:,0], y = coord_list[:,1]),crs = trans.crs) 
    
    # Pull the bathymetry from the DEM
    trans_z = [x for x in bathy.sample(coord_list)]
    trans_z = np.asarray(trans_z)

    # smooth the bathy series a bit
    spl = InterpolatedUnivariateSpline(distances, trans_z)
    spl.set_smoothing_factor(s=0.1)
    trans_z = spl(distances)
    
    # Find any Bathy for this transect
    Bathy = HS.query('Trans == @tt')
    
    # Pull Z if there is Bathy
    if len(Bathy) > 0 :
        print('Bathy found for Transect {transect}'.format(transect = tt))
        
        # Snap data to the nearest coord line.  
        temp = ckdnearest(trans_pnt, Bathy)
        
        fig, ax = matplotlib.pyplot.subplots(1, 1)
        ax.plot(distances,trans_z,'k',label = 'DEM')
        ax.plot(distances,temp[elev_name],'b', label = 'Hydrosurvey')
        #ax.set_xlim(pd.Timestamp('2000-08-01'),pd.Timestamp('2001-08-01'))
        #ax.set_xlim(100,200)
        ax.grid()
        ax.set_title('Transect {} Elevation'.format(tt))
        ax.set_ylabel('Elevation (NAVD88,m)')
        ax.set_ylabel('Along Transect Distance (m)')
        ax.legend()
        matplotlib.pyplot.show()
        
        
        thalweg_z_data[tt]  = temp[elev_name].min()
        trns_edit[tt] = 1

    # Also save the non surveyed (DEM) Version
    thalweg_z[tt]  = trans_z.min()
          
#===============================================================================
# %% Find thalweg intersection with transects and interpolate
#===============================================================================

thalweg_trans =  [0]*len(trans)
for tt in range(len(trans)):
    temp = thalweg.intersection(trans[tt])
    if (temp[0].geometryType() == 'Point'):
        thalweg_trans[tt] = temp[0]
    elif temp[0].geometryType() == 'MultiPoint':
        print('Multiple Crosses for Transect {}'.format(tt))

temp = gpd.GeoSeries(thalweg_trans,crs=trans.crs)
thalweg_trans = gpd.GeoSeries(LineString(temp),crs = trans.crs)

# Distance along the thalweg
dist_x = Dist_LineString(thalweg_trans)

# Fit a model between locations we have bathy.
ind_bathy = (trns_edit == 1) # Locations where we fit bathy to data

thalweg_z_new = np.interp(dist_x, dist_x[ind_bathy], thalweg_z_data[ind_bathy])

#spl = InterpolatedUnivariateSpline(dist_x[ind_bathy], thalweg_z[ind_bathy])
#spl.set_smoothing_factor(s=0.1)
#thalweg_z_new = spl(dist_x)

fig, ax = matplotlib.pyplot.subplots(1, 1)
ax.plot(dist_x,thalweg_z,'k',label = 'Thalweg')
ax.plot(dist_x[ind_bathy],thalweg_z_data[ind_bathy], "o", color = 'red',label = 'Hydrosurvey')
ax.plot(dist_x,thalweg_z_new,'r',label = 'Thalweg Fit')
#ax.set_xlim(pd.Timestamp('2000-08-01'),pd.Timestamp('2001-08-01'))
#ax.set_xlim(100,200)
ax.grid()
ax.set_title('Thalweg Elevation')
ax.set_ylabel('Elevation (NAVD88,m)')
ax.set_xlabel('Distance Along Thalweg (m)')
ax.legend()

matplotlib.pyplot.show()

# Export to arc
thalweg_trans.to_file(os.path.join(dir_gis,name,'thalweg_trans.shp'))


# # Interpolate to finer resolution as well (for later)  
# thalweg_fine = redistribute_vertices(thalweg[0], 1)

# x = thalweg_fine.coords.xy[0]
# y = thalweg_fine.coords.xy[1]

# thalweg_fine = gpd.GeoSeries(thalweg_fine,crs = trans.crs)

# dist_x_fine = Dist_LineString(thalweg_fine)

# dist_x_fine = np.arange(0,len(x), 1)
# z =  np.interp(dist_x_fine, dist_x[ind_bathy], thalweg_z[ind_bathy])
# thalweg_fine_xyz = np.stack((x,y,z),axis = 1)
# trans_fine_pnts = gpd.GeoSeries(gpd.points_from_xy(x = thalweg_fine_xyz[:,0], 
#                                                    y = thalweg_fine_xyz[:,1],
#                                                    z = thalweg_fine_xyz[:,2]),
#                                 crs = trans.crs)     # Geoseries of line coordinates


#===============================================================================
# %% Read the NHDPLUS data 
#===============================================================================
#polygon_edit = gpd.read_file(os.path.join(dir_gis,name,'{}_EditPolygon.shp'.format(name)))
polygon_edit = gpd.read_file(os.path.join(dir_gis,name,'EditPolygon.shp'))

lyrs = fiona.listlayers(os.path.join(dir_NHD,'NHDPLUS_H_1711_HU4_GDB.gdb'))
Streams = gpd.read_file(os.path.join(dir_NHD,'NHDPLUS_H_1711_HU4_GDB.gdb'), driver='FileGDB', layer='NHDFlowline')
Streams = Streams.to_crs(crs = polygon_edit.crs)

# Find the NHD streams within the edit polygon 
NHD_pull = gpd.sjoin(Streams, polygon_edit, op='intersects')

# First way to cut down the dataset (get rid of unamed chunks)
# temp = NHD_pull['GNIS_ID'].dropna()
# NHD_pull = NHD_pull.loc[temp.index]

# Second way based on name
NHD_pull = NHD_pull.query('GNIS_Name == "Puyallup River"')
NHD_pull.to_file(os.path.join(dir_gis,name,'{}_NHD.shp'.format(name)))
NHDPlusID = NHD_pull['NHDPlusID']

# Load in the attribute table 
NHD_VAA = gpd.read_file(os.path.join(dir_NHD,'NHDPLUS_H_1711_HU4_GDB.gdb'), driver='FileGDB', layer='NHDPlusFlowlineVAA')

for ii in range(len(NHDPlusID)):
    ID = NHDPlusID.iloc[ii]
    pull = NHD_pull.iloc[ii].geometry
    merged_line = ops.linemerge(pull) # Merge the multiline into a single line so i can access the coordinates 
        
    # Pull the "Value Added Table" for the steram segment
    Channel_VAA = NHD_VAA.query('NHDPlusID == @ID')
  
    pnt_start = Point(merged_line.coords[0][0], merged_line.coords[0][1], Channel_VAA['MinElevSmo'].to_numpy()/100)
    pnt_end   = Point(merged_line.coords[-1][0], merged_line.coords[-1][1], Channel_VAA['MinElevSmo'].to_numpy()/100)
    
    # By NHDPlus guidlines, Start is up stream, end is downstream.  

    pnt_start = gpd.GeoSeries(pnt_start,crs = polygon_edit.crs)
    pnt_end    = gpd.GeoSeries(pnt_end,crs = polygon_edit.crs)
    
    # Only save the downstream since it is the minimum depth (what we are interested in)
    # I tested start and end but you end up with differing depths at the same point (where two segments overlap.)
    if ii == 0:
       # NHD_pnts = pd.concat([pnt_start,pnt_end],axis = 0)
        NHD_pnts = pnt_end

    else:
        #NHD_pnts = pd.concat([NHD_pnts,pnt_start,pnt_end],axis = 0)
        NHD_pnts = pd.concat([NHD_pnts,pnt_end],axis = 0)

NHD_pnts = NHD_pnts.reset_index(drop = True)
NHD_pnts = gpd.GeoDataFrame({'Zelev': NHD_pnts.z, 
                               'geometry': NHD_pnts},
                              crs = polygon_edit.crs) 

NHD_pnts.to_file(os.path.join(dir_gis,name,'{}_pnts_NHD.shp'.format(name)))


#==============================================================
# %% Get NHD information along the thalweg 
#===============================================================================
thalweg_pnts = linestring_to_points(thalweg_trans)

# 'Trans' is an index corresponding to the transect the points should be mapped to.
NHD_pnts['Thalweg_pnt'] = np.empty(len(NHD_pnts),dtype = 'int64')
pnts_temp = NHD_pnts.geometry
for tt in range(len(NHD_pnts)):
    pnt_pull = NHD_pnts.geometry.iloc[tt]
    
    # Find distance of all transects
    dist =  np.empty(len(thalweg_pnts))
    for dd in range(len(thalweg_pnts)):
        dist[dd] = pnt_pull.distance(thalweg_pnts[dd])
    
    # Find the minimum 
    if (dist.min() <= dist_lim):
        NHD_pnts['Thalweg_pnt'][tt] = np.argmin(dist)
        
    else:
        NHD_pnts['Thalweg_pnt'][tt] = -999



fig, ax = matplotlib.pyplot.subplots(1, 1)
ax.plot(dist_x,thalweg_z,'k',label = 'Thalweg')
ax.plot(dist_x[ind_bathy],thalweg_z_data[ind_bathy], "o", color = 'red',label = 'Hydrosurvey')
ax.plot(dist_x,thalweg_z_new,'r',label = 'Thalweg Fit')
ax.plot(dist_x[NHD_pnts['Thalweg_pnt']],NHD_pnts['Zelev'],"o", color = 'blue',label = 'NHDplus')
#ax.set_xlim(pd.Timestamp('2000-08-01'),pd.Timestamp('2001-08-01'))
#ax.set_xlim(100,200)
ax.grid()
ax.set_title('Thalweg Elevation')
ax.set_ylabel('Elevation (NAVD88,m)')
ax.set_xlabel('Distance Along Thalweg (m)')
ax.legend()

matplotlib.pyplot.show()



