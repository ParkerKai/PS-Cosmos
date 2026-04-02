# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:19:36 2023

This function interpolates bathy to transects and exports to a shapefile

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
from shapely import geometry
import matplotlib
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d

from shapely.geometry import LineString
#from osgeo import gdal

#===============================================================================
# %% User Defined inputs
#===============================================================================

#### Directories #####
dir_gis = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\GIS\Shapefiles\StreamBathy'
dir_bathy = r'D:\BathyChunks\Puget_Sound_CoNED_Topobathy_DEM_1m'
dir_hydrosurvey = r'D:\BathyChunks\hydrology'
dir_raster_out  = r'D:\BathyChunks\hydrology\River_FixDems'
fig_out         = r'C:\Users\kaparker\Documents\Weekly_Updates\02_07_2024\figures'
name = 'Duwamish'

#####  Define coordinate systems #####
epsg_wgs = "EPSG:4326" # World Geodetic System 1984
epsg_utm = "EPSG:26910" # UTM Zone 10N (West coast of the US.)

###  Numerical ###
# Distance limit at which points are interpolated to a transect.
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

# Edges
Edge1 = gpd.read_file(os.path.join(dir_gis,name,'edge1_{}.shp'.format(name)))
Edge2 = gpd.read_file(os.path.join(dir_gis,name,'edge2_{}.shp'.format(name)))

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

# Initializ trans_edit to keep track of transect edits
# trns_edit = 0  DEM bathy (no edits)
# trsn_edit = 1  Hydrosurvey bathy (transect fixed with observed data)
# trns_edit = 2  Interpolated bathy 
# trns_edit = 3  Assumed bathy.

# Initialize the "thalweg z" variablen to keep track of the channel depth.
#thalweg_z  = np.zeros(shape= len(trans))

thalweg_z  ={'Dem': np.zeros(shape= len(trans)),
        'Survey': np.zeros(shape= len(trans)),
        'trns_edit': np.zeros(shape= len(trans))}
thalweg_z = pd.DataFrame(thalweg_z)

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
    
    # Interpolate to transect
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
        
        thalweg_z['trns_edit'][tt] = 1
        thalweg_z['Survey'][tt]  = temp[elev_name].min()
    
    # Save the DEM depth
    thalweg_z['Dem'][tt]  = trans_z.min()
        
    # Save things 
    # Note this currently saves the bathy extracted from the DEM (not survey based)
    # Next version should blend data and DEM here
    dat_save = pd.DataFrame({'Distance': distances, 'Z': trans_z,
                             'X': coord_list[:,0], 'Y': coord_list[:,1],
                               'transect': np.ones(len(distances),dtype = 'int32')*tt})
    
    if (tt == 0):
        trns_z =dat_save
    else:
        trns_z = pd.concat((trns_z,dat_save),axis= 0)


#===============================================================================
# %% Find thalweg intersection with transects
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
thalweg_dist = Dist_LineString(thalweg_trans)

#===============================================================================
# %% Find Model bias values for tweaking thalweg
#===============================================================================
if name == 'Puyallup':
    file_in1 = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\PierceKing\20231229_biascorrection_forKai\2_Payallup\bias_correcton.shp'
    file_in2 = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\PierceKing\20240119_BiasCorrection_forKaii\Payallup\bias_correcton.shp'
    file_in3 = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\PierceKing\20240126_BiasCorrection_forKai\2_Payallup\bias_correcton.shp'
    file_in4 = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\PierceKing\20240201_BiasCorrection_forKai\2_Payallup\bias_correcton.shp'

elif  name == 'Duwamish':
    file_in1 = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\PierceKing\20231229_biascorrection_forKai\1_Green\bias_correcton.shp'
    file_in2 = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\PierceKing\20240119_BiasCorrection_forKaii\Green\bias_correcton.shp'
    file_in3 = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\PierceKing\20240126_BiasCorrection_forKai\1_Green\bias_correcton.shp'
    file_in4 = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\PierceKing\20240201_BiasCorrection_forKai\1_Green\bias_correcton.shp'

Bias1 = gpd.read_file(file_in1)
Bias1.crs = epsg_wgs
Bias2 = gpd.read_file(file_in2)
Bias2.crs = epsg_wgs
Bias3 = gpd.read_file(file_in3)
Bias3.crs = epsg_wgs
Bias4 = gpd.read_file(file_in3)
Bias4.crs = epsg_wgs

# little bit of cleaning up the dataframe
Bias1.drop(['ID'], axis=1,inplace = True)
Bias1.rename(columns={"Val_1": "Bias"},inplace = True)

Bias2.drop(['ID'], axis=1,inplace = True)
Bias2.rename(columns={"Val_1": "Bias"},inplace = True)

Bias3.drop(['ID'], axis=1,inplace = True)
Bias3.rename(columns={"Val_1": "Bias"},inplace = True)

Bias4.drop(['ID'], axis=1,inplace = True)
Bias4.rename(columns={"Val_1": "Bias"},inplace = True)


# Project
Bias1.to_crs(crs = 'EPSG:32610',inplace = True) 
Bias2.to_crs(crs = 'EPSG:32610',inplace = True) 
Bias3.to_crs(crs = 'EPSG:32610',inplace = True) 
Bias4.to_crs(crs = 'EPSG:32610',inplace = True) 


# Snap data to the nearest coord line.  
# First create a dataframe for the thalweg points
coord_list = np.stack((thalweg_trans[0].coords.xy[0],
                       thalweg_trans[0].coords.xy[1]),axis=1)
thalweg_pnt = gpd.GeoSeries(gpd.points_from_xy(x = coord_list[:,0], y = coord_list[:,1]),crs = trans.crs) 
d = {'Trans_index': trans.index, 'geometry': thalweg_pnt}
thalweg_pnt = gpd.GeoDataFrame(d, crs=thalweg_pnt.crs)

# Find nearrest point to each location we have a bias measurement
temp1 = ckdnearest(Bias1,thalweg_pnt)
temp2 = ckdnearest(Bias2,thalweg_pnt)
temp3 = ckdnearest(Bias3,thalweg_pnt)
temp4 = ckdnearest(Bias4,thalweg_pnt)


# sort by index 
temp1.sort_values(by='Trans_index', inplace=True)
temp2.sort_values(by='Trans_index', inplace=True)
temp3.sort_values(by='Trans_index', inplace=True)
temp4.sort_values(by='Trans_index', inplace=True)


# Ignore negative bias (never increase river depth)
ind_mod = temp1.Bias <0
temp1.Bias[ind_mod] = 0  # Sets negative values to zero 
ind_mod = temp2.Bias <0
temp2.Bias[ind_mod] = 0  # Sets negative values to zero 
ind_mod = temp3.Bias <0
temp3.Bias[ind_mod] = 0  # Sets negative values to zero 
ind_mod = temp4.Bias <0
temp4.Bias[ind_mod] = 0  # Sets negative values to zero 

# add to the thalweg z dataframe
thalweg_z['bias1']  = np.zeros(shape= len(trans))
thalweg_z['bias2']  = np.zeros(shape= len(trans))
thalweg_z['bias3']  = np.zeros(shape= len(trans))
thalweg_z['bias4']  = np.zeros(shape= len(trans))

thalweg_z['bias1'].iloc[temp1['Trans_index'].values] =temp1['Bias'].values
thalweg_z['bias2'].iloc[temp2['Trans_index'].values] =temp2['Bias'].values
thalweg_z['bias3'].iloc[temp2['Trans_index'].values] =temp3['Bias'].values
thalweg_z['bias4'].iloc[temp2['Trans_index'].values] =temp4['Bias'].values


# Interpolate
thalweg_z['bias1_interp']  = np.zeros(shape= len(trans))
thalweg_z['bias2_interp']  = np.zeros(shape= len(trans))
thalweg_z['bias3_interp']  = np.zeros(shape= len(trans))
thalweg_z['bias4_interp']  = np.zeros(shape= len(trans))

ind_data1 = temp1['Trans_index'].values # Locations where we fit bathy to data
ind_data2 = temp2['Trans_index'].values # Locations where we fit bathy to data
ind_data3 = temp3['Trans_index'].values # Locations where we fit bathy to data
ind_data4 = temp4['Trans_index'].values # Locations where we fit bathy to data

thalweg_z['bias1_interp']  = np.interp(thalweg_dist, thalweg_dist[ind_data1], temp1['Bias'].values)
thalweg_z['bias2_interp']  = np.interp(thalweg_dist, thalweg_dist[ind_data2], temp2['Bias'].values)
thalweg_z['bias3_interp']  = np.interp(thalweg_dist, thalweg_dist[ind_data3], temp3['Bias'].values)
thalweg_z['bias4_interp']  = np.interp(thalweg_dist, thalweg_dist[ind_data4], temp4['Bias'].values)

thalweg_z['bias_interp']  = thalweg_z['bias1_interp'] + thalweg_z['bias2_interp']+ thalweg_z['bias3_interp'] + thalweg_z['bias4_interp']

# Deal with possibility of nans
ind_mod = np.isnan(thalweg_z['bias_interp'])
f = interp1d(thalweg_dist[~ind_mod], thalweg_z['bias_interp'][~ind_mod], kind='nearest',bounds_error=False, fill_value="extrapolate")
thalweg_z['bias_interp'][ind_mod] = f(thalweg_dist[ind_mod])


# NOTE THAT BIAS IS DOWN (need to deepen the channel)
#===============================================================================
# %%  Interpolate Survey data 
#===============================================================================

# Fit a model between locations we have bathy.
ind_bathy = (thalweg_z['trns_edit'] == 1) # Locations where we fit bathy to data

thalweg_z['Survey_interp'] = np.interp(thalweg_dist, thalweg_dist[ind_bathy], thalweg_z['Survey'][ind_bathy].values)

if name == 'Puyallup':
    thalweg_z['Survey_interp'].iloc[353:None] = thalweg_z['Dem'].iloc[353:None].values

    thalweg_z['bias_interp'].iloc[902:None] = np.linspace(thalweg_z['bias_interp'].iloc[902], 0, num=thalweg_z.shape[0]-902)
    thalweg_z['bias_interp'].iloc[0:117] = thalweg_z['bias_interp'].iloc[117]

elif  name == 'Duwamish':
    # Hack for fitting (very adhoc just to make things work.Doesn't make sense logically in any way)
    #thalweg_z['bias_interp'].iloc[411:522] = np.linspace(thalweg_z['bias_interp'].iloc[411],
    #                                                     thalweg_z['bias_interp'].iloc[411], num=522-411)

    ind_dem = (thalweg_z['trns_edit'] == 0) # Locations where we fit bathy to data
    thalweg_z['Survey_interp'][ind_dem] = thalweg_z['Dem'][ind_dem].values
    
    thalweg_z['Survey_interp'].iloc[522:670] = np.linspace(thalweg_z['Survey_interp'].iloc[522],
                                                           thalweg_z['Survey_interp'].iloc[670],
                                                           num=670-522)
    

thalweg_z['New_Z'] = thalweg_z['Survey_interp'] - thalweg_z['bias_interp']

spl = InterpolatedUnivariateSpline(thalweg_dist, thalweg_z['New_Z'])
spl.set_smoothing_factor(s=200)
thalweg_z['New_Z_smth'] = spl(thalweg_dist)

fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 6)

ax.plot(thalweg_dist,thalweg_z['Dem'],'k',label = 'DEM')
ax.plot(thalweg_dist[thalweg_z['trns_edit']== 1],thalweg_z['Survey'][thalweg_z['trns_edit']== 1],
        '+', color = 'red',label = 'Survey')

ax.plot(thalweg_dist[ind_data1],thalweg_z['Survey_interp'][ind_data1] - 
        thalweg_z['bias1_interp'][ind_data1],'.', color = 'magenta',label = 'Bias1')
ax.plot(thalweg_dist[ind_data2],thalweg_z['Survey_interp'][ind_data2] - 
        thalweg_z['bias1_interp'][ind_data1] - thalweg_z['bias2_interp'][ind_data2],
        '.', color = 'darkviolet',label = 'Bias2')
ax.plot(thalweg_dist[ind_data2],thalweg_z['Survey_interp'][ind_data2] -
        thalweg_z['bias1_interp'][ind_data1] - thalweg_z['bias2_interp'][ind_data2] -
        thalweg_z['bias3_interp'][ind_data3],'.', color = 'maroon',label = 'Bias3')
ax.plot(thalweg_dist[ind_data2],thalweg_z['Survey_interp'][ind_data2] -
        thalweg_z['bias1_interp'][ind_data1] - thalweg_z['bias2_interp'][ind_data2] -
        thalweg_z['bias3_interp'][ind_data3]  -
        thalweg_z['bias4_interp'][ind_data4],'.', color = 'blue',label = 'Bias4')


ax.plot(thalweg_dist,thalweg_z['New_Z'], color = 'blue',label = 'Combined_Bias')
ax.plot(thalweg_dist,thalweg_z['New_Z_smth'], color = 'green',label = 'Combined_Smoothed')


#ax.set_xlim(pd.Timestamp('2000-08-01'),pd.Timestamp('2001-08-01'))
#ax.set_xlim(100,200)
ax.grid()
ax.set_title('Thalweg Elevation')
ax.set_ylabel('Elevation (NAVD88,m)')
ax.set_xlabel('Distance Along Thalweg (m)')
ax.legend()


matplotlib.pyplot.show()
fig.savefig(os.path.join(fig_out,'Thalweg_{}.png'.format(name)),  dpi=800,
        bbox_inches='tight')           


# Export to arc
thalweg_trans.to_file(os.path.join(dir_gis,name,'thalweg_trans.shp'))

# Interpolate to finer resolution as well (for later)  
thalweg_fine = redistribute_vertices(thalweg[0], 1)

x = thalweg_fine.coords.xy[0]
y = thalweg_fine.coords.xy[1]

thalweg_fine = gpd.GeoSeries(thalweg_fine,crs = trans.crs)

thalweg_dist_fine = Dist_LineString(thalweg_fine)

thalweg_dist_fine = np.arange(0,len(x), 1)
z =  np.interp(thalweg_dist_fine, thalweg_dist, thalweg_z['New_Z_smth'])
thalweg_fine_xyz = np.stack((x,y,z),axis = 1)

#===============================================================================
# %% Create new transect profile
#===============================================================================
trns_z_new = trns_z

for tt in range(len(trans)):
    line = trans.loc[tt]
    
    # Redestribute vertices to every 1 m on a projected MultiLineString
    line = redistribute_vertices(line, 1)
    coord_list = np.stack((line.coords.xy[0],line.coords.xy[1]),axis=1)
    trans_pnts = gpd.GeoSeries(gpd.points_from_xy(x = coord_list[:,0], y = coord_list[:,1]),crs = trans.crs)     # Geoseries of line coordinates

    # Pull transect z from the matrix of all data
    pull = trns_z.query('transect == @tt')
    trans_z =pull['Z'].values
    distances = pull['Distance'].values
    
    # Edges of the Region to interpolate
    temp1 = Edge1.intersection(line)  # Intersection Edge1 and transect
    new = ckdnearest(trans_pnts,temp1)
    ind_edge1 = np.argmin(new['dist'])  # Find the index of the nearest point

    temp1 = Edge2.intersection(line)  # Intersection Edge2 and transect
    new = ckdnearest(trans_pnts,temp1)
    ind_edge2 = np.argmin(new['dist'])
    
    temp1 = thalweg.intersection(line)  # Intersection Thalweg and transect
    new = ckdnearest(trans_pnts,temp1)
    ind_mid = np.argmin(new['dist'])
    
    # Sort to make sure things go in the right order
    ind_1 = np.minimum(ind_edge1,ind_edge2)
    ind_2 = np.maximum(ind_edge1,ind_edge2)
    
    if Type == 1:
        # Create a profile for the missing region using a Trapezoidal Channel
        # Create interp region by remove the region we are trying to interpolate through
        # and adding a point in the middle at the depth we want for the thalweg
        mid_z =  np.ones(2)*thalweg_z['New_Z_smth'][tt]
        
        # add 2 thalweg points near the mid
        dx = np.round((ind_2-ind_1)/6)
        mid1 = (ind_mid - dx).astype('int32')
        mid2 = (ind_mid + dx).astype('int32')
        
        # Case where mid estimate is outside of edges
        if (mid1 <= ind_1):
            mid1 = ind_1+1
            
        if (mid2 >= ind_2):
            mid2 = ind_2-1
        
        mid_x = np.array([distances[mid1], distances[mid2]])
    
        temp_z = np.concatenate((trans_z[0:ind_1+1], mid_z,
                                 trans_z[ind_2:-1]), axis=0)
        
        temp_x = np.concatenate((distances[0:ind_1+1], mid_x,
                                 distances[ind_2:-1]), axis=0)
        
        spl = InterpolatedUnivariateSpline(temp_x, temp_z, k=4)
        trans_z_new = spl(distances)
        
        trans_z_new = np.interp(distances,temp_x, temp_z)
        
    else:
        print('not programmed yet!')
    
    # Plot
    fig, ax = matplotlib.pyplot.subplots(1, 1)
    ax.plot(temp_x,temp_z,'k')
    ax.plot(distances,trans_z,'k')
    ax.plot(distances[ind_1],trans_z[ind_1], "x", color = 'red')
    ax.plot(distances[ind_2],trans_z[ind_2], "x", color = 'red')
    ax.plot(distances,trans_z_new,'r')
    #ax.set_xlim(pd.Timestamp('2000-08-01'),pd.Timestamp('2001-08-01'))
    #ax.set_xlim(100,200)
    ax.grid()
    ax.set_title('Transect Elevation')
    ax.set_ylabel('Elevation (NAVD88,m)')
    ax.set_xticks([])
    matplotlib.pyplot.show()
        
    # Save to the dataframe
    trns_z_new['Z'].loc[(trns_z_new['transect'] == tt)] = trans_z_new
    

# Export as a shapefile

# First combine thalweg and transects
# x = np.concatenate((trns_z_coord[:,0],thalweg_fine_xyz[:,0]), axis = 0)
# y = np.concatenate((trns_z_coord[:,1],thalweg_fine_xyz[:,1]), axis = 0)
# z = np.concatenate((trns_z.flatten(),thalweg_fine_xyz[:,2]), axis = 0)
x = trns_z_new['X'].values
y = trns_z_new['Y'].values
z = trns_z_new['Z'].values

trns_z_out = gpd.GeoDataFrame({'Zelev': z, 
                               'geometry': gpd.points_from_xy(x = x, y = y, z= z)},
                              crs = trans.crs) 

#trns_z_out = trns_z_out.to_frame()
#trns_z_out['Zelev'] = 

trns_z_out.to_file(os.path.join(dir_raster_out,name,'NewBathy8_{}.shp'.format(name)))

#===============================================================================
# %% Interpolate transects and thalweg to new mini-dem
#===============================================================================


# Create polygon for area of interest
# Extract the points for the edge lines. Flipud is because we are going around the polygon, so need the end of the first edge to align with the start of the next.
X_point_list = np.concatenate((Edge1.geometry[0].coords.xy[0], np.flipud(Edge2.geometry[0].coords.xy[0])),
               axis = 0)

Y_point_list = np.concatenate((Edge1.geometry[0].coords.xy[1],  np.flipud(Edge2.geometry[0].coords.xy[1])),
               axis = 0)


polygon_geom = geometry.Polygon(zip(X_point_list, Y_point_list))
polygon_edit = gpd.GeoDataFrame(index=[0], crs = trans.crs, geometry=[polygon_geom])       


# Save the file
polygon_edit.to_file(os.path.join(dir_gis,name,'{}_EditPolygon8.shp'.format(name)))


# # Find raster points within the area of interest
# # Find x values for the river of interest
# xmin, ymin, xmax, ymax = polygon_edit.total_bounds


# height = bathy.shape[0]
# width = bathy.shape[1]
# cols, rows = np.meshgrid(np.arange(width), np.arange(height))

# # Find x and y values for raster that overlaps region of interest
# dx = 50 # little extra to make sure we cover the region of interest
# xs, ys = rasterio.transform.xy(bathy.transform, np.arange(height), 0)
# ind_y  = np.where((ys>= ymin-dx) & (ys<= ymax+dx))

# xs, ys = rasterio.transform.xy(bathy.transform,  0,np.arange(width))
# ind_x  = np.where((xs>= xmin-dx) & (xs<= xmax+dx))

# cols, rows = np.meshgrid(ind_x[0], ind_y[0])


# xs, ys = rasterio.transform.xy(bathy.transform, rows, cols)
# X_grd = np.array(xs)
# Y_grd = np.array(ys)

# bathy_grd = gpd.points_from_xy(X_grd.flatten(), Y_grd.flatten(), crs = trans.crs)

# # Clip to the region of interest
# bathy_grd = gpd.clip(bathy_grd, polygon_edit)


# Create a raster from the scattered data 
# rasterDs = gdal.Grid(os.path.join(dir_raster_out,'Puylallup_NewBathy.tif'), 
#                      os.path.join(dir_raster_out,'NewBathy.shp'), 
#                      format='GTiff',
#                      algorithm='invdist')




#===============================================================================
# %% Overwrite values in raster and export
#===============================================================================


