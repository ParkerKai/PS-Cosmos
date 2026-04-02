# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:51:51 2024

This funciton adds weirs to DEM tiles.
The idea is to force Weir features onto the edge of the river during subgridding.

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
import matplotlib 
import numpy as np
from osgeo import gdal

#===============================================================================
# %% User Defined
#===============================================================================
dir_in_Dem = r'Y:\PS_Cosmos\01_data\topo_bathymetry\DEM\Pierce_King'
dir_in_gis = r'Y:\PS_Cosmos\GIS\StreamBathy'
dir_temp = r'Y:\PS_Cosmos\GIS\StreamBathy\tempfiles'
dir_figs = r'Y:\PS_Cosmos\Figures\DEM_Fixes'

river= 'Duwamish'
county = 'King'

#===============================================================================
# %% Functions
#===============================================================================

sys.path.append(r'C:\Users\kaparker\Documents\GitHub\Python\Kai_Python\General_Functions')
from Kai_GeoTools import linestring_to_points,ckdnearest,redistribute_vertices, points_to_linestring, reverse_geom
from Kai_SignalProcessingTools import smooth 


def InterpWeir2Line(line1,line2,dem,dem_elev, thalweg):
    # For each value of the edge snap the nearest weir value
    line1 = linestring_to_points(line1)
    thalweg = linestring_to_points(thalweg)
    thalweg['ID_Thalweg'] = thalweg.index
    
    line1_weirs = ckdnearest(line1,line2)
    line1_weirs = line1_weirs.rename(columns={"ID": "ID_weir", "Val_1": "Elev","dist":"dist2weir"})
    
    row, col = dem.index(line1['geometry'].x, line1['geometry'].y)
    line1_weirs['Dem_Elev']= dem_elev[row,col]
    
    # Check to see if the nearest algorithm found the wrong side of the river
    # assumes the weir IDs are continuous along the river
    check_IdDiff = np.diff(line1_weirs['ID_weir'])
    if check_IdDiff.max() > 50:
        id_bad = np.argwhere(check_IdDiff > 50)
        print(f'Discontinuity in nearest weir IDs found for index {id_bad}')
        
        temp = line1_weirs['ID_weir'][id_bad.flatten()]
        
        ind_bad =  np.argwhere(np.isin(line2['ID'].values,temp.values))
        line2_subset = line2.drop(ind_bad.flatten(),axis=0)
        
        # REdo the nearest with the subset
        line1_weirs = ckdnearest(line1,line2_subset)
        line1_weirs = line1_weirs.rename(columns={"ID": "ID_weir", "Val_1": "Elev","dist":"dist2weir"})
        
        row, col = dem.index(line1['geometry'].x, line1['geometry'].y)
        line1_weirs['Dem_Elev']= dem_elev[row,col]
    
    # Add thalweg rivermile to the file
    out = ckdnearest(line1_weirs,thalweg)
    
    
    return out




def burn_shape(file_raster,file_shapefile,dir_temp):

    src = gdal.Open(file_raster)
    ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)
    
    outputBounds = [ulx,lry, lrx,uly]  # output bounds as (minX, minY, maxX, maxY) in target SRS.
    
    # FIlename for the temp file
    temp_file = os.path.join(dir_temp,'tempfile.tif')
    ds = gdal.Rasterize(temp_file, file_shapefile, xRes=xres, yRes=yres, allTouched=True,
                        outputBounds= outputBounds, useZ=True, 
                        outputType=gdal.GDT_Float32)
    mask = ds.ReadAsArray()
    ds = None
    gdal.Unlink(temp_file)
    
    return mask

def align_geom(df):
    '''
    takes a dataframe of lines and shifts geometry so they are all aligned end to end
    '''
    
    new_df = []
    for cnt,line in df.iterrows():
        
        # Get the endpoints of the line
        line = line['geometry']
        x = line.xy[0]
        y = line.xy[1]
        p = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y),crs= df.crs)
        
        # Reduce to only first and last points. 
        p = p.iloc[[0,-1]]
        
        # If we are past the first point
        if cnt > 0:
            # Distance to previous 
            # p and pnts_prev are [beginning, end]
            # We want the beginning of line 2 to be next to the end of line 1
            pnts_prev = pnts_prev.iloc[::-1]
            dist = p.distance(pnts_prev,align=False)
            
            # Reverse the line
            if (dist.iloc[-1] < dist.iloc[0]):
                line_new = reverse_geom(line)
                print(f'line {cnt} is reversed')                
                line = line_new
                
                # REcreate points so they are aligned for saving 
                p = gpd.GeoDataFrame(geometry=gpd.points_from_xy(line.xy[0], line.xy[1]),
                                     crs= df.crs)
                
                # Reduce to only first and last points. 
                p = p.iloc[[0,-1]]
                
                
        # Save for concatenating later
        new_df.append(gpd.GeoDataFrame({'Name': [f'L{cnt}'], 'geometry': line},crs = crs_save))

        # Save end of last line
        pnts_prev = p
        
    out = pd.concat(new_df)

        
    return out    
    

#===============================================================================
# %% Read in DEM
#===============================================================================
if county == 'King':
    file_raster = os.path.join(dir_in_Dem,f'{county}_2m_fixed_leaking_levees.tif')
elif county == 'Pierce':
    file_raster = os.path.join(dir_in_Dem,f'{county}_2m.tif')

dem = rasterio.open(file_raster)
dem_elev = dem.read(1)

#===============================================================================
# %% Read in shapefiles
#===============================================================================

# Read in the weir shapefile
weirs = gpd.read_file(os.path.join(dir_in_Dem,'weirs',county,'weir_dots.shp'))

# Strip out Nan values
weirs = weirs.dropna(ignore_index = True)

# Read in the thalweg file
if county == 'King':
    thalweg =  gpd.read_file(os.path.join(dir_in_gis,river,f'{river}_thalweg3.shp'))
elif county == 'Pierce':
    thalweg =  gpd.read_file(os.path.join(dir_in_gis,river,f'{river}_thalweg.shp'))
crs_save = thalweg.crs
thalweg = gpd.GeoDataFrame({'Name': ['Thalweg'], 'geometry': [redistribute_vertices(thalweg.geometry[0],1)]},crs = crs_save)


#===============================================================================
# %% Calculate Weir elevation snapped to levee location
#===============================================================================

file_l1 = os.path.join(dir_in_gis,river,f'Levee1_{river}_Chunks.shp')
file_l2 = os.path.join(dir_in_gis,river,f'Levee2_{river}_Chunks.shp')

# Edge 1 and edge 2
l1 =  gpd.read_file(file_l1)
l2 =  gpd.read_file(file_l2)
crs_save = l1.crs


# Order is messed up with lines so re-arrange
if county == 'Pierce':
    l1 = l1.iloc[[0,2,1]].reset_index()
    l2 = l2.iloc[[1,2,0,4,3]].reset_index()
elif county == 'King':
    l2 = l2.iloc[[3,2,1,0]].reset_index()

# Align so all linestrings are sequential 
l1_new = align_geom(l1)
l2_new = align_geom(l2)


l1_new=[]
l2_new=[]
for index, row in l1.iterrows():        
    l1_new.append(gpd.GeoDataFrame({'Name': [f'L{index}'], 'geometry': [redistribute_vertices(row.geometry,2)]},crs = crs_save))
l1 = pd.concat(l1_new, ignore_index=True)

for index, row in l2.iterrows():        
    l2_new.append(gpd.GeoDataFrame({'Name': [f'L{index}'], 'geometry': [redistribute_vertices(row.geometry,2)]},crs = crs_save))
l2 = pd.concat(l2_new, ignore_index=True)


e1_weirs = InterpWeir2Line(l1, weirs,dem, dem_elev, thalweg)
e2_weirs = InterpWeir2Line(l2, weirs,dem, dem_elev, thalweg)

# Smooth
e1_weirs['Elev_smth'] = smooth(e1_weirs['Elev'],window_len=50)

# spl = scipy.interpolate.UnivariateSpline(e1_weirs['ID_Thalweg'], e1_weirs['Elev'])
# spl.set_smoothing_factor(s=40)
# e1_weirs['Elev_smth'] = spl(e1_weirs['ID_Thalweg'])

# spl = scipy.interpolate.UnivariateSpline(e2_weirs['ID_Thalweg'], e2_weirs['Elev'])
# spl.set_smoothing_factor(s=350)
# e2_weirs['Elev_smth'] = spl(e2_weirs['ID_Thalweg'])
e2_weirs['Elev_smth'] = smooth(e2_weirs['Elev'],window_len=50)


# Plot
fig, ax = matplotlib.pyplot.subplots(2, 1)
ax[0].plot(e1_weirs['ID_Thalweg']/1000,e1_weirs['Elev'],'k',label='Weir Elevation')
ax[0].plot(e1_weirs['ID_Thalweg']/1000,e1_weirs['Dem_Elev'],'r', label='Dem Elevation')
ax[0].plot(e1_weirs['ID_Thalweg']/1000,e1_weirs['Elev_smth'],'b',label='Weir Elevation Smoothed')


#ax.set_xlim(pd.Timestamp('2000-08-01'),pd.Timestamp('2001-08-01'))
#ax[0].set_xlim(43,46)
#ax[0].set_ylim(18,25)
ax[0].set_title('Comparison of Weir Elevation South Levee')
ax[0].set_ylabel('Elevation (NAVD88,m)')
ax[0].set_xticklabels([])
ax[0].legend()
ax[0].grid()


ax[1].plot(e2_weirs['ID_Thalweg']/1000,e2_weirs['Elev'],'k',label='Weir Elevation')
ax[1].plot(e2_weirs['ID_Thalweg']/1000,e2_weirs['Dem_Elev'],'r', label='Dem Elevation')
ax[1].plot(e2_weirs['ID_Thalweg']/1000,e2_weirs['Elev_smth'],'b',label='Weir Elevation Smoothed')

#ax.set_xlim(pd.Timestamp('2000-08-01'),pd.Timestamp('2001-08-01'))
#ax.set_xlim(100,200)
ax[1].grid()
#ax[1].set_xlim(43,46)
#ax[1].set_ylim(18,50)
ax[1].set_title('Comparison of Weir Elevation North Levee')
ax[1].set_ylabel('Elevation (NAVD88,m)')

fig.savefig(os.path.join(dir_figs,f'Dem_Compare_{river}.png'),  dpi=800,
        bbox_inches='tight')   

#===============================================================================
# %% Apply weir elevation to DEM
#===============================================================================

# Export 3d line with weir elevation
e1_weir_line = points_to_linestring(e1_weirs,Z_field='Elev_smth')
e2_weir_line = points_to_linestring(e2_weirs,Z_field='Elev_smth')

e1_weir_line.to_file(os.path.join(dir_temp,'e1_Line3D.shp'))
e2_weir_line.to_file(os.path.join(dir_temp,'e2_Line3D.shp'))

mask_e1 = burn_shape(file_raster,os.path.join(dir_temp,'e1_Line3D.shp'),dir_temp)
mask_e2 = burn_shape(file_raster,os.path.join(dir_temp,'e2_Line3D.shp'),dir_temp)

mask_weir = mask_e1 + mask_e2

kwargs = dem.meta
kwargs.update(
    dtype=rasterio.int32,
    count=1,
    compress='lzw',
    nodata = -9999)

with rasterio.open(os.path.join(dir_temp, 'weir_out.tif'), 'w', **kwargs) as dst:
    dst.write_band(1, mask_weir.astype(rasterio.int32))
    
    
# export modified DEM
file_raster_out = os.path.join(dir_in_Dem,f'{county}_2m_AddWeir.tif')

# Replace valuesfor weir locations
ind_replace = (mask_weir>0)

dem_elev_new = dem_elev
dem_elev_new[ind_replace] = mask_weir[ind_replace]

kwargs = dem.meta
with rasterio.open(file_raster_out, 'w', **kwargs) as dst:
    dst.write_band(1, dem_elev_new.astype(rasterio.float32))




