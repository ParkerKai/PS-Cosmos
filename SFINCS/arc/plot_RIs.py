# -*- coding: utf-8 -*-
"""
Created on Sat Nov 01 13:06:21 2022

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
import arcpy
import pandas as pd
import numpy as np
from glob import glob

#===============================================================================
# %% User Defined inputs
#===============================================================================

# Group layer for which to set symbology
Region = '02_Pierce'

# Attribute to Set symbology to 
param = 'zsmax'

# Directory where the shapefiles reside
dir_in = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\PierceKing\20240531_synthetic'

RP_list =  ['001', '002', '005', '010','020','050', '100']


#===============================================================================
# %% Define some functions
#===============================================================================
sys.path.append(r'C:\Users\kaparker\Documents\GitHub\Kai_Python\ArcTools')
from Kai_ArcTools import read_shapefile,set_symbology_GradColors,get_layerHandle


# Figure out the break values

#print('Set colormap limits of {}'.format(lyr2[0]))
#print('{}:  Min: {}, Max: {}'.format(Attr,min_lim,max_lim))


#for cnt,lyr in enumerate(grp_lyr):
#    if (cnt > 0):
#        arcpy.management.ApplySymbologyFromLayer(lyr, grp_lyr[0],update_symbology='UPDATE')



#===============================================================================
# %% Import layers  
#===============================================================================
p = arcpy.mp.ArcGISProject('current')
m  = arcpy.mp.ArcGISProject("CURRENT").activeMap

# Files to import 
work_folder = os.path.join(dir_in,Region,'downscaled_2m')
files = glob(os.path.join(work_folder,f'*{param}*.tif'))

#Create folder for layer files
# lyrFol = os.path.join(dir_in,Region,'downscaled_2m_lyrs')
# if not os.path.exists(lyrFol):
  # # Create Final Merged Raster Folder if it doesn't exist
  # os.makedirs(junkFol)
  # print("Layer folder created")


# Build Pyramids
# arcpy.management.BuildPyramidsandStatistics(
    # in_workspace=work_folder,
    # include_subdirectories="INCLUDE_SUBDIRECTORIES",
    # build_pyramids="BUILD_PYRAMIDS",
    # calculate_statistics="CALCULATE_STATISTICS",
    # BUILD_ON_SOURCE="NONE",
    # block_field="",
    # estimate_statistics="NONE",
    # x_skip_factor=1,
    # y_skip_factor=1,
    # ignore_values=[],
    # pyramid_level=-1,
    # SKIP_FIRST="NONE",
    # resample_technique="NEAREST",
    # compression_type="DEFAULT",
    # compression_quality=75,
    # skip_existing="SKIP_EXISTING",
    # where_clause="",
    # sips_mode="NONE"
# )


# Create Group layer
grp_lyr = m.createGroupLayer(param)

# Have to do some tweaks because of nans in the zsmax tifs.
# Multipling by 1 makes the nans become nodata and display properly.
if param == 'zsmax':
    
    # Run "RasterCleanNans.py" to convert Nans to NoData (for display reasons).
        
    # re create "files" so it gets added in later 
    files = glob(os.path.join(work_folder,f'*{param}*_clean.tif'))
    
    if len(files) == 0:
        print('Run RasterCleanNans.py')
    
# Add all files to the map in a group layer 

for file in reversed(files):
    file_name = os.path.basename(os.path.normpath(file))
    if param == 'zsmax':
        lyr_name  = file_name.replace('_2m_clean.tif','')
    else:
        lyr_name  = file_name.replace('_2m.tif','')
    
    arcpy.MakeRasterLayer_management(file, lyr_name)
    
    lyr = get_layerHandle(lyr_name)
    m.addLayerToGroup(grp_lyr,lyr)
    m.removeLayer(lyr)



#===============================================================================
# %% Determine max/mins 
#===============================================================================
for cnt,lyr in enumerate(grp_lyr.listLayers()):

    # Get raster properties
    min_mod = arcpy.management.GetRasterProperties(lyr, 'MINIMUM')[0]
    max_mod = arcpy.management.GetRasterProperties(lyr, 'MAXIMUM')[0]

    # Convert to numpy numbers
    min_mod = np.array(min_mod).astype(np.float64)
    max_mod = np.array(max_mod).astype(np.float64)
    
    if (cnt == 0):
        min_lim = min_mod
        max_lim = max_mod
        
    else:
        min_lim = np.min(np.array([min_lim,min_mod]))
        max_lim = np.max(np.array([max_lim,max_mod]))

    
# # Round 
min_lim = np.round(min_lim,decimals=0)
max_lim = np.round(max_lim,decimals=0)


#===============================================================================
# %% change the map 
#===============================================================================


# Set the symbology color 
for cnt,lyr in enumerate(grp_lyr.listLayers()):

    sym = lyr.symbology
    sym.colorizer.colorRamp = p.listColorRamps("Bathymetry #3")[0]
    lyr.symbology = sym
    
    lyr.transparency=0.5
    
    
    