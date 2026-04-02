# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 14:19:36 2025

Creates a QAQC map for Cosmos.


****************************************************************************************

@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"

#===============================================================================
# %% Import Modules
#===============================================================================
import arcpy
from arcpy import env
import os
import numpy as np
# Reset geoprocessing environment settings to clear extent and mask setting from other scripts
arcpy.ResetEnvironments()
p = arcpy.mp.ArcGISProject('current')
m  = arcpy.mp.ArcGISProject("CURRENT").activeMap
env.overwriteOutput = "True"

sys.path.append(r'C:\Users\kaparker\GitHub\Python\Kai_Python\ArcTools')
from Kai_ArcTools import get_layerHandle


#===============================================================================
# %% User Parameters
#===============================================================================

#select a single storm RP you want to use in the new output map  
RP = "100"   # "RP000" "RP001" "RP020" OR "RP100"

county = 'Pierce'

####### Directories ######
#directory containing data you want in output map
dir_root = r"Y:\PS_Cosmos\06_FinalProducts\02_Pierce" 
#directory containing template layers
dir_templates = r'Y:\PS_Cosmos\GIS\Shapefiles\general\QAQC_Templates'

#set up to run through entire SLR suite for a single storm RP
SLR_list = ["000", "025", "050", "100", "150", "200", "300"]
#SLR_list = ["000", "050", "100",  "200"]


# Features to add to the QAQC map
LowLying        = True      # Add LowLying layer?  
Rasters         = True      # Raster Products 
VelocityDepth   = True      # Velocity Depth Shapefiles

# Raster vars to process
Raster_list = ['flood_depth','water_elevation','flood_duration']
    
 
 
if RP == '000':
    RP_name = 'average_conditions'
    
elif RP == '001':
    RP_name = '1-year_storm'

elif RP == '010':
    RP_name = '10-year_storm'

elif RP == '020':
    RP_name = '20-year_storm'

elif RP == '050':
    RP_name = '50-year_storm'
  
elif RP == '100':
    RP_name = '100-year_storm'

else:
    print('Incorrect RP designation')
 
#===============================================================================
# %% VelocityDepth Shapefiles
#===============================================================================
    
if VelocityDepth:
    # Create grouplayer 
    grp_lyr = m.createGroupLayer('VelocityDepth')
    
    for SLR in np.flip(SLR_list):
        SLR_var = f'SLR{SLR}'
        print(f'Processing: {SLR_var} Shapefiles')
                
        ############ Velocity shapefile ###########
        lyr = m.addDataFromPath(os.path.join(dir_root,f'CoSMoS-PS_velocity_hazard_projections_{county}',f'CoSMoS-PS_velocity_hazard_projections_{RP_name}_{county}',f'{county}_velHzrd_slr{SLR}_rp{RP}.shp'))
        lyr.name = lyr.name.replace('.shp','')
        
        # Set symbology using topography 
        lyr_template = m.addDataFromPath(os.path.join(dir_templates,'template_DepthVelocity.lyrx'))
        arcpy.ApplySymbologyFromLayer_management(lyr, lyr_template)
        m.removeLayer(lyr_template)
        lyr.transparency=20


        # Add Layer to group
        m.addLayerToGroup(grp_lyr,lyr)
        m.removeLayer(lyr)
            
            
 
#===============================================================================
# %% Rasters 
#===============================================================================
if Rasters: 
    
    for Raster_var in Raster_list:
        
                
        ############ Load the raster ###########
        if Raster_var == 'flood_depth':
            folder_name = 'CoSMoS-PS_flood_depth_projections'
            Raster_name = 'fldDpth'
            colorRamp   = "Blue Bright"
            
        elif Raster_var == 'water_elevation':
            folder_name = 'CoSMoS-PS_water_elevation_projections'
            Raster_name = 'wsel'
            colorRamp   = "Green Light to Dark"
            
        elif Raster_var == 'flood_duration':
            folder_name = 'CoSMoS-PS_flood_duration_projections'
            Raster_name = 'duration'
            colorRamp   = "Orange Bright"
        
        
        print(f'Building Pyramids for {Raster_var}')
        
        # Build Pyramids
        arcpy.management.BuildPyramidsandStatistics(
            in_workspace=os.path.join(dir_root, f'{folder_name}_{county}'),
            include_subdirectories="INCLUDE_SUBDIRECTORIES",
            build_pyramids="BUILD_PYRAMIDS",
            calculate_statistics="CALCULATE_STATISTICS",
            BUILD_ON_SOURCE="NONE",
            block_field="",
            estimate_statistics="NONE",
            x_skip_factor=1,
            y_skip_factor=1,
            ignore_values=[],
            pyramid_level=-1,
            SKIP_FIRST="NONE",
            resample_technique="NEAREST",
            compression_type="DEFAULT",
            compression_quality=75,
            skip_existing="SKIP_EXISTING",
            where_clause="",
            sips_mode="NONE"
        )


        # Create grouplayer 
        grp_lyr = m.createGroupLayer(f'{Raster_var}')


        for SLR in np.flip(SLR_list):
            SLR_var = f'SLR{SLR}'
            print(f'Processing: {SLR_var} {Raster_var} Rasters')
            

                
            raster_in = os.path.join(dir_root, f'{folder_name}_{county}',f'{folder_name}_{RP_name}_{county}',f'{county}_{Raster_name}_slr{SLR}_rp{RP}.tif')
            file_name = os.path.basename(os.path.normpath(raster_in))
            lyr_name  = file_name.replace('.tif','')
                
   
            
            arcpy.MakeRasterLayer_management(raster_in, lyr_name)
            
            lyr = get_layerHandle(lyr_name)
            m.addLayerToGroup(grp_lyr,lyr)
            m.removeLayer(lyr)
                        


        # # Have to do some tweaks because of nans in the zsmax tifs.
        # # Multipling by 1 makes the nans become nodata and display properly.
        # if param == 'zsmax':
            
            # # Run "RasterCleanNans.py" to convert Nans to NoData (for display reasons).
                
            # # re create "files" so it gets added in later 
            # files = glob(os.path.join(work_folder,f'*{param}*_clean.tif'))
            
            # if len(files) == 0:
                # print('Run RasterCleanNans.py')
         


        #===============================================================================
        # %% Determine max/mins 
        #===============================================================================
        # for cnt,lyr in enumerate(grp_lyr.listLayers()):
            
            # # Get raster properties
            # min_mod = arcpy.management.GetRasterProperties(lyr, 'MINIMUM')[0]
            # max_mod = arcpy.management.GetRasterProperties(lyr, 'MAXIMUM')[0]

            # # Convert to numpy numbers
            # min_mod = np.array(min_mod).astype(np.float64)
            # max_mod = np.array(max_mod).astype(np.float64)
            
            # if (cnt == 0):
                # min_lim = min_mod
                # max_lim = max_mod
                
            # else:
                # min_lim = np.min(np.array([min_lim,min_mod]))
                # max_lim = np.max(np.array([max_lim,max_mod]))

            
        # # # Round 
        # min_lim = np.round(min_lim,decimals=0)
        # max_lim = np.round(max_lim,decimals=0)


        #===============================================================================
        # %% change the map 
        #===============================================================================


        # Set the symbology color 
        for cnt,lyr in enumerate(grp_lyr.listLayers()):

            sym = lyr.symbology
            sym.colorizer.colorRamp = p.listColorRamps(colorRamp)[0]
            lyr.symbology = sym
            
            lyr.transparency=0.5
            
        
    

#===============================================================================
# %% Flood Extent shapefiles
#===============================================================================
    

for SLR in np.flip(SLR_list):
    SLR_var = f'SLR{SLR}'
    print(f'Processing: {SLR_var} Shapefiles')
    
    # Create grouplayer 
    grp_lyr = m.createGroupLayer(f'{SLR_var}_FloodExtent')
    
    ############ LowLying shapefile ###########
    if LowLying:
        file_in = os.path.join(dir_root,f'CoSMoS-PS_flood_extent_and_uncertainty_projections_{county}',f'CoSMoS-PS_flood_extent_projections_{RP_name}_{county}',f'{county}_lowlyingPoly_slr{SLR}_rp{RP}.shp')
        lyr = m.addDataFromPath(file_in)
        lyr.name = lyr.name.replace('_disconnected','_LowLying')
        
        # Set symbology using topography 
        lyr_template = m.addDataFromPath(os.path.join(dir_templates,'template_LowLyingFlooding.lyrx'))
        arcpy.ApplySymbologyFromLayer_management(lyr, lyr_template)
        m.removeLayer(lyr_template)
        lyr.transparency=20


        # Add Layer to group
        m.addLayerToGroup(grp_lyr,lyr)
        m.removeLayer(lyr)
    
    ############ High shapefile ###########
    file_in = os.path.join(dir_root,f'CoSMoS-PS_flood_extent_and_uncertainty_projections_{county}',f'CoSMoS-PS_flood_uncertainty_projections_{RP_name}_{county}',f'{county}_max_flood_slr{SLR}_rp{RP}.shp')
    lyr = m.addDataFromPath(file_in)
    lyr.name = lyr.name.replace('_connected','_high')
    
    # Set symbology using topography 
    lyr_template = m.addDataFromPath(os.path.join(dir_templates,'template_InnundationHigh.lyrx'))
    arcpy.ApplySymbologyFromLayer_management(lyr, lyr_template)
    m.removeLayer(lyr_template)
    lyr.transparency=20


    # Add Layer to group
    m.addLayerToGroup(grp_lyr,lyr)
    m.removeLayer(lyr)

    ############ Medium shapefile ###########
    file_in = os.path.join(dir_root,f'CoSMoS-PS_flood_extent_and_uncertainty_projections_{county}',f'CoSMoS-PS_flood_extent_projections_{RP_name}_{county}',f'{county}_fldPoly_slr{SLR}_rp{RP}.shp')
    lyr = m.addDataFromPath(file_in)
    lyr.name = lyr.name.replace('_connected','_med')
 
    lyr.transparency=20

    # Add Layer to group
    m.addLayerToGroup(grp_lyr,lyr)
    m.removeLayer(lyr)
    
 
    ############ Low shapefile ###########
    file_in = os.path.join(dir_root,f'CoSMoS-PS_flood_extent_and_uncertainty_projections_{county}',f'CoSMoS-PS_flood_uncertainty_projections_{RP_name}_{county}',f'{county}_min_flood_slr{SLR}_rp{RP}.shp')
    lyr = m.addDataFromPath(file_in)
    lyr.name = lyr.name.replace('_connected','_low')
    
    # Set symbology using topography 
    lyr_template = m.addDataFromPath(os.path.join(dir_templates,'template_InnundationLow.lyrx'))
    arcpy.ApplySymbologyFromLayer_management(lyr, lyr_template)
    m.removeLayer(lyr_template)

    lyr.transparency=20

    # Add Layer to group
    m.addLayerToGroup(grp_lyr,lyr)
    m.removeLayer(lyr) 
