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

#sys.path.append(r'C:\Users\kaparker\Documents\GitHub\Python\Kai_Python\ArcTools')
#from Kai_ArcTools import read_shapefile,set_symbology_GradColors,get_layerHandle


#===============================================================================
# %% User Parameters
#===============================================================================

#select a single storm RP you want to use in the new output map 
RP = "100"   # "RP000" "RP001" "RP020" OR "RP100"

####### Directories ######
#directory containing data you want in output map
dir_root = r"Y:\PS_Cosmos\02_models\SFINCS\20250122_synthetic_future_meanchange_100yr_Intel\PostProcess\02_Pierce" 
#directory containing template layers
dir_templates = r'Y:\PS_Cosmos\GIS\Shapefiles\general\QAQC_Templates'

#set up to run through entire SLR suite for a single storm RP
SLR_list = ["000", "025", "050", "100", "150", "200", "300"]

# Add LowLying layer?  
LowLying = True


#===============================================================================
# %% Process
#===============================================================================
    

for SLR in np.flip(SLR_list):
    SLR_var = f'SLR{SLR}'
    print(f'Processing: {SLR_var}')
    
    # Create grouplayer 
    grp_lyr = m.createGroupLayer(SLR_var)
    
    ############ LowLying shapefile ###########
    if LowLying:
        lyr = m.addDataFromPath(os.path.join(dir_root,f'{SLR}', 'final_shapefile',f'flood_{RP}_disconnected.shp'))
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
    lyr = m.addDataFromPath(os.path.join(dir_root,f'{SLR}_high', 'final_shapefile',f'flood_{RP}_connected.shp'))
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
    lyr = m.addDataFromPath(os.path.join(dir_root,f'{SLR}', 'final_shapefile',f'flood_{RP}_connected.shp'))
    lyr.name = lyr.name.replace('_connected','_med')
 
    lyr.transparency=20

    # Add Layer to group
    m.addLayerToGroup(grp_lyr,lyr)
    m.removeLayer(lyr)
    
 
    ############ Low shapefile ###########
    lyr = m.addDataFromPath(os.path.join(dir_root,f'{SLR}_low', 'final_shapefile',f'flood_{RP}_connected.shp'))
    lyr.name = lyr.name.replace('_connected','_low')
    
    # Set symbology using topography 
    lyr_template = m.addDataFromPath(os.path.join(dir_templates,'template_InnundationLow.lyrx'))
    arcpy.ApplySymbologyFromLayer_management(lyr, lyr_template)
    m.removeLayer(lyr_template)

    lyr.transparency=20

    # Add Layer to group
    m.addLayerToGroup(grp_lyr,lyr)
    m.removeLayer(lyr) 

    
    