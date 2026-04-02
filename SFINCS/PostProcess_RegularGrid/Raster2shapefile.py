# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:36:28 2024
File originally for N. VanArendonk, S. C. Crosby

Convert mask rasters to polygon extents
Helpful link, https://anitagraser.com/2019/03/03/stand-alone-pyqgis-scripts-with-osgeo4w/

environment: qgis

@author: kaparker
kaparker@usgs.gov
"""

# Import Modules
from os.path import isfile, join
import os
import sys
from osgeo import gdal
from qgis.core import QgsApplication, QgsProcessingFeedback

# from qgis.gui import *
# from qgis.analysis import QgsNativeAlgorithms
import glob

# os.chdir(r'Y:\PS_Cosmos\02_models\Cosmos_WhatcomNWIFC\PS_Cosmos_OldCodebase\02_post_processing\part06_merge_raster_shapefiles')
sys.path.append(
    r"Y:\PS_Cosmos\02_models\Cosmos_WhatcomNWIFC\PS_Cosmos_OldCodebase\02_post_processing\part06_merge_raster_shapefiles"
)
from qgis_fxns import (
    cleanup_folder,
    merge_netcdf_no_tiles,
    polygonize_raster,
    filter_polygons,
    fix_geometries,
)


# ===============================================================================
# %% User Defined inputs
# ===============================================================================

# USER Specific locations of processessing and python libraries - MUST BE CHANGED FOR LOCAL PATHS
processPath = r"C:\Program Files\QGIS 3.28.15\apps\qgis-ltr\python\plugins\processing"
scriptPath = r"C:\Program Files\QGIS 3.28.15\apps\Python39\Scripts"

# Load the paths
base_dir = r"Y:\PS_Cosmos\02_models\SFINCS\20250122_synthetic_future_meanchange_100yr_Intel\PostProcess\02_Pierce"  # Location of the PS-Cosmos codebase

scenario_list = [
    "000",
    "025",
    "050",
    "100",
    "150",
    "200",
    "300",
    "000_low",
    "025_low",
    "050_low",
    "100_low",
    "150_low",
    "200_low",
    "300_low",
    "000_high",
    "025_high",
    "050_high",
    "100_high",
    "150_high",
    "200_high",
    "300_high",
]

scenario_list = ["100_low"]

# ===============================================================================
# %% Initialize
# ===============================================================================

# Initialize QGIS & Toolboxes
QgsApplication.setPrefixPath("/usr", True)
qgs = QgsApplication([], False)
qgs.initQgis()

# Add paths to necessary libraries
sys.path.append(processPath)
sys.path.append(scriptPath)

# Add the processing toolbox
from processing.core.Processing import Processing

Processing.initialize()

# What is this? Not sure but we need it - This mostly gives meta on the processing step....might be able to remove
feedback = QgsProcessingFeedback()

for scenario in scenario_list:
    print(f"Processing {scenario}")

    # INPUTS
    netFol = os.path.join(base_dir, scenario, "downscaled_2m")

    # OUTPUTS
    datOutFol = os.path.join(base_dir, scenario, "final_shapefile")  # Output shapefiles

    # ===============================================================================
    # %% Process
    # ===============================================================================
    # JUNK
    junkFol = os.path.join(base_dir, "junk", scenario)  # Temporary folder

    if not os.path.exists(datOutFol):
        # Create Final Merged Raster Folder if it doesn't exist
        os.makedirs(datOutFol)
        print("New final_shapefile folder created")

    if not os.path.exists(junkFol):
        # Create Final Merged Raster Folder if it doesn't exist
        os.makedirs(junkFol)
        print("New junk folder created")
    else:
        files = glob.glob(os.path.join(junkFol, "*"))
        for f in files:
            os.remove(f)

    # Get list of all netcdf files now
    all_files = [
        f for f in os.listdir(netFol) if isfile(join(netFol, f))
    ]  # Finds all files
    netcdf_files = [
        f for f in all_files if f.startswith("connection_")
    ]  # Finds all netcdf files
    netcdf_files = [
        f for f in netcdf_files if f.endswith("_attentuated.tif")
    ]  # Finds all netcdf files

    # netcdf_files = glob(os.path.join(base_dir,'wetdry_*'))

    # Uncomment to test single file
    # ncf = [netcdf_files[0]]

    # Loop over rasters
    for ncf in netcdf_files:
        # #ncf = netcdf_files[0]
        # # Merge tiles and make a single raster ( There is no merging here?)
        # vName = ncf.replace('.nc','.tiff') # Name to be passed to function
        # fullNetName = os.path.join(netFol,ncf) # Full path to file
        # fullRasName = os.path.join(datOutFol,vName) # Full path to output

        # # Options for the Raster merging - use as many/little as we want
        # EPSG        = 'EPSG:6339' # Defines SRS for dataset
        # options     = gdal.BuildVRTOptions(outputSRS=EPSG)

        # # Convert from nc to tiff store in intermediate junk folder
        # mrgRast     = merge_netcdf_no_tiles(fullNetName, junkFol, vName, options)
        # print('\n')
        # print('netCDF %s successfully loaded, merged and saved in %s\n' % (fullNetName,junkFol))

        # Name of shape file to output
        pName = ncf.replace("_attentuated.tif", "_poly.shp")

        # Converts mask raster to polygon with labels for 1 and 2 (conn/disconn) and stores in junk folder
        polygonize_raster(netFol, junkFol, ncf, pName)

        # Filter Polygons for connected (DN=1) and disconnected (DN=2).  This DN code is automatic somehow (in matlab we have 1 for flood, 2 for disconnected)
        #     This takes polygon with multiple shape/flood types (1,2) and makes individual files
        fullPolygonPath = os.path.join(junkFol, pName)  # Location of polygon file
        ext1 = ncf.replace("_2m_attentuated.tif", "_connected.shp")
        ext1 = ext1.replace("connection", "flood")
        ext2 = ncf.replace("_2m_attentuated.tif", "_disconnected.shp")
        ext2 = ext2.replace("connection", "flood")

        extOut1 = os.path.join(junkFol, ext1)  # Name of filtered data for DN = 1
        extOut2 = os.path.join(junkFol, ext2)  # Name of filtered data for DN = 2
        exp1 = "DN = 1"  # Expression for DN = 1
        exp2 = "DN = 2"  # Expression for DN = 2

        # polygonPath=fullPolygonPath
        # outName    =extOut1
        # expression =exp1
        # feedback   =feedback
        # processPath=processPath
        # scriptPath =scriptPath

        # import processing
        # from processing.core.Processing import Processing
        # Processing.initialize()

        # extract = \
        # processing.run('native:extractbyexpression', {'INPUT': polygonPath, 'EXPRESSION': expression, 'OUTPUT': outName},
        #                feedback=feedback)['OUTPUT']
        # state = '%s extracted from %s' % (expression, polygonPath)
        # print(state)
        # print('\n')

        filter_polygons(
            fullPolygonPath, extOut1, exp1, feedback, processPath, scriptPath
        )
        filter_polygons(
            fullPolygonPath, extOut2, exp2, feedback, processPath, scriptPath
        )

        # Fix Geometries, sometimes shape files look funny in QGIS (less so in ArcGIS) but this fixed issues
        fixed1 = os.path.join(datOutFol, ext1)
        fixed2 = os.path.join(datOutFol, ext2)
        fix_geometries(extOut1, fixed1, processPath, scriptPath)
        fix_geometries(extOut2, fixed2, processPath, scriptPath)
