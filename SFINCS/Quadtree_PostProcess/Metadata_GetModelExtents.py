# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 10:59:52 2025

This script extracts needed metadata ouput from Cosmos outputs.


@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"

# ===============================================================================
# %% Import Modules
# ===============================================================================

import os
from glob import glob
import numpy as np
import geopandas as gpd
import hydromt_sfincs
from shapely.geometry import Point


# ===============================================================================
# %% User Defined inputs
# ===============================================================================

# Processing information
dir_in = r"Y:\PS_Cosmos\06_FinalProducts\02_Pierce\CoSMoS-PS_SFINCS_ModelInput_Pierce"


# ===============================================================================
# %% Define some functions
# ===============================================================================


# ===============================================================================
# %% get the file structure
# ===============================================================================
print(dir_in)


mod = hydromt_sfincs.SfincsModel(dir_in, mode="r")


mod.grid.y.min()


# Create a list of (x, y) tuples for each bounding coordinate
coords = [
    (mod.grid.x.min(), mod.grid.y.min()),  # West-South
    (mod.grid.x.min(), mod.grid.y.max()),  # West-North
    (mod.grid.x.max(), mod.grid.y.min()),  # East-South
    (mod.grid.x.max(), mod.grid.y.max()),  # East-North
]

# Create Point geometries
geometry = [Point(x, y) for x, y in coords]

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(
    geometry=geometry, crs="EPSG:32610"
)  # Replace with your current CRS

# Project to geographic coordinate system (e.g., WGS84)
gdf_geo = gdf.to_crs("EPSG:4326")

print(f"        West_Bounding_Coordinate: {gdf_geo.bounds['minx'].min():.6f}")
print(f"        East_Bounding_Coordinate: {gdf_geo.bounds['maxx'].max():.6f}")
print(f"        North_Bounding_Coordinate: {gdf_geo.bounds['maxy'].max():.6f}")
print(f"        South_Bounding_Coordinate: {gdf_geo.bounds['miny'].min():.6f}")

print("")
