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
import rioxarray
import geopandas as gpd
from shapely.geometry import Point, box

# ===============================================================================
# %% User Defined inputs
# ===============================================================================

# Processing information
dir_in = r"Y:\PS_Cosmos\06_FinalProducts\02_Pierce"
RPs = ["000", "001", "010", "020", "050", "100"]
RP_OutName = [
    "average_conditions",
    " 1-year_storm",
    "10-year_storm",
    "20-year_storm",
    "50-year_storm",
    "100-year_storm",
]
SLRs = ["000", "025", "050", "100", "150", "200", "300"]

county = "Pierce"


# Directory out for extent information
dir_out = r"Y:\PS_Cosmos\06_FinalProducts\02_Pierce\metadata\Extents"

# Settings
# f"CoSMoS-PS_velocity_hazard_projections_{county}"
# f"CoSMoS-PS_water_elevation_projections_{county}"
# f"CoSMoS-PS_flood_depth_projections_{county}"
# f'CoSMoS-PS_flood_extent_and_uncertainty_projections_{county}'
# f"CoSMoS-PS_flood_duration_projections_{county}"

var = f'CoSMoS-PS_flood_extent_and_uncertainty_projections_{county}'


# ===============================================================================
# %% Define some functions
# ===============================================================================
def get_Bounds(files):
    WBC = np.full(len(files), np.nan)
    EBC = np.full(len(files), np.nan)
    NBC = np.full(len(files), np.nan)
    SBC = np.full(len(files), np.nan)
    for cnt, file in enumerate(files):
        # Handle case if file is a raster or shapefile
        root, ext = os.path.splitext(file)
        if ext == ".tif":
            xds = rioxarray.open_rasterio(file)

            # Get bounds
            bnds = xds.rio.bounds()  # minimum x (left), minimum y (bottom), maximum x (right), and maximum y (top)

        elif ext == ".shp":
            gdf = gpd.read_file(file)

            # Get bounds
            bnds = gdf.total_bounds

        # Get bounds in the format we want it
        WBC[cnt] = bnds[0]
        EBC[cnt] = bnds[2]
        NBC[cnt] = bnds[3]
        SBC[cnt] = bnds[1]

    # Export as a tuple
    return WBC.min(), EBC.max(), NBC.max(), SBC.min()


def get_DataRange(files):
    val_min = np.full(len(files), np.nan)
    val_max = np.full(len(files), np.nan)

    for cnt, file in enumerate(files):
        # Handle case if file is a raster or shapefile
        root, ext = os.path.splitext(file)
        if ext == ".tif":
            xds = rioxarray.open_rasterio(file)

            # Get Min/Max Values
            val_min[cnt] = xds.where(xds != xds._FillValue, other=np.nan).min().item()
            val_max[cnt] = xds.where(xds != xds._FillValue, other=np.nan).max().item()

        elif ext == ".shp":
            val_min[cnt] = np.nan
            val_max[cnt] = np.nan

    return val_min.min(), val_max.max()


# ===============================================================================
# %% get the file structure
# ===============================================================================

files = []


if "flood_extent" in var:
    for RP in RP_OutName:
        files.extend(
            glob(
                os.path.join(
                    dir_in,
                    var,
                    f"CoSMoS-PS_flood_extent_projections_{RP}_{county}",
                    "*.shp",
                )
            )
        )
        files.extend(
            glob(
                os.path.join(
                    dir_in,
                    var,
                    f"CoSMoS-PS_flood_uncertainty_projections_{RP}_{county}",
                    "*.shp",
                )
            )
        )

elif "velocity_hazard" in var:
    for RP in RP_OutName:
        files.extend(
            glob(
                os.path.join(
                    dir_in, var, var.replace(f"_{county}", f"_{RP}_{county}"), "*.shp"
                )
            )
        )

else:
    for RP in RP_OutName:
        files.extend(
            glob(
                os.path.join(
                    dir_in, var, var.replace(f"_{county}", f"_{RP}_{county}"), "*.tif"
                )
            )
        )


if len(files) == 0:
    print(f"No files found for {var}")


# Bounding box
WBC, EBC, NBC, SBC = get_Bounds(files)

# Turn into a GeoDataFrame

# Create a list of (x, y) tuples for each bounding coordinate
coords = [
    (WBC, SBC),  # West-South
    (WBC, NBC),  # West-North
    (EBC, SBC),  # East-South
    (EBC, NBC),  # East-North
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


# Export as a shapefile for the data release
minx, miny, maxx, maxy = gdf_geo.total_bounds
bbox_poly = box(minx, miny, maxx, maxy)
bbox_poly = box(minx, miny, maxx, maxy)

bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox_poly]}, crs=gdf_geo.crs)

bbox_gdf.to_file(os.path.join(dir_out, f"Bounding_Box_{var}.shp"))

# Min/Max Values
MaxVal, MinVal = get_DataRange(files)

print(f"            Range_Domain_Minimum: {MaxVal:.2f}")
print(f"            Range_Domain_Maximum: {MinVal:.2f}")
