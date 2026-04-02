# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 10:59:52 2025

This script Exports everything to make a data release overview image for the sfincs miles.


@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"

# ===============================================================================
# %% Import Modules
# ===============================================================================

# Modules needed
import os
import numpy as np
from matplotlib import pyplot as plt
from hydromt_sfincs import SfincsModel, utils
import matplotlib
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio import features
from shapely.geometry import shape

county = "Pierce"

sfincs_root = r"D:\SFINCS_Pierce\100_low"  # Location of the PS-Cosmos codebase
dir_out = r"Y:\PS_Cosmos\02_models\SFINCS\shapefiles"


# ===============================================================================
# %% Read SFINCS inputs
# ===============================================================================

# Open the Model configuration files
print(sfincs_root)
mod = SfincsModel(sfincs_root, mode="r")

mod.read_subgrid()
x = mod.grid.x.values
y = mod.grid.y.values
x, y = np.meshgrid(x, y)
zb = mod.grid.dep.values
zb = mod.subgrid.z_zmin
zb = zb.values

forcing = mod.forcing

results = mod.read_results()


# ===============================================================================
# %% Output WL points
# ===============================================================================

x = forcing["bzs"].x.values
y = forcing["bzs"].y.values

gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs="EPSG:32610")
gdf.to_file(os.path.join(dir_out, f"SFINCS_Input_WlStat_{county}.shp"))

# ===============================================================================
# %% Output Discharge points
# ===============================================================================

data = pd.read_csv(
    os.path.join(sfincs_root, "sfincs.src"),
    sep="  ",
    skipinitialspace=True,
    dtype=np.float64,
    header=None,
    names=["X", "Y"],
)

x = data.iloc[:, 0].values
y = data.iloc[:, 1].values

gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs="EPSG:32610")
gdf.to_file(os.path.join(dir_out, f"SFINCS_Input_DischargeStat_{county}.shp"))


# ===============================================================================
# %% Output Discharge points
# ===============================================================================
x = mod.grid.x.values
y = mod.grid.y.values
zb = mod.grid.dep.values

# Assuming x and y are 1D arrays representing the grid centers
pixel_size_x = (x.max() - x.min()) / (len(x) - 1)
pixel_size_y = (y.max() - y.min()) / (len(y) - 1)

transform = from_origin(
    x.min() - pixel_size_x / 2,
    y.max()
    - (zb.shape[0] * pixel_size_y)
    + pixel_size_y / 2,  # shift origin south by raster height
    pixel_size_x,
    -pixel_size_y,
)

# Define writing
kwargs2 = dict(
    driver="GTiff",
    height=zb.shape[0],
    width=zb.shape[1],
    count=1,
    dtype="float64",
    crs="EPSG:32610",  # Assuming 'src' is defined from previous operations
    transform=transform,  # Assuming 'src' is defined from previous operations
    tiled=True,
    blockxsize=128,  # reduced this from 256
    blockysize=128,
    compress="deflate",
    predictor=2,  # Adjust based on your data's nature (floating-point or integer)
    zlevel=6,  # reduced to 6 from 9
    profile="COG",
)

# Do Write a raster of bathymeter
output_filename = os.path.join(dir_out, f"SFINCS_Grid_{county}.tif")
with rasterio.open(output_filename, "w", **kwargs2) as dst:
    dst.write(zb, 1)  # Write the first (and only) band

    from shapely.geometry import shape


output_filename = os.path.join(dir_out, f"SFINCS_Grid_{county}.tif")
with rasterio.open(output_filename, "r") as src:
    image = src.read(1)
    transform = src.transform

    # Extract polygons and values
    results = features.shapes(image, transform=transform)
    polygons = []
    values = []
    for geom, val in results:
        if not np.isnan(val):  # Skip nodata if present
            polygons.append(shape(geom))
            values.append(val)

# Create GeoDataFrame
gdf_poly = gpd.GeoDataFrame({"value": values, "geometry": polygons}, crs="EPSG:32610")

# Export to shapefile
gdf_poly.to_file(os.path.join(dir_out, f"SFINCS_Grid_{county}_polygons.shp"))

# %%
