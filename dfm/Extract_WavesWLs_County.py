#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extract_WavesWLs_County.py

This script extracts wave and water level data for a specified county from the Combined_DFM dataset.
Specifically it aggregates the data to the county level and outputs as a single netcdf.

This is intend to feed into SFINCS for county level modelling

__author__ = Kai Parker (USGS)
__email__ = kaparker@usgs.gov
__status__ = Dev
__created__ = 2025-11-18
"""

# ===============================================================================
# %% Import Modules
# ===============================================================================

import xarray as xr
from glob import glob
import os
import geopandas as gpd
from shapely import Point
import numpy as np
import matplotlib.pyplot as plt

# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the DFM data resides
# dir_in = r'D:\DFM'
dir_in_data = r"D:\Combined_DFM\ERA5"
dir_in_gis = r"Y:\PS_Cosmos\GIS\general"
dir_out = r"D:\Combined_DFM\ERA5_ByCounty"


county_pull = "SNOHOMISH"


# ===============================================================================
# %% Read in the county shapefile and subset
# ===============================================================================

# Read in the counties
Counties = gpd.read_file(
    os.path.join(dir_in_gis, "Washington_Counties_(no_water)___washco_area.shp")
)

# Subset to county of interest
county = Counties[Counties["COUNTY"] == county_pull]


# ===============================================================================
# %% Read in the data
# ===============================================================================


files = glob(os.path.join(dir_in_data, "*.nc"))

dropVar = [
    "waterlevel",
    "bedlevel",
    "tide",
    "ntr",
    "Hs",
    "Dm",
    "Tp",
    "lat_wave",
    "lon_wave",
    "depth_wave",
    "stat_wave",
]

is_within = np.full(len(files), False)
for cnt, file in enumerate(files):
    print(cnt)
    ds = xr.open_mfdataset(file, drop_variables=dropVar)

    # Create a list of shapely Point geometries
    # geometry = [Point(lon, lat) for lon, lat in zip(ds['lon_wl'].values, ds['lat_wl'].values)]
    geometry = [Point(ds["lon_wl"].values, ds["lat_wl"].values)]

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=geometry)

    # Set the coordinate reference system (CRS) to WGS84 (EPSG:4326)
    gdf.set_crs(epsg=4326, inplace=True)

    is_within[cnt] = gdf.within(county["geometry"], align=False).values


# Filter to just files within the county
files_county = [file for file, m in zip(files, is_within) if m]

ds = xr.open_mfdataset(files_county, parallel=True,
                       combine="nested",
                       concat_dim="station")



# ===============================================================================
# %% output the Data 
# ===============================================================================
ds.to_netcdf(os.path.join(dir_out,'Regional_ERA5_WlWaves.nc'))


# ===============================================================================
# %% plot for QAQC  
# ===============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 7))


 # Create a list of shapely Point geometries
geometry = [Point(lon, lat) for lon, lat in zip(ds['lon_wl'].values, ds['lat_wl'].values)]

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(geometry=geometry)

# Set the coordinate reference system (CRS) to WGS84 (EPSG:4326)
gdf.set_crs(epsg=4326, inplace=True)


county.plot(ax = ax, color='lightblue', edgecolor='black')

gdf.plot(ax = ax, color='black', markersize=5)


# Add a title to the plot
ax.set_title("DFM points")

# Display the plot
plt.show()
