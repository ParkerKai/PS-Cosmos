# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:37:51 2024

This script exports DFM output locations as GIS points
and plots them for a quick visual check.

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
import geopandas as gpd
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
dir_in = r'D:\Kai\DFM\Combined_025'
dir_out = r'D:\Kai\DFM\GIS'

nc_path = os.path.join(dir_in, 'Reanalysis_and_Projected_CoSMoSwaterlevels_whatcom.nc')




#===============================================================================
# %% Load the Data
#===============================================================================
# Use open_dataset for a single file
data = xr.open_dataset(nc_path, engine='netcdf4')

# Try to find lon/lat variables by common names
lon_name_candidates = ['lon', 'longitude', 'x']
lat_name_candidates = ['lat', 'latitude', 'y']

def find_var(ds, candidates):
    for name in candidates:
        if name in ds.variables:
            return name
    # allow coords as well
    for name in candidates:
        if name in ds.coords:
            return name
    raise KeyError(f"None of {candidates} found in dataset variables/coords.")

lon_name = find_var(data, lon_name_candidates)
lat_name = find_var(data, lat_name_candidates)

lon_da = data[lon_name]
lat_da = data[lat_name]

# Station id/name: prefer a meaningful text field if present
station_candidates = ['station', 'station_id', 'id', 'name']
station_name = None
for cand in station_candidates:
    if cand in data.variables or cand in data.coords:
        station_name = cand
        break

#===============================================================================
# %% Handle dimensions (1D vs 2D lon/lat)
#===============================================================================
# We want 1D arrays of unique stations. If lon/lat are 2D (e.g., time x station),
# take the first time index or reduce along time if needed.
def ensure_1d_coords(lon_da, lat_da):
    # If they share dims and have 'station', extract it
    if 'station' in lon_da.dims and 'station' in lat_da.dims:
        # If there is also 'time' or other dims, squeeze/select a representative slice
        extra_dims = [d for d in lon_da.dims if d != 'station']
        if len(extra_dims) > 0:
            # Pick the first index along extra dims
            sel = {d: lon_da[d][0] for d in extra_dims}
            lon_1d = lon_da.sel(**sel)
            lat_1d = lat_da.sel(**sel)
        else:
            lon_1d = lon_da
            lat_1d = lat_da
        return np.asarray(lon_1d), np.asarray(lat_1d)
    # If 2D but share a common dimension, try to reduce
    if lon_da.ndim == 2 and lat_da.ndim == 2 and lon_da.shape == lat_da.shape:
        # Assume dims like (time, station); pick the first time
        # More robust: identify station-like dim
        dims = lon_da.dims
        if 'station' in dims:
            # index dims; keep station dimension
            other_dims = [d for d in dims if d != 'station']
            sel = {d: lon_da[d][0] for d in other_dims}
            lon_1d = lon_da.sel(**sel)
            lat_1d = lat_da.sel(**sel)
            return np.asarray(lon_1d), np.asarray(lat_1d)
        else:
            # fallback: take first row
            lon_1d = lon_da.isel({dims[0]: 0})
            lat_1d = lat_da.isel({dims[0]: 0})
            return np.asarray(lon_1d), np.asarray(lat_1d)
    # Already 1D
    if lon_da.ndim == 1 and lat_da.ndim == 1:
        return np.asarray(lon_da), np.asarray(lat_da)
    # Final fallback: ravel
    return np.asarray(lon_da).ravel(), np.asarray(lat_da).ravel()

lon_vals, lat_vals = ensure_1d_coords(lon_da, lat_da)

# Optionally grab station labels (ensure same length)
if station_name is not None:
    station_vals = data[station_name].values
    # If station is multi-dim, reduce similar to lon/lat
    if station_vals.ndim > 1:
        station_vals = station_vals.reshape(-1)[:len(lon_vals)]
    # Ensure string dtype without truncation
    station_vals = pd.Series(station_vals).astype(str).values
else:
    # Create a simple index-based station label
    station_vals = np.array([f"stn_{i:04d}" for i in range(len(lon_vals))])

# Build a DataFrame
df = pd.DataFrame({
    'station': station_vals,
    'lon': lon_vals,
    'lat': lat_vals
})

# Drop rows with invalid coordinates
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['lon', 'lat'])

# If there are duplicate stations or duplicate coordinates, consider deduping
df = df.drop_duplicates(subset=['station']).drop_duplicates(subset=['lon', 'lat'])

#===============================================================================
# %% Build GeoDataFrame and export
#===============================================================================
gdf = gpd.GeoDataFrame(
    df[['station']],
    geometry=gpd.points_from_xy(df['lon'], df['lat']),
    crs="EPSG:4326"
)

# Ensure output dir exists
os.makedirs(dir_out, exist_ok=True)

# Recommended: GeoPackage to preserve full station names & UTF-8
gpkg_path = os.path.join(dir_out, 'DFM_station_points.gpkg')
#gdf.to_file(gpkg_path, layer='stations', driver='GPKG')

# Optional: ESRI Shapefile (note: field names <= 10 chars; strings may be truncated)
shp_path = os.path.join(dir_out, 'DFM_station_points.shp')
try:
    gdf.to_file(shp_path, driver='ESRI Shapefile')
except Exception as e:
    print(f"Shapefile write failed (likely schema limits): {e}")

print(f"Wrote: {gpkg_path}")
print(f"Also attempted Shapefile: {shp_path}")

#===============================================================================
# %% Quick plot for QA
#===============================================================================
ax = gdf.plot(markersize=8, figsize=(8, 6), color='royalblue')
ax.set_title('DFM Station Locations (EPSG:4326)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.tight_layout()
plt.show()


