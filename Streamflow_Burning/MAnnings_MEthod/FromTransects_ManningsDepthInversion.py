#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stilliguamish Bathy Fix

This script ingests GIS data (made by Amy Foxgrover)
It then uses mannings equation to invert a depth for the channel.


__author__ = Kai Parker (USGS)
__email__ = kaparker@usgs.gov
__status__ = Dev
__created__ = 2026-02-04
"""

# ===============================================================================
# %% Import Modules
# ===============================================================================
import geopandas as gpd
import os
import pandas as pd
from scipy.ndimage import gaussian_filter1d, minimum_filter
import xarray as xr 
import matplotlib
from matplotlib import pyplot as plt    
import numpy as np

# ===============================================================================
# %% User Defined inputs
# ===============================================================================
dir_in = r"Y:\PS_Cosmos\01_data\topo_bathymetry\Stillaguamish_StreamBurn\StilliMain_Xsecs_50m_to_I5"
dir_in_wflow = r'Y:\PS_Cosmos\02_models\WFLOW\11_20_2025_Discharges_SnohomishKitsap\snohomish'
dir_out = r'Y:\PS_Cosmos\02_models\SFINCS\Snohomish\gis\Stillaguamish_Transects'


n = 0.03  # mannings n

# correction to increase or decrease depth (per simulations)
BC_Correction = 0.7 # positive is increase depth.

# ===============================================================================
# %% Define some functions
# ===============================================================================


def manning_depth_wide_rectangular(
    Q: float,
    b: float,
    n: float = 0.030,  # Default Mannings  (typically range from 0.025 to 0.045 depending on vegetatino, bed materials, etc)
    S: float = 0.001,  # Default bed slope (set to moderate streams) (0.01 for steep mountain stream, 0.0001 for flat rivers)
):
    """
    Compute flow depth y (m) for a wide rectangular channel using Manning's equation
    rearranged for y, assuming hydraulic radius R ≈ y (wide rectangular channel).

    Formula:
        y = ((Q * n) / (b * sqrt(S))) ** (3/5)

    Parameters
    ----------
    Q : float
        Discharge [m^3/s]. Must be > 0.
    b : float
        Channel bottom width [m]. Must be > 0.
    n : float
        Manning's roughness coefficient [-]. Must be > 0.
    S : float
        Energy grade line (bed) slope [-], typically 0 < S << 1. Must be > 0.

    Returns
    -------
    (y, V) : Tuple[float, Optional[float]]
        y: Flow depth [m]
        V: Mean velocity [m/s] if return_velocity=True, otherwise None.
    """
    from math import sqrt

    # Basic validations
    for name, val in [("Q", Q), ("b", b), ("n", n), ("S", S)]:
        if val <= 0:
            raise ValueError(f"{name} must be > 0. Got {val}.")

    # Rearranged Manning's equation for depth
    y = ((Q * n) / (b * sqrt(S))) ** (3.0 / 5.0)

    return y


# ===============================================================================
# %% Load the WFLOW info for the time of the lidar survey 
# ===============================================================================
wflow = xr.open_dataset(os.path.join(dir_in_wflow, "output_scalar.nc"))

# Subset to the gauge of interest (located just upstream of where the transects start)
wflow = wflow['Q_contour'].sel(Q_contour_gauges_contour='8')



fig, [ax1,ax2] = plt.subplots(2, 1, figsize=[8, 5])


ax1.plot(
    wflow["time"],
    wflow,
    "w",
    label="Wflow",
)


ax1.grid()
ax1.set_xlim([pd.Timestamp('2016-03-17'), pd.Timestamp('2016-03-18')])
ax1.set_ylabel('Q (m3/s)')


ax2.plot(
    wflow["time"],
    wflow,
    "w",
    label="Wflow",
)


ax2.grid()
ax2.set_xlim([pd.Timestamp('2016-03-29'), pd.Timestamp('2016-04-01')])
ax2.set_ylabel('Q (m3/s)')



wflow_slice_per1 = wflow.sel(time=slice(pd.Timestamp('2016-03-17'), pd.Timestamp('2016-03-18'))).mean(dim='time').values
wflow_slice_per2 = wflow.sel(time=slice(pd.Timestamp('2016-03-29'), pd.Timestamp('2016-04-01'))).mean(dim='time').values


Q = np.mean([wflow_slice_per1, wflow_slice_per2])  # in m3/s

# ===============================================================================
# %% Calculate the Depth
# ===============================================================================


# There are attribute fields for the shapefile:
# Transect ID (ID_E2W)
# Transect Length (Length_m)
# Elevation of the hydro-flattened surface, in meters relative to NAVD88 (HydoElev_m)
# Right now, the NAVD88_m field only has a single assigned elevation for transect #398, on the tidal flats.
ds = gpd.read_file(os.path.join(dir_in, "StilliMain_Xsecs_50m_to_I5.shp"))


# Reorder by ID_E2W
ds = ds.sort_values(by="ID_E2W")

trans_resample = 10  # resample every 10 transects (so every 500m)
mask = np.arange(len(ds)) % 10 == 0   # True for 0,10,20,...

ds_resamp = ds[mask]


# Calculate the slope (assume Slope of the surface is equal to slope of the bottom....)
ds_resamp["S"] = -ds_resamp["HydoElev_m"].diff() / (50*trans_resample)  # 50m spacing between transects
ds["S"] = np.nan  # initialize with NaN
ds.loc[ds_resamp.index, 'S'] = ds_resamp['S']

ds['S'] = ds['S'].interpolate(method='linear', limit_direction='both', inplace=False)

# Smooth the slope values
# Lot of zeros (which doesn't really make sense for Mannings equation), so use a gaussian filter to smooth out
sigma = 2  # in samples; increase for more smoothing
ds["S"] = gaussian_filter1d(ds["S"].to_numpy(), sigma=2, mode="nearest")

# filter transect width to try to get rid of larger transects (keep to minimum)
ds['Length_m'] = minimum_filter(ds['Length_m'], size=10)


# Calculate depth from mannings equation
# Q = (A)(R^(2/3))(S^(1/2))/n
# Rearranged to solve for depth (and assuming a large rectangular channel, so A = depth*width, R = depth)

ds["dz"] = ds.apply(
    lambda r: manning_depth_wide_rectangular(
        Q=Q,
        b=r["Length_m"],
        n=r["n"] if "n" in r and pd.notna(r["n"]) else 0.030,
        S=r["S"] if "S" in r and pd.notna(r["S"]) else 0.0001,
    ),
    axis=1,
)



ds['Station_m'] = (ds['ID_E2W']-1) * 50  # 50m spacing between transects

# Create the NAVD88 timerseries but save out the downstream match point)
NAVD88_RealData = ds['NAVD88_m'].iloc[-1]
ds['NAVD88_m'] = ds['HydoElev_m'] - ds['dz']



# Smooth to get rid of sills with somoe heavy smoothing
dz_temp = gaussian_filter1d(ds['dz'], sigma=20, mode="nearest")

# Decrease depth to account for simulation errors
dz_temp = dz_temp + BC_Correction


# Calculate a fix to match the downstream boundary
# Interpolates so that over num_transects we go from the location we trust the depth (ds['NAVD88_m'].iloc[-1])
# to the full mannings calculated depth.  
num_trans = 80
offset = np.full(ds['NAVD88_m'].shape,0,dtype=np.float64)
offset[-1] =  NAVD88_RealData -ds['HydoElev_m'].iloc[-1]+dz_temp[-1]
offset[-num_trans:-1] = np.linspace(0,offset[-1],num_trans-1)



ds['dz_interp'] = dz_temp - offset



ds['NAVD88_interp'] = ds['HydoElev_m'] -ds['dz_interp'] 



# ===============================================================================
# %% Plot
# ===============================================================================


fig, [ax1,ax2] = plt.subplots(2, 1, figsize=[8, 5])

ax1.plot(
     ds['Station_m'],
    ds['HydoElev_m'],
    "b",
    label="Water Surface Elev",
)

ax1.plot(
     ds['Station_m'],
    ds['HydoElev_m'] -ds['dz'] ,
    "k",
    label="Bottom Depth",
)

ax1.plot(
     ds['Station_m'],
    ds['NAVD88_interp'] ,
    "r",
    label="Bottom Depth Smooth",
)

ax1.grid()
ax1.set_ylabel('Bottom Elev (m)')
ax1.set_title('Bottom Elevations')
ax1.legend()
ax1.set_xticklabels([])

ax2.plot(
    ds['Station_m'],
    ds['dz'],
    "k",
    label="Mannings Depth",
)

ax2.plot(
    ds['Station_m'],
    ds['dz_interp'],
    "r",
    label="Mannings Depth Smooth + BC",
)

ax2.grid()
ax2.set_ylabel('Channel Depth')
ax2.set_title('Channel Depth from Mannings Equation')
ax2.set_xlabel('Distance along river (m)')

ax2.invert_yaxis()

ds.to_file(os.path.join(dir_out,'Stilli_Transects3.shp'), driver='ESRI Shapefile')

