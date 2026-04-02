# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:56:45 2025

This script compares DFM outputs vs Tidegauge data

@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"

# ===============================================================================
# %% Import Modules
# ===============================================================================
import sys
import os
import scipy
import xarray as xr
import numpy as np
import pandas as pd

# from sklearn.metrics import root_mean_squared_error,r2_score,mean_absolute_error
# from sklearn.linear_model import LinearRegression
# import matplotlib
from glob import glob
import geopandas as gpd

# import pandas as pd
# from scipy.interpolate import interp1d
# import h5py
import h5py
import shapely


# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the DFM data resides
# dir_in = r'D:\DFM'
dir_in_wl = r"D:\DFM_Regional"
dir_in_waves = r"D:\LUT_timeSeries"

dir_out = r"D:\Combined_DFM\ERA5"


# ===============================================================================
# %% Define some functions
# ===============================================================================
sys.path.append(
    r"C:\Users\kaparker\Documents\GitHub\Python\Kai_Python\General_Functions"
)
from Kai_MatlabTools import matlab2datetime
from Kai_XarrayTools import Get_Station_index
from Kai_GeoTools import distance_ll


def toTimestamp(d):
    return d.astype("int64") // 10**9  # Divide by 10^9 to get seconds


# def LoadWaveLUTmats(file_in):

#     # Load the .mat file using h5py
#     with h5py.File(file_in, "r") as f:
#         group = f['LUTout']
#         hs  = group['Hs'][()]
#         dm  = group['Dm'][()]
#         tp  = group['Tp'][()]
#         t     = group['t'][:,0][()]
#         lat   = np.squeeze(group['lat'][()])
#         lon   = np.squeeze(group['lon'][()])
#         depth =  np.squeeze(group['depth'][()])
#         stat  = np.arange(0,len(lon),dtype='int32')
#         dfm_id=  np.squeeze(group['DFMid'][()])

#         # Convert from matlab to pandas datetimeIndex.
#         # timeseries is in hours so round to hours to clean up conversion error.
#         t_dt  = matlab2datetime(t,'h')

#     # Turn into an Xarray dataset
#     ds = xr.Dataset({
#         'Hs': xr.DataArray(
#                     data   = hs,    # enter data here
#                     dims   = ['time','station'],
#                     coords = {'time': t_dt,
#                               'station':stat},
#                     attrs  = {
#                         '_FillValue': -9999,
#                         'units'     : 'meters'
#                         }
#                     ),
#         'Dm': xr.DataArray(
#                     data   = dm,    # enter data here
#                     dims   = ['time','station'],
#                     coords = {'time': t_dt,
#                               'station':stat},
#                     attrs  = {
#                         '_FillValue': -9999,
#                         'units'     : 'degrees'
#                         }
#                     ),
#         'Tp': xr.DataArray(
#                     data   = tp,    # enter data here
#                     dims   = ['time','station'],
#                     coords = {'time': t_dt,
#                               'station':stat},
#                     attrs  = {
#                         '_FillValue': -9999,
#                         'units'     : 'seconds'
#                         }
#                     ),
#         'Lat': xr.DataArray(
#                     data   = lat,   # enter data here
#                     dims   = ['station'],
#                     coords = {'station': stat},
#                     attrs  = {
#                         '_FillValue': -9999,
#                         'units'     : 'Degree'
#                         }
#                     ),
#         'Lon': xr.DataArray(
#                     data   = lon,   # enter data here
#                     dims   = ['station'],
#                     coords = {'station': stat},
#                     attrs  = {
#                         '_FillValue': -9999,
#                         'units'     : 'Degree'
#                         }
#                     ),
#         'depth': xr.DataArray(
#                     data   = depth,   # enter data here
#                     dims   = ['station'],
#                     coords = {'station': stat},
#                     attrs  = {
#                         '_FillValue': -9999,
#                         'units'     : 'm'
#                         }
#                     ),
#         'DFMid': xr.DataArray(
#                     data   = dfm_id,   # enter data here
#                     dims   = ['station'],
#                     coords = {'station': stat},
#                     attrs  = {
#                         '_FillValue': -9999,
#                         'units'     : 'ID'
#                         }
#                     ),
#                 },
#             attrs = {'DataSource': 'Y:\PS_Cosmos\PS_Cosmos\09_wave_lut_predictions\LUT_output\LUT_output_KingPierce_ERA5',
#                      'ProducedBy': 'Anita Englestad and Kai Parker',
#                      'General': 'Tm,Tp and Dm are NaN for Hs < 0.05m. Output is for lat_10mIso/lon_10mIso locations. lat_ncPoint and lon_ncPoint are the original nc points to which the closest gridpoints on the 10m isobath were found'}
#         )


#     return ds


def LoadWaveLUTmats_stat(file_in, stat_geometry):
    # Load the .mat file using h5py
    with h5py.File(file_in, "r") as f:
        group = f["LUTout"]
        lat = np.squeeze(group["lat"][()])
        lon = np.squeeze(group["lon"][()])

        Index_wave = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(lon, lat), crs="EPSG:6318"
        )
        Ind_pull = Index_wave.sindex.nearest(stat_geometry)

        # multiple repeat stations so just chose the first.  [1 is because we want the indices (not ball tree),0 is first element]
        stat = Ind_pull[1, 0]

        # Pull the geometry sothat we can calculate the distance.
        lat = group["lat"][:, stat][()].squeeze()
        lon = group["lon"][:, stat][()].squeeze()

        # Define points

        # Create GeoSeries
        points = gpd.GeoDataFrame(
            {"geometry": [stat_geometry, shapely.geometry.Point(lon, lat)]},
            crs="EPSG:4326",
        )
        points = points.to_crs("EPSG:32610")

        # Calculate distance
        points_df = (
            points.shift()
        )  # We shift the dataframe by 1 to align pnt1 with pnt2
        dist = points_df.distance(points)[1]  # In meters

        if dist < 5000:
            hs = group["Hs"][:, stat].squeeze()
            dm = group["Dm"][:, stat].squeeze()
            tp = group["Tp"][:, stat].squeeze()
            t = group["t"][()][()].squeeze()
            depth = group["depth"][:, stat][()].squeeze()

        else:
            print(f"Station: lat {lat}, lon {lon} has no nearby waves. Dist {dist}")

            temp = group["Hs"][:, stat].squeeze()

            hs = np.full(temp.shape, np.nan)
            dm = np.full(temp.shape, np.nan)
            tp = np.full(temp.shape, np.nan)
            t = group["t"][()][()].squeeze()
            depth = group["depth"][:, stat][()].squeeze()

        # Convert from matlab to pandas datetimeIndex.
        # timeseries is in hours so round to hours to clean up conversion error.
        t_dt = matlab2datetime(t, "h")

        # Turn into an Xarray dataset
        ds = xr.Dataset(
            {
                "Hs": xr.DataArray(
                    data=hs,  # enter data here
                    dims=["time"],
                    coords={"time": t_dt},
                    attrs={"_FillValue": -9999, "units": "meters"},
                ),
                "Dm": xr.DataArray(
                    data=dm,  # enter data here
                    dims=["time"],
                    coords={"time": t_dt},
                    attrs={"_FillValue": -9999, "units": "degrees"},
                ),
                "Tp": xr.DataArray(
                    data=tp,  # enter data here
                    dims=["time"],
                    coords={"time": t_dt},
                    attrs={"_FillValue": -9999, "units": "seconds"},
                ),
                "Lat": xr.DataArray(
                    data=lat,  # enter data here
                    attrs={"_FillValue": -9999, "units": "Degree"},
                ),
                "Lon": xr.DataArray(
                    data=lon,  # enter data here
                    attrs={"_FillValue": -9999, "units": "Degree"},
                ),
                "depth": xr.DataArray(
                    data=depth,  # enter data here
                    attrs={"_FillValue": -9999, "units": "m"},
                ),
                "stat_wave": xr.DataArray(
                    data=stat,  # enter data here
                    attrs={"_FillValue": -9999, "units": "none", "file": file_in},
                ),
            },
            attrs={
                "DataSource": "Y:\PS_Cosmos\PS_Cosmos\09_wave_lut_predictions\LUT_output\LUT_output_KingPierce_ERA5",
                "ProducedBy": "Anita Englestad and Kai Parker",
                "General": "Tm,Tp and Dm are NaN for Hs < 0.05m. Output is for lat_10mIso/lon_10mIso locations. lat_ncPoint and lon_ncPoint are the original nc points to which the closest gridpoints on the 10m isobath were found",
            },
        )

    return ds


# ===============================================================================
# %% Read in the match file
# ===============================================================================

Index_DFM = pd.read_csv(
    r"Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries\dfm_waveIndex_DFM.csv"
)
Index_LUT = pd.read_csv(
    r"Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries\dfm_waveIndex_LUT.csv"
)

Index_DFM = gpd.GeoDataFrame(
    Index_DFM,
    geometry=gpd.points_from_xy(Index_DFM["lon"], Index_DFM["lat"]),
    crs="EPSG:6318",
)
Index_LUT = gpd.GeoDataFrame(
    Index_LUT,
    geometry=gpd.points_from_xy(Index_LUT["lon"], Index_LUT["lat"]),
    crs="EPSG:6318",
)

Index_DFM.sindex
Index_LUT.sindex

# Calculate the distance between wave and WL stations
dist = distance_ll(
    np.vstack((Index_LUT["lat"].values, Index_LUT["lon"].values)).transpose(),
    np.vstack((Index_LUT["DFMlat"].values, Index_LUT["DFMlon"].values)).transpose(),
)

Index_LUT["DistFromDFM"] = dist

# ===============================================================================
# %% Read in the
# Load the data DFM Model Data
# ===============================================================================

files = glob(os.path.join(dir_in_wl, "ERA5", "ERA5_000", "DFM_wl*.nc"))
ds_full = xr.open_mfdataset(
    files, engine="h5netcdf", parallel=True, chunks={"time": -1, "station": 1}
)

files = glob(os.path.join(dir_in_wl, "ERA5_Tidal", "000", "DFM_wl*.nc"))
ds_tidal = xr.open_mfdataset(
    files, engine="h5netcdf", parallel=True, chunks={"time": -1, "station": 1}
)


# Subset to station and from 1995 on (since thats when gauge records start)
ds_full["waterlevel"] = ds_full["waterlevel"] / 10000

# Add tide from tide only runs
ds_tidal["waterlevel"] = ds_tidal["waterlevel"] / 10000
ds_full = ds_full.assign(tide=ds_tidal["waterlevel"])

ds_tidal = ds_tidal.interp(time=ds_full["time"])

ds_full = ds_full.assign(
    ntr=(["time", "station"], ds_full["waterlevel"].data - ds_tidal["waterlevel"].data)
)

# ===============================================================================
# %% Read in the Wave Model Data at each DFM node, combine with WL, and Save
# ===============================================================================
stations = ds_full["station"].values

wl_stats = gpd.GeoDataFrame(
    pd.DataFrame(stations),
    geometry=gpd.points_from_xy(
        ds_full["lon"].isel(time=0).values, ds_full["lat"].isel(time=0).values
    ),
    crs="EPSG:4326",
)

# Figure out the index match between the Index_LUT table and DFM stations.
LUT_DFMstats = gpd.points_from_xy(
    Index_LUT["DFMlon"], Index_LUT["DFMlat"], crs="EPSG:4326"
)
ind_match = wl_stats.sindex.nearest(LUT_DFMstats)

# For each Water Level point read in the waves
for index, Wave_index in Index_LUT.iterrows():
    print(f"Processing, Number {index}")

    file_out = os.path.join(dir_out, f"CombinedTWL_{index:04d}.nc")

    if os.path.isfile(file_out):
        print(f"'{file_out}' is an existing file. Skipping")

    else:
        # FInd index for station_id
        ind4Index = ind_match[
            1, index
        ]  # 0th row is the index in Index_LUT, 1st is in LUT_DFMstats

        ds_wl = ds_full.isel(station=ind4Index)

        ds_wl["lat"] = xr.DataArray(
            data=ds_wl["lat"].values[0],  # enter data here
            attrs=ds_wl["lat"].attrs,
        )

        ds_wl["lon"] = xr.DataArray(
            data=ds_wl["lon"].values[0],  # enter data here
            attrs=ds_wl["lon"].attrs,
        )

        ds_wl["bedlevel"] = xr.DataArray(
            data=ds_wl["bedlevel"].values[0],  # enter data here
            attrs=ds_wl["bedlevel"].attrs,
        )

        # Wave County file to load
        county = str(Wave_index["county"]).rstrip()

        if county == "SanJuan":
            file_in = os.path.join(
                dir_in_waves,
                f"LUT_output_{county}_ERA5",
                f"LUT_output_{county}_ERA5_5mIsobath.mat",
            )

        else:
            file_in = os.path.join(
                dir_in_waves,
                f"LUT_output_{county}_ERA5",
                f"LUT_output_{county}_ERA5_10mIsobath.mat",
            )

        ds_wave = LoadWaveLUTmats_stat(file_in, Index_DFM.iloc[ind4Index].geometry)

        # Interpolate wave to new time vector
        ds_wave = ds_wave.interp(time=ds_wl["time"].values)

        # Rename so everything makes sense when jammed together
        ds_wave = ds_wave.rename(
            {"Lat": "lat_wave", "Lon": "lon_wave", "depth": "depth_wave"}
        )
        ds_wl = ds_wl.rename({"lon": "lon_wl", "lat": "lat_wl"})

        # Combine
        ds = xr.merge([ds_wl, ds_wave])

        # Export

        ds.to_netcdf(file_out)
