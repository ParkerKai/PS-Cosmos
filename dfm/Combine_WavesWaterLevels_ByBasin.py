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

#===============================================================================
# %% Import Modules
#===============================================================================
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

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in_wl = r'D:\DFM_Regional'
dir_in_waves = r'D:\LUT_timeSeries'
dir_in_gis   = r'Y:\PS_Cosmos\GIS\Shapefiles\general'


dir_out = r'D:\Combined_DFM\ERA5'

county_list = ['Kitsap','Clallam','IslandCounty','Snohomish','Skagit','Jefferson',
          'King','Pierce','Thurston','Whatcom']    # 

file_match = r'Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries\dfm_wave_Index2.mat'



#===============================================================================
# %% Define some functions
#===============================================================================
sys.path.append(r'C:\Users\kaparker\Documents\GitHub\Python\Kai_Python\General_Functions')
from Kai_MatlabTools import matlab2datetime
from Kai_XarrayTools import Get_Station_index

def toTimestamp(d):

    return d.astype('int64') // 10**9  # Divide by 10^9 to get seconds


def LoadWaveLUTmats(file_in):
    
    # Load the .mat file using h5py
    with h5py.File(file_in, "r") as f:
        group = f['LUTout']
        hs  = group['Hs'][()]
        dm  = group['Dm'][()]
        tp  = group['Tp'][()]
        t     = group['t'][:,0][()]
        lat   = np.squeeze(group['lat'][()])
        lon   = np.squeeze(group['lon'][()])
        depth =  np.squeeze(group['depth'][()])
        stat  = np.arange(0,len(lon),dtype='int32')
        dfm_id=  np.squeeze(group['DFMid'][()])
        
        # Convert from matlab to pandas datetimeIndex.  
        # timeseries is in hours so round to hours to clean up conversion error.
        t_dt  = matlab2datetime(t,'h') 
    
    # Turn into an Xarray dataset 
    ds = xr.Dataset({
        'Hs': xr.DataArray(
                    data   = hs,    # enter data here
                    dims   = ['time','station'],
                    coords = {'time': t_dt,
                              'station':stat},
                    attrs  = {
                        '_FillValue': -9999,
                        'units'     : 'meters'
                        }
                    ),
        'Dm': xr.DataArray(
                    data   = dm,    # enter data here
                    dims   = ['time','station'],
                    coords = {'time': t_dt,
                              'station':stat},
                    attrs  = {
                        '_FillValue': -9999,
                        'units'     : 'degrees'
                        }
                    ),
        'Tp': xr.DataArray(
                    data   = tp,    # enter data here
                    dims   = ['time','station'],
                    coords = {'time': t_dt,
                              'station':stat},
                    attrs  = {
                        '_FillValue': -9999,
                        'units'     : 'seconds'
                        }
                    ),
        'Lat': xr.DataArray(
                    data   = lat,   # enter data here
                    dims   = ['station'],
                    coords = {'station': stat},
                    attrs  = {
                        '_FillValue': -9999,
                        'units'     : 'Degree'
                        }
                    ),
        'Lon': xr.DataArray(
                    data   = lon,   # enter data here
                    dims   = ['station'],
                    coords = {'station': stat},
                    attrs  = {
                        '_FillValue': -9999,
                        'units'     : 'Degree'
                        }
                    ),
        'depth': xr.DataArray(
                    data   = depth,   # enter data here
                    dims   = ['station'],
                    coords = {'station': stat},
                    attrs  = {
                        '_FillValue': -9999,
                        'units'     : 'm'
                        }
                    ),
        'DFMid': xr.DataArray(
                    data   = dfm_id,   # enter data here
                    dims   = ['station'],
                    coords = {'station': stat},
                    attrs  = {
                        '_FillValue': -9999,
                        'units'     : 'ID'
                        }
                    ),
                },
            attrs = {'DataSource': 'Y:\PS_Cosmos\PS_Cosmos\09_wave_lut_predictions\LUT_output\LUT_output_KingPierce_ERA5',
                     'ProducedBy': 'Anita Englestad and Kai Parker',
                     'General': 'Tm,Tp and Dm are NaN for Hs < 0.05m. Output is for lat_10mIso/lon_10mIso locations. lat_ncPoint and lon_ncPoint are the original nc points to which the closest gridpoints on the 10m isobath were found'}
        )

        
        
    return ds

def LoadWaveLUTmats_stat(file_in,stat_geometry):
    
    # Load the .mat file using h5py
    with h5py.File(file_in, "r") as f:
        group = f['LUTout']
        lat   = np.squeeze(group['lat'][()])
        lon   = np.squeeze(group['lon'][()])
    
        Index_wave = gpd.GeoDataFrame(geometry = gpd.points_from_xy(lon, lat),crs = 'EPSG:6318')
        Ind_pull = Index_wave.sindex.nearest(stat_geometry)
        
        # multiple repeat stations so just chose the first.  [1 is because we want the indices (not ball tree),0 is first element]
        Ind_pull = Ind_pull[1,0]
        
        stat  = Ind_pull

        
        hs  = group['Hs'][:,Ind_pull].squeeze()
        dm  = group['Dm'][:,Ind_pull].squeeze()
        tp  = group['Tp'][:,Ind_pull].squeeze()
        t     = group['t'][()][()].squeeze()
        depth = group['depth'][:,Ind_pull][()].squeeze()
        lat  = group['lat'][:,Ind_pull][()].squeeze()
        lon  = group['lon'][:,Ind_pull][()].squeeze()
        
        # Convert from matlab to pandas datetimeIndex.  
        # timeseries is in hours so round to hours to clean up conversion error.
        t_dt  = matlab2datetime(t,'h')  
       
       
        # Turn into an Xarray dataset 
        ds = xr.Dataset({
            'Hs': xr.DataArray(
                        data   = hs,    # enter data here
                        dims   = ['time'],
                        coords = {'time': t_dt},
                        attrs  = {
                            '_FillValue': -9999,
                            'units'     : 'meters'
                            }
                        ),
            'Dm': xr.DataArray(
                        data   = dm,    # enter data here
                        dims   = ['time'],
                        coords = {'time': t_dt},
                        attrs  = {
                            '_FillValue': -9999,
                            'units'     : 'degrees'
                            }
                        ),
            'Tp': xr.DataArray(
                        data   = tp,    # enter data here
                        dims   = ['time'],
                        coords = {'time': t_dt},
                        attrs  = {
                            '_FillValue': -9999,
                            'units'     : 'seconds'
                            }
                        ),
            'Lat': xr.DataArray(
                        data   = lat,   # enter data here
                        attrs  = {
                            '_FillValue': -9999,
                            'units'     : 'Degree'
                            }
                        ),
            'Lon': xr.DataArray(
                        data   = lon,   # enter data here
                        attrs  = {
                            '_FillValue': -9999,
                            'units'     : 'Degree'
                            }
                        ),
            'depth': xr.DataArray(
                        data   = depth,   # enter data here
                        attrs  = {
                            '_FillValue': -9999,
                            'units'     : 'm'
                            }
                        ),
            'stat_wave': xr.DataArray(
                        data   = stat,   # enter data here
                        attrs  = {
                            '_FillValue': -9999,
                            'units'     : 'none',
                            'file'      : file_in
                            }
                        )},
                attrs = {'DataSource': 'Y:\PS_Cosmos\PS_Cosmos\09_wave_lut_predictions\LUT_output\LUT_output_KingPierce_ERA5',
                         'ProducedBy': 'Anita Englestad and Kai Parker',
                         'General': 'Tm,Tp and Dm are NaN for Hs < 0.05m. Output is for lat_10mIso/lon_10mIso locations. lat_ncPoint and lon_ncPoint are the original nc points to which the closest gridpoints on the 10m isobath were found'}
            )                   

        
        
    return ds

#===============================================================================
# %% Read in the match file
#===============================================================================

# Index_DFM = pd.read_csv(r'Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries\dfm_waveIndex_DFM.csv')
# Index_LUT = pd.read_csv(r'Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries\dfm_waveIndex_LUT.csv')

# Index_LUT = gpd.GeoDataFrame(Index_LUT,geometry = gpd.points_from_xy(Index_LUT['lon'], Index_LUT['lat']),crs = 'EPSG:6318')

# Index_DFM.sindex
# Index_LUT.sindex


#===============================================================================
# %% Read in the 
# Load the data DFM Model Data
#===============================================================================

files = glob(os.path.join(dir_in_wl,'ERA5','ERA5_000','DFM_wl*.nc'))
ds_full = xr.open_mfdataset(files, engine='h5netcdf', parallel=True,
                       chunks={"time": -1, "station": 1})

files = glob(os.path.join(dir_in_wl,'ERA5_Tidal','000','DFM_wl*.nc'))
ds_tidal = xr.open_mfdataset(files, engine='h5netcdf', parallel=True,
                       chunks={"time": -1, "station": 1})



#Subset to station and from 1995 on (since thats when gauge records start)
ds_full['waterlevel'] = ds_full['waterlevel']/10000


# Add tide from tide only runs
ds_tidal['waterlevel'] = ds_tidal['waterlevel']/10000
ds_full = ds_full.assign(tide=ds_tidal['waterlevel'])

ds_tidal = ds_tidal.interp(time=ds_full['time'])

ds_full = ds_full.assign(ntr = (['time','station'], ds_full['waterlevel'].data - ds_tidal['waterlevel'].data))





#===============================================================================
# %% Read in the Basin File and find aggregating indexes
#===============================================================================

basins = gpd.read_file(os.path.join(dir_in_gis,'SalishSea_Basins.shp'))
basins = basins.to_crs(crs = 'EPSG:6318')

DFM_pnts = gpd.GeoDataFrame(pd.DataFrame({'Station':ds_full['station']}),
                            geometry = gpd.points_from_xy(ds_full['lon'].isel(time=0),
                                                                    ds_full['lat'].isel(time=0)),
                            crs = 'EPSG:6318')


# For each basin find the DFM_Pnts within
DFM_pnts = gpd.sjoin(basins,DFM_pnts,how='right')
DFM_pnts = DFM_pnts.rename(columns={'index_left': 'BasinID'})
DFM_pnts['BasinID'] = DFM_pnts['BasinID'].fillna(-999)
DFM_pnts['BasinID'] = DFM_pnts['BasinID'].values.astype('int32')


ds_full['BasinID']=DFM_pnts['BasinID'].values



asdf

ds_full = ds_full.groupby('BasinID').mean()

#===============================================================================
# %% Read in the Wave Model Data
#===============================================================================


# file_in = os.path.join(dir_in_waves, f'LUT_output_{county}_ERA5', f'LUT_output_{county}_ERA5_10mIsobath.mat')

# ds_wave = LoadWaveLUTmats(file_in)




#===============================================================================
# %% Read in the Wave Model Data at each DFM node, combine with WL, and Save
#===============================================================================
stations = ds_full['station'].values


# For each Water Level point read in the waves
for ii,station_sel in enumerate(stations):
    
    ii = ii+1183
    
    print(f'Processing {station_sel}, Number {ii}')
    
    file_out = os.path.join(dir_out,f'CombinedTWL_{ii:04d}.nc')
    
    if os.path.exists(file_out):
        print(f'File exists: {file_out}')  
        
    else:
              
        # FInd index for station_id 
        ind4Index = Get_Station_index(ds_full,station_sel)
    
    
        ds_wl = ds_full.sel(station = station_sel)
        
    
        ds_wl['lat'] = xr.DataArray(
                    data   = ds_wl['lat'].values[0],    # enter data here
     
                    attrs  = ds_wl['lat'].attrs
                    )
        
        ds_wl['lon'] = xr.DataArray(
                    data   = ds_wl['lon'].values[0],    # enter data here
     
                    attrs  = ds_wl['lon'].attrs
                    )
        
        ds_wl['bedlevel'] = xr.DataArray(
                    data   = ds_wl['bedlevel'].values[0],    # enter data here
     
                    attrs  = ds_wl['bedlevel'].attrs
                    )
    
        # Wave index for nearest points
        Wave_index = Index_LUT.iloc[ind4Index]    
        
        # Wave County file to load
        county = str(Wave_index['county']).rstrip()
        file_in = os.path.join(dir_in_waves, f'LUT_output_{county}_ERA5', f'LUT_output_{county}_ERA5_10mIsobath.mat')
        ds_wave= LoadWaveLUTmats_stat(file_in,Index_DFM.iloc[ind4Index].geometry)
        
        # Interpolate wave to new time vector
        ds_wave = ds_wave.interp(time = ds_wl['time'].values)
        
        # Rename so everything makes sense when jammed together
        ds_wave = ds_wave.rename({'Lat':'lat_wave','Lon':'lon_wave','depth':'depth_wave'})
        ds_wl   = ds_wl.rename({'lon':'lon_wl','lat':'lat_wl'})
        
        #Combine
        ds  = xr.merge([ds_wl , ds_wave])
        
        # Export
        
        ds.to_netcdf(file_out)
    



