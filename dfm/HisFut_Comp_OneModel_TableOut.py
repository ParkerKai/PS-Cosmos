# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:22:20 2024

This script plots the waterlevels for CMIP6 historic-future runs 


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
from glob import glob
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import scipy
import pyextremes

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'Y:\PS_Cosmos\02_models\DFM_Regional'
dir_out = r'Y:\PS_Cosmos\GIS\DFM\DFM_CmipDiff_byModel'

# Model to process
Mod = 'HadGEM_GC31_HM_highResSST'

#===============================================================================
# %% Define some functions
#===============================================================================

def CalcStats(data):
    # Data: Xarray dataset 
    # dim: Condense along this dimention. So statistics are for this dimention (e.ge. mean of time if dimention is time
    
    
    num_stat = data['waterlevel'].shape[1]
    time = data['time'].values
    
    stats = np.full((num_stat,10),np.nan)
    for cnt in range(num_stat):
        print(f'Processing Station: {cnt}')
        # Load values into memeory
        data_pull  = data['waterlevel'].isel(station=cnt).values/ data['waterlevel'].attrs['ScaleFactor']    # convert to m
        
        # Calculate statistics on data 
        stats[cnt,0] = np.nanmean(data_pull)          # Mean
        stats[cnt,1] = np.nanmax(data_pull)           # Max
        stats[cnt,2] = np.nanstd(data_pull)           # Standard Deviation
        stats[cnt,3] = np.nanquantile(data_pull,.99)  # 99th Quantile
        stats[cnt,4] = np.nanquantile(data_pull,.95)  # 95th Quantile
        
        
        # Load TWL values into memeory as a pandas series
        data_series = pd.Series(data_pull,  index= time)
        
        # FInd extremes
        extremes = pyextremes.get_extremes(ts=data_series,
            method="BM",
            extremes_type="high",
            block_size="365.2425D",
            errors="raise",
            min_last_block=0.5)
        
        return_periods = pyextremes.get_return_periods(
            ts=data_series,
            extremes=extremes,
            extremes_method="BM",
            extremes_type="high",
            block_size="365.2425D",
            return_period_size="365.2425D",
            plotting_position="weibull",
        )
        
        return_periods.sort_values("return period", ascending=True,inplace=True)
        
        
        interp_rp = scipy.interpolate.interp1d(return_periods['return period'].to_numpy(),
                                             return_periods['extreme values'].to_numpy(),
                                             copy=False,
                                             assume_sorted=True,
                                             fill_value=(return_periods['extreme values'].min(),np.nan),
                                             bounds_error=False) 
        
        stats[cnt,5] = interp_rp(np.array(1))
        stats[cnt,6] = interp_rp(np.array(5))
        stats[cnt,7] = interp_rp(np.array(10))
        stats[cnt,8] = interp_rp(np.array(15))
        stats[cnt,9] = interp_rp(np.array(30))
                
    
    # Turn into a dataframe
    d = {'Mean': stats[:,0],
         'Max': stats[:,1],
         'Std': stats[:,2],
         'Q99': stats[:,3],
         'Q95': stats[:,4],
         'RI_1': stats[:,5],
         'RP_5': stats[:,6],
         'RP_10': stats[:,7],
         'RP_15': stats[:,8],
         'RP_30': stats[:,9]}
    
    # Convert to pandas dataframe
    stats_out = pd.DataFrame(data=d)
    
    # add geometry and turn into geopandas dataset
    geometry = gpd.points_from_xy(x=data['lon'].isel(time=1, drop=True),
                                  y=data['lat'].isel(time=1, drop=True),
                                  crs = 'EPSG:4326')
    
    out = gpd.GeoDataFrame(data =stats_out, geometry=geometry)
    
    return out


def CalcDiff(dataH,dataF):
    
    # Check geometry
    check = (dataH['geometry'] ==  dataF['geometry'])
    if (sum(check) != len(check)):
        print('Geometry doesnt match between historic and future datasets')
    
    # Convert to numpy for easy diff calculating
    H = pd.DataFrame(dataH.drop(columns='geometry')).to_numpy()
    F = pd.DataFrame(dataF.drop(columns='geometry')).to_numpy()
    
    # Calculate the difference (future - Historic)
    diff = (F-H)
    
    # Pull out the columns of the stats array
    columns = dataH.columns
    
    # Re-build the pandas dataframe
    df_diff = pd.DataFrame(diff, columns=columns[0:-1])
    
    # Convert to a geopandas dataframe
    out = gpd.GeoDataFrame(data =df_diff, geometry=dataH['geometry'])
    
    return out
    

#===============================================================================
# %% Load the data 
#===============================================================================

# CMIP6 His
print('Processing Historic Period')
files = glob(os.path.join(dir_in,'CMIP6_Results',Mod,'historic','*.nc'))
ds_cmipH = xr.open_mfdataset(files, engine='netcdf4', parallel=True, mask_and_scale=False)

# Subset the historic run to have equal timeperiods
ds_cmipH = ds_cmipH.sel(time=slice("1980-01-01", "2014-01-01"))


stats_cmipH = CalcStats(ds_cmipH)

# CMIP6 Fut
print('Processing Future 000 Period')
files = glob(os.path.join(dir_in,'CMIP6_Results',Mod,'future','000','*.nc'))
ds_cmipF000 = xr.open_mfdataset(files, engine='netcdf4', parallel=True, mask_and_scale=False)

if Mod == 'EC-Earth_HR':
    ind_bad = (ds_cmipF000['time'] >= pd.Timestamp('2016-12-25')) & (ds_cmipF000['time'] <= pd.Timestamp('2017-01-01'))
    ds_cmipF000['waterlevel'][ind_bad] = ds_cmipF000['waterlevel'].attrs['_FillValue']

stats_cmipF000 = CalcStats(ds_cmipF000)

print('Processing Future 050 Period')
files = glob(os.path.join(dir_in,'CMIP6_Results',Mod,'future','050','*.nc'))
ds_cmipF050 = xr.open_mfdataset(files, engine='netcdf4', parallel=True, mask_and_scale=False)

if Mod == 'EC-Earth_HR':
    ind_bad = (ds_cmipF050['time'] >= pd.Timestamp('2016-12-25')) & (ds_cmipF050['time'] <= pd.Timestamp('2017-01-01'))
    ds_cmipF050['waterlevel'][ind_bad] = ds_cmipF050['waterlevel'].attrs['_FillValue']

stats_cmipF050 = CalcStats(ds_cmipF050)    

print('Processing Future 100 Period')
files = glob(os.path.join(dir_in,'CMIP6_Results',Mod,'future','100','*.nc'))
ds_cmipF100 = xr.open_mfdataset(files, engine='netcdf4', parallel=True, mask_and_scale=False)

if Mod == 'EC-Earth_HR':
    ind_bad = (ds_cmipF100['time'] >= pd.Timestamp('2016-12-25')) & (ds_cmipF100['time'] <= pd.Timestamp('2017-01-01'))
    ds_cmipF100['waterlevel'][ind_bad] = ds_cmipF100['waterlevel'].attrs['_FillValue']
    
stats_cmipF100 = CalcStats(ds_cmipF100)

print('Processing Future 200 Period')
files = glob(os.path.join(dir_in,'CMIP6_Results',Mod,'future','200','*.nc'))
ds_cmipF200 = xr.open_mfdataset(files, engine='netcdf4', parallel=True, mask_and_scale=False)

if Mod == 'EC-Earth_HR':
    ind_bad = (ds_cmipF200['time'] >= pd.Timestamp('2016-12-25')) & (ds_cmipF200['time'] <= pd.Timestamp('2017-01-01'))
    ds_cmipF200['waterlevel'][ind_bad] = ds_cmipF200['waterlevel'].attrs['_FillValue']

stats_cmipF200 = CalcStats(ds_cmipF200)

print('Processing Future 300 Period')
files = glob(os.path.join(dir_in,'CMIP6_Results',Mod,'future','300','*.nc'))
ds_cmipF300 = xr.open_mfdataset(files, engine='netcdf4', parallel=True, mask_and_scale=False)

if Mod == 'EC-Earth_HR':
    ind_bad = (ds_cmipF300['time'] >= pd.Timestamp('2016-12-25')) & (ds_cmipF300['time'] <= pd.Timestamp('2017-01-01'))
    ds_cmipF300['waterlevel'][ind_bad] = ds_cmipF300['waterlevel'].attrs['_FillValue']
    
stats_cmipF300 = CalcStats(ds_cmipF300)

# calculate difference from historic to future period.  
diff_000 = CalcDiff(stats_cmipH,stats_cmipF000)
diff_050 = CalcDiff(stats_cmipH,stats_cmipF050)
diff_100 = CalcDiff(stats_cmipH,stats_cmipF100)
diff_200 = CalcDiff(stats_cmipH,stats_cmipF200)
diff_300 = CalcDiff(stats_cmipH,stats_cmipF300)

#===============================================================================
# %% Export
#===============================================================================
diff_000.to_file(os.path.join(dir_out,'Dfm_diff000_{}.shp'.format(Mod)))
diff_050.to_file(os.path.join(dir_out,'Dfm_diff050_{}.shp'.format(Mod)))
diff_100.to_file(os.path.join(dir_out,'Dfm_diff100_{}.shp'.format(Mod)))
diff_200.to_file(os.path.join(dir_out,'Dfm_diff200_{}.shp'.format(Mod)))
diff_300.to_file(os.path.join(dir_out,'Dfm_diff300_{}.shp'.format(Mod)))

