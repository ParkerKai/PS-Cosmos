# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:45:11 2024

This script plots differences to wave CDFS 

 
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
import scipy 
import geopandas as gpd

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries'
dir_out = r'Y:\PS_Cosmos\GIS\Waves\Mean_CmipDiff'

SLR_list =['000','025','050','100','150','200','300']
#SLR_list = ['025']

# Return Periods wanted 
RPs_want = [1,10,25,50,84]

#===============================================================================
# %% Define some functions
#===============================================================================


def calc_RIs(data,time, RPs):
    import pyextremes
    
    RIs = np.full([len(RPs),data.shape[1]],np.nan,dtype='float64')
    for stat in range(data.shape[1]):
        # Load TWL values into memeory as a pandas series
        data_series = pd.Series(data_mean[:,stat],  index= t)
        
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
        RIs[:,stat] = interp_rp(RPs)
        
    
    
    return RIs 
    

#===============================================================================
# %% Plots
#===============================================================================    
for SLR in SLR_list:
    print(f'Processing: {SLR}')

    # Load the data 

    files = glob(os.path.join(dir_in,'LUt_KingPierce_CMIP6_Diff',SLR,'*.nc'))
    ds = xr.open_mfdataset(files, engine='h5netcdf', parallel=True)

    #===============================================================================
    # %% Export
    #===============================================================================
    
    # pull out the data condensened to the mean of the ensemble
    data_mean = ds['cmip_diff'].mean(dim='cmip6').values
    t  = ds['time'].values
    
    RIs = calc_RIs(data_mean,t, RPs_want)
    
    
    # Turn into a dataframe
    d = {'Mean': np.mean(data_mean, axis=0),
          'Max': np.max(data_mean, axis=0),
          'Std': np.std(data_mean, axis=0),
          'Q99': np.quantile(data_mean,.99, axis=0),
          'Q95': np.quantile(data_mean,.95, axis=0),
          'RI_1': RIs[0,:],
          'RP_10': RIs[1,:],
          'RP_25': RIs[2,:],
          'RP_50': RIs[3,:],
          'RP_84': RIs[4,:]}
    
    
    # Convert to pandas dataframe
    stats_out = pd.DataFrame(data=d)
    
    # add geometry and turn into geopandas dataset
    geometry = gpd.points_from_xy(x=ds['lon'].isel(time=0),
                                  y=ds['lat'].isel(time=0),
                                  crs = 'EPSG:4326')
    
    # Convert to a geopandas dataframe
    out = gpd.GeoDataFrame(data =stats_out, geometry=geometry)
    
    
    out.to_file(os.path.join(dir_out,f'DFM_Cmip6DiffStats_{SLR}.shp'))
    
    