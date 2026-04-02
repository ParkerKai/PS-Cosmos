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
dir_in = r'Y:\WFLOW'
dir_out = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\figures\CDF_Diff\WFLOW\Shapefiles'

# County
cnty = 'king'

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
        data_series = pd.Series(data[:,stat],  index= t)
        
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
# %% Load the data 
#===============================================================================    

# Load the data 
files = glob(os.path.join(dir_in,'20240801_discharges',cnty,'cdf_diff','*.nc'))
ds = xr.open_mfdataset(files, engine='h5netcdf', parallel=True)


# Load the geojson
df = gpd.read_file(os.path.join(dir_in,'20240801_discharges',cnty,f'{cnty.title()}_gauges_contour.geojson'))

# Add Lat Lon to xarray file 
ds['station'].values

lat = np.full(ds['station'].size,np.nan)
lon = np.full(ds['station'].size,np.nan)
stat_geom = []
for cnt,stat in enumerate(ds['station'].values):
    pull = df.query('oid == @stat')
    
    lat[cnt] = pull['geometry'].y.to_numpy()[0]
    lon[cnt] = pull['geometry'].x.to_numpy()[0]
    
ds['lat']  = xr.DataArray(
         data   = lat,    # enter data here
         dims   = ['station'],
         coords = {'station': ds['station']},
         attrs  = {
             'LongName':'Latitude',
             '_FillValue': -9999,
             'units'     : 'degrees',
             'projection': 'EPSG:4269'
             })

ds['lon']  = xr.DataArray(
         data   = lon,    # enter data here
         dims   = ['station'],
         coords = {'station': ds['station']},
         attrs  = {
             'LongName':'longitude',
             '_FillValue': -9999,
             'units'     : 'degrees',
             'projection': 'EPSG:4269'
             })


#===============================================================================
# %% Export
#===============================================================================

# pull out the data condensened to the mean of the ensemble
data = ds['Q']
t  = ds['time'].values

RIs = calc_RIs(data,t, RPs_want)

# Turn into a dataframe
d = {'Mean': np.mean(data, axis=0),
      'Max': np.max(data, axis=0),
      'Std': np.std(data, axis=0),
      'Q99': np.quantile(data,.99, axis=0),
      'Q95': np.quantile(data,.95, axis=0),
      'RI_1': RIs[0,:],
      'RP_10': RIs[1,:],
      'RP_25': RIs[2,:],
      'RP_50': RIs[3,:],
      'RP_84': RIs[4,:]}

# Convert to pandas dataframe
stats_out = pd.DataFrame(data=d)

# add geometry and turn into geopandas dataset
geometry = gpd.points_from_xy(x=ds['lon'],
                              y=ds['lat'],
                              crs = df.crs)

# Convert to a geopandas dataframe
out = gpd.GeoDataFrame(data =stats_out, geometry=geometry)


out.to_file(os.path.join(dir_out,f'WFLOW_Era5Stats_{cnty}.shp'))

    