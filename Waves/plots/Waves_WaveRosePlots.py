# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:12:15 2024

This script plots wave roses for the ERA5 period 

 
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
import windrose

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'X:\PS_Cosmos\LUT\LUT_output'
dir_out = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\figures\CDF_Diff\Waves'

SLR ='000'

stat_want = 337

# Return Periods wanted 
RPs_want = [1,10,25,50,84]

#===============================================================================
# %% Define some functions
#===============================================================================


def pull_extremes(data,var_extremes,var_other):
    # Data : Xarray of data you are pulling extremes from
    #var_extremes: variable you are using to choose extremes with. string
    # var_other: other variables you want added to the output dataframe
    #            these are other variables that happen at the same time as the extreme.
    #            for example: Hs may be your var_extreme and you want the concurrent wave period and direction (var_other)
    #            var_other is a list of strings
    #
    # output has dimentions of [num_years, num_vars, num_stations]
    
    import pyextremes
    
    if 'station' in data.coords:
        num_stat = ds['station'].shape[0]
        
        for stat in range(num_stat):
            # Load TWL values into memeory as a pandas series
            data_series = pd.Series(data[var_extremes].isel(station=stat),  index= data['time'])
            
            # FInd extremes
            extremes = pyextremes.get_extremes(ts=data_series,
                method="BM",
                extremes_type="high",
                block_size="365.2425D",
                errors="raise",
                min_last_block=0.5)
            
            # Intialize with first run-through
            if stat == 0:
                out = np.full([extremes.shape[0],len(var_other)+1,num_stat],np.nan)
            
            # Convert to a dataframe and add other variables that are temporally concurrent
            df = extremes.to_frame()
            df.rename(columns={"extreme values": var_extremes},inplace=True)
            for var in var_other:
                df[var] =  data[var].sel(time = extremes.index, station=stat)
            
            # convert to numpy so we can stack stations
            out[:,:,stat] = df.to_numpy()
    
    else:
        # Load TWL values into memeory as a pandas series
        data_series = pd.Series(data[var_extremes],  index= data['time'])
        
        # FInd extremes
        extremes = pyextremes.get_extremes(ts=data_series,
            method="BM",
            extremes_type="high",
            block_size="365.2425D",
            errors="raise",
            min_last_block=0.5)
        
        # Intialize with first run-through
        
        # Convert to a dataframe and add other variables that are temporally concurrent
        df = extremes.to_frame()
        df.rename(columns={"extreme values": var_extremes},inplace=True)
        for var in var_other:
            df[var] =  data[var].sel(time = extremes.index)
        
        # convert to numpy so we can stack stations
        out = df.to_numpy()
    
    
    return out 
    

#===============================================================================
# %% Plots
#===============================================================================    
print(f'Processing: {SLR} Station: {stat_want}')

# Load the data 

files = glob(os.path.join(dir_in,'LUt_KingPierce_CMIP6_Diff',SLR,'*.nc'))
ds = xr.open_mfdataset(files, engine='h5netcdf', parallel=True)

ds = ds.isel(station=stat_want)
ds = ds.drop_vars('station')
extremes = pull_extremes(ds,'Hs',['Dm','Tp'])

#===============================================================================
# %% Plot
#===============================================================================

# pull out the data condensened to the mean of the ensemble


ax = windrose.WindroseAxes.from_ax()
ax.bar(ds['Dm'], ds['Hs'], normed=True, opening=0.8, edgecolor="white")
ax.set_legend()
fig = ax.get_figure()
fig.savefig(os.path.join(dir_out,f'WaveRose_all_Stat{stat_want}.tiff'), dpi=600)


ax = windrose.WindroseAxes.from_ax()
ax.bar(extremes[:,1], extremes[:,0], normed=True, opening=0.8, edgecolor="white")
ax.set_legend()
fig = ax.get_figure()
fig.savefig(os.path.join(dir_out,f'WaveRose_extremes_Stat{stat_want}.tiff'), dpi=600)


