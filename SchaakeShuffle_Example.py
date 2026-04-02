# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:07:08 2024

This script tries to explain the SHaake Shuffle strategy of creating novel timeseries.


@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"

# Import Modules
import os
import xarray as xr
import numpy as np
import pandas as pd
#import datetime
import matplotlib
#import netcdf4

#===============================================================================
# %% User Defined inputs
#===============================================================================

# Directory where DFM .fou Data is stored
folder_in = r'C:\Users\kaparker\Documents\Weekly_Updates\04_17_2024\figures\Shuffle\data'

# Output directory
dir_out = r"C:\Users\kaparker\Documents\Weekly_Updates\04_17_2024\figures\Shuffle"

# Model
Mod = 'ECEarth'


His_chunk = [pd.Timestamp('2000-11-20'),pd.Timestamp('2000-11-30')]
Fut_chunk = [pd.Timestamp('2020-11-20'),pd.Timestamp('2020-11-30')]


#===============================================================================
# %% Define functions
#===============================================================================

def subset_Wave(data,stat,t_lim):
    # Get variables of interest
    data = data[['hs','th1p','fp']]
    data['Tp'] = 1/data['fp']
    data = data.drop_vars('fp')

    # Subset to station of interest (using index)
    data = data.isel(station = stat)
    
    # Subset to time of interest(using slice)
    data = data.sel(time=slice(t_lim[0],t_lim[1]))
    
    return data

def schaake_Shuffle(data_target,data_model):
    import sys

    # A couple checks
    # Datasets need to be the same length
    if (data_target['time'].size != data_model['time'].size):
        raise Exception('Target and Model arent the same length')
        sys.exit(1) 
        
    # Datasets need the same variables
    if list(data_target.keys()) != list(data_model.keys()):
        raise Exception('Target and Model variables arent the same')
        sys.exit(1) 
    
    # Calculate ranks
    target_Ranks = data_target.rank('time').astype(int)
    # model_Ranks  = data_model.rank('time').astype(int)


    # Shuffle The model to match the target
    Fut_shuf = Fut.copy()
    
    # Go through each variable and shuffle
    for var in list(data_target.keys()):
        pull = Fut[var].sortby(Fut[var]).values
        Fut_shuf[var].data   = pull[target_Ranks[var].values-1]

    return Fut_shuf 
    
    

#===============================================================================
# %% Load some data
#===============================================================================
His = xr.open_dataset(os.path.join(folder_in,'WavePnts_{}_ep_10m_His.nc'.format(Mod)),engine='netcdf4')
Fut = xr.open_dataset(os.path.join(folder_in,'WavePnts_{}_ep_10m_Fut.nc'.format(Mod)),engine='netcdf4')

# Subset to a smaller dataset (for space)
His = subset_Wave(His,1,His_chunk)
Fut = subset_Wave(Fut,1,Fut_chunk)

# Calculate ranks
His_Ranks = His.rank('time').astype(int)
Fut_Ranks = Fut.rank('time').astype(int)


# Shuffle The future to match the historic
Fut_shuf = schaake_Shuffle(His,Fut)

#===============================================================================
# %% Plot the data
#===============================================================================

fig1, [ax1,ax2,ax3] = matplotlib.pyplot.subplots(3,1)
fig1.set_size_inches([8,6])
ax1.scatter(His['time'],His['hs'], c = His_Ranks['hs'], cmap = 'YlOrRd', marker = '.')
ax2.scatter(His['time'],His['Tp'], c = His_Ranks['Tp'], cmap = 'YlOrRd', marker = '.')
ax3.scatter(His['time'],His['th1p'], c = His_Ranks['th1p'], cmap = 'YlOrRd', marker = '.')

ax1.grid(); ax2.grid();  ax3.grid() 
ax1.set_xticklabels([]); ax2.set_xticklabels([]);

ax1.set_ylim(1,3.2); ax2.set_ylim(5,20); ax3.set_ylim(80,240);
ax1.set_ylabel('Hs'); ax2.set_ylabel('Tp'); ax3.set_ylabel('MWD')


fig2, [ax1,ax2,ax3] = matplotlib.pyplot.subplots(3,1)
fig2.set_size_inches([8,6])
ax1.scatter(Fut['time'],Fut['hs'], c = Fut_Ranks['hs'],cmap = 'YlOrRd', marker = '.')
ax2.scatter(Fut['time'],Fut['Tp'], c = Fut_Ranks['Tp'],cmap = 'YlOrRd', marker = '.')
ax3.scatter(Fut['time'],Fut['th1p'], c = Fut_Ranks['th1p'],cmap = 'YlOrRd', marker = '.')

ax1.grid(); ax2.grid();  ax3.grid() 
ax1.set_xticklabels([]); ax2.set_xticklabels([]);

ax1.set_ylim(1,3.2); ax2.set_ylim(5,20); ax3.set_ylim(80,240);
ax1.set_ylabel('Hs'); ax2.set_ylabel('Tp'); ax3.set_ylabel('MWD')



fig3, [ax1,ax2,ax3] = matplotlib.pyplot.subplots(3,1)
fig3.set_size_inches([8,6])
ax1.scatter(Fut_shuf['time'],Fut_shuf['hs'], c = Fut_Ranks['hs'],cmap = 'YlOrRd', marker = '.')
ax2.scatter(Fut_shuf['time'],Fut_shuf['Tp'], c = Fut_Ranks['Tp'],cmap = 'YlOrRd', marker = '.')
ax3.scatter(Fut_shuf['time'],Fut_shuf['th1p'], c = Fut_Ranks['th1p'],cmap = 'YlOrRd', marker = '.')

ax1.grid(); ax2.grid();  ax3.grid() 
ax1.set_xticklabels([]); ax2.set_xticklabels([]);


ax1.set_ylim(1,3.2); ax2.set_ylim(5,20); ax3.set_ylim(80,240);
ax1.set_ylabel('Hs'); ax2.set_ylabel('Tp'); ax3.set_ylabel('MWD')


fig1.savefig(os.path.join(dir_out,'TimeSeries_Target.png'),  dpi=800,
            bbox_inches='tight')        
fig2.savefig(os.path.join(dir_out,'TimeSeries_Model.png'),  dpi=800,
            bbox_inches='tight') 
fig3.savefig(os.path.join(dir_out,'TimeSeries_ModelShuffled.png'),  dpi=800,
            bbox_inches='tight') 