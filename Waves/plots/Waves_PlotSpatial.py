# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:52:38 2024

This script Plots the spatial variablity in CMIP6 Diff
 
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
import matplotlib
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import scipy.stats as stats
import pyextremes
import scipy.optimize as optimize
import scipy

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'Y:\PS_Cosmos\GIS\Waves'
dir_out_fig = r'Y:\PS_Cosmos\Figures\Waves'

# Metric to plot
metric = 'RP_50'   

# 'CMIP6_historic' 'CMIP6_future' 'ERA5'
Period = 'ERA5'

# Model  (if cmip6)  CMCC CNRM EcEarth GFDL HadGemHH HadGemHM HadGemHMsst
Mod = 'GFDL'

#===============================================================================
# %% Define some functions
#===============================================================================


#===============================================================================
# %% Load the  data
#===============================================================================

# Load state shapefiles.
cnty = gpd.read_file(r'Y:\PS_Cosmos\GIS\general\Washington_Counties_with_Natural_Shoreline___washsh_area.shp')

# Load in the landmass file
lm = gpd.read_file(r"Y:\PS_Cosmos\GIS\general\PoliticalBoundaries_Shapefile\NA_PoliticalDivisions\data\bound_p\boundaries_p_2021_v3.shp")

if (Period == 'ERA5'):
    file_in = os.path.join(dir_in,Period,f'WaveHs_{Period}.shp')
        
elif (Period == 'CMIP6_historical') or (Period == 'CMIP6_future'):
    file_in = os.path.join(dir_in,Period,f'WaveHs_{Period}_{Mod}.shp')


waves = gpd.read_file(file_in)


#===============================================================================
# %% Load the  data
#===============================================================================
lm = lm.to_crs(crs=waves.crs)


ax1 = lm.plot(figsize=(10, 10), color='0.8', edgecolor='black', alpha=0.4)
s1 = waves.plot(ax=ax1,column=metric, marker='.', markersize=10,
                    legend=True, cmap='OrRd', legend_kwds={"label": "Hs (m)"},
                    label=f'{metric} Hs',
                    vmin=0,vmax=5)

ax1.set_xlim([-124.75,-122])
ax1.set_ylim([47,49.5])
ax1.grid()
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latittude')
ax1.set_title(f'{Period} Modelled {metric} Hs')
ax1.legend()

fig =ax1.get_figure()


if (Period == 'ERA5'):
    file_out = os.path.join(dir_out_fig,f'{Period}_Map_{metric}.tiff')
        
elif (Period == 'CMIP6_historical') or (Period == 'CMIP6_future'):
    file_out = os.path.join(dir_out_fig,f'{Period}_Map_{Mod}_{metric}.tiff')


fig.savefig(file_out, dpi=300)


