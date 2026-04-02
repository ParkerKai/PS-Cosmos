# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:52:38 2024

This script Plots the monthly cdf corrections
 
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


#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'Y:\PS_Cosmos\GIS\Waves\DFM_CmipDiff_byModel'
dir_out = r'Y:\PS_Cosmos\Figures\CDF_Diff\Waves'

# Model to process
Mod_list = ['CMCC','CNRM','EcEarth','GFDL','HadGemHH','HadGemHM','HadGemHMsst']

#SLR_list =['000','025','050','100','150','200','300']
SLR_list =['000']

# Metric to plot
metric = 'Q99'   


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


# REad in a single file to get dimentions 
file_in = os.path.join(dir_in,f'Wave_diff{SLR_list[0]}_{Mod_list[0]}.shp')
pull = gpd.read_file(file_in)

data = np.full([pull.shape[0],pull.shape[1]-2,len(Mod_list)],np.nan)
for slr in SLR_list:
    for cnt,Mod in enumerate(Mod_list):
        print(f'Processing: {Mod}')
        file_in = os.path.join(dir_in,f'Wave_diff{slr}_{Mod}.shp')
        
        pull = gpd.read_file(file_in)
        data[:,:,cnt] = pull.drop(columns=['geometry','county']).to_numpy()
        



mean_data = np.nanmean(data,axis=2)
mean_data = pd.DataFrame(mean_data, 
    columns=pull.drop(columns= ['geometry','county']).columns)

plot_data = gpd.GeoDataFrame(
    mean_data, 
    geometry=pull['geometry'],
    crs=pull.crs)  # Set the coordinate reference system)

std_data = np.nanstd(data,axis=2)
std_data = pd.DataFrame(std_data, 
    columns=pull.drop(columns=['geometry','county']).columns)

plot_data2 = gpd.GeoDataFrame(
    std_data, 
    geometry=pull['geometry'],
    crs=pull.crs)  # Set the coordinate reference system)


# find index for metric of interest
columns = pull.drop(columns=['geometry','county']).columns
ind_want = np.ravel(np.argwhere(columns=='RP_30'))

# One sample t-test if mean isn't zero
p_vals=np.full([data.shape[0]],np.nan)
sign  =np.full([data.shape[0]],False)
for cnt in range(data.shape[0]):
    # Sample data
    sample_pull = np.ravel(data[cnt,ind_want,:])
    
    # Population mean to test against
    population_mean = 0
    
    # Perform the one-sample t-test
    
    _, p_vals[cnt] = stats.ttest_1samp(sample_pull, population_mean,nan_policy='omit')
    
    if p_vals[cnt] <  0.01:
        sign[cnt] = True
        
plot_data_sig = plot_data.iloc[sign,:]

#===============================================================================
# %% Load the  data
#===============================================================================
lm = lm.to_crs(crs=plot_data_sig.crs)


ax1 = lm.plot(figsize=(10, 10), color='0.8', edgecolor='black', alpha=0.4)
s1 = plot_data_sig.plot(ax=ax1,column=metric, markersize=10, color='k',label='Statistically Significant')
s2 = plot_data.plot(ax=ax1,column=metric, marker='.', markersize=10,
                    legend=True, cmap='OrRd', legend_kwds={"label": "Ensemble Mean Change (m)"},
                    label='Mean Change')
ax1.set_xlim([-124.75,-122])
ax1.set_ylim([47,49.25])
ax1.grid()
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latittude')
ax1.set_title(f'CMIP6 Difference in {metric} water levels')
ax1.legend()

fig =ax1.get_figure()
fig.savefig(os.path.join(dir_out,f'MeanDiff_Map_{metric}.tiff'), dpi=600)


ax1 = lm.plot(figsize=(10, 10), color='0.8', edgecolor='black', alpha=0.4)
plot_data2.plot(ax=ax1,column=metric, marker='.', markersize=10, legend=True, cmap='Purples')
ax1.set_xlim([-124,-122])
ax1.set_ylim([47,49.25])
ax1.grid()
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latittude')
ax1.set_title(f'Std of CMIP6 Ensemble {metric} water levels')



#fig[0].savefig(os.path.join(dir_out,f'Cmip6Diff_{Stat}.tiff'), dpi=600)

