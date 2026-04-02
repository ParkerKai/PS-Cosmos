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
import windrose
import sys
import matplotlib

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'Y:\PS_Cosmos\LUT\LUT_output'
dir_out = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\figures\CDF_Diff\Waves'

Mod_list = ['CMCC','CNRM','EcEarth','GFDL','HadGemHH','HadGemHM','HadGemHMsst']
Month_list = ['January','February','March','April','May','June','July','August',
              'September','October','November','December']

SLR ='000'

stat_want = 628

#===============================================================================
# %% Define some functions
#===============================================================================
sys.path.append(r'C:\Users\kaparker\Documents\GitHub\Python\Kai_Python\General_Functions')
from Kai_MatlabTools import read_mat,read_mat_ts,read_mat_geo
from Kai_MapTools import Get_StationOrder


def pull_extremes(data,var_extremes,var_other):
    # Data : Pandas Dataframe of data you are pulling extremes from
    #var_extremes: variable you are using to choose extremes with. string
    # var_other: other variables you want added to the output dataframe
    #            these are other variables that happen at the same time as the extreme.
    #            for example: Hs may be your var_extreme and you want the concurrent wave period and direction (var_other)
    #            var_other is a list of strings
    #
    # output has dimentions of [num_years, num_vars, num_stations]
    
    import pyextremes
    
    # strip the data of nans
    data = data.dropna()
    
    # Load TWL values into memeory as a pandas series
    data_series = data[var_extremes]
    
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
        ind = data[var].index.isin(extremes.index)
        df[var] =  data[var][ind]
    
    # convert to numpy so we can stack stations
    out = df.to_numpy()
    
    
    return out 


#===============================================================================
# %% Figure out station re-order
#===============================================================================  
#Fix for stations being mixed up for the CMIP6 data vs. the ERA5 data

# LatLon_ds = pd.DataFrame(np.transpose(np.vstack((ds['Lat'].values[0,:], ds['lon'].values[0,:]))),columns = ['Lat','Lon'])
file_in = os.path.join(dir_in, 'LUT_output_KingPierce_ERA5', 'LUT_output_KingPierce_ERA5_10mIso_1941_2023.mat')
LatLon_era5 = read_mat_geo(file_in,'lat_10mIso','lon_10mIso')


file_in = os.path.join(dir_in, 'LUT_output_KingPierce_CMIP6_historical', 'LUT_wave_timeSeries_CMCC_his.mat')
#hs = read_mat(file_in,'Hs',stat_want)
#dm = read_mat(file_in,'Dm',stat_want)
#data_h = pd.concat([hs,dm],axis=1)
LatLon_cmip = read_mat_geo(file_in,'lat','lon')

Ind_reorder = Get_StationOrder(LatLon_era5, LatLon_cmip, dist_min = 0.3)

stat_want_new = Ind_reorder[stat_want]

#===============================================================================
# %% Plots
#===============================================================================    
# Get station order for Cmip

out_h = []
out_f = []
Lat   = []
Lon   = []
for Mod_cnt,Mod in enumerate(Mod_list):
    print(f'Processing Model {Mod}, SLR {SLR}')
    
    file_in = os.path.join(dir_in, 'LUT_output_KingPierce_CMIP6_historical', f'LUT_wave_timeSeries_{Mod}_his.mat')
    #hs = read_mat(file_in,'Hs',stat_want)
    #dm = read_mat(file_in,'Dm',stat_want)
    #data_h = pd.concat([hs,dm],axis=1)

    data_h = pd.concat([read_mat_ts(file_in,'Hs',stat_want_new),read_mat_ts(file_in,'Dm',stat_want_new)],axis=1)
    
    file_in = os.path.join(dir_in, 'LUT_output_KingPierce_CMIP6_future',Mod, f'LUT_wave_timeSeries_{Mod}_fut_SLR{SLR}.mat')
    data_f = pd.concat([read_mat_ts(file_in,'Hs',stat_want_new),read_mat_ts(file_in,'Dm',stat_want_new)],axis=1)
    
    #Pull extremes
    extremes_h = pull_extremes(data_h,'Hs',['Dm'])
    extremes_f = pull_extremes(data_f,'Hs',['Dm'])

    out_h.append(extremes_h)
    out_f.append(extremes_f)
    
    Lat.append(read_mat(file_in,'lat',stat_want_new))
    Lon.append(read_mat(file_in,'lon',stat_want_new))

    
# Check that the lat Lat's are teh same for each model
if ((np.unique( np.array(Lat)).size != 1) | (np.unique( np.array(Lon )).size != 1)):
    raise Exception('Lat/Lon inconsistent across array')
else:
    Lat = np.unique( np.array(Lat))
    Lon = np.unique( np.array(Lon))


# Concatenate to a pandas datafame 
extr_h = pd.DataFrame(data = np.concatenate(out_h, axis=0),columns=['Hs','Dm'])
extr_f = pd.DataFrame(data = np.concatenate(out_f, axis=0),columns=['Hs','Dm'])

#===============================================================================
# %% Plot
#===============================================================================

fig ,[ax1,ax2] = matplotlib.pyplot.subplots(1, 2,subplot_kw={'projection': 'windrose'})
fig.set_size_inches(8, 6)

# pull out the data condensened to the mean of the ensemble

max_bin =np.round( np.max([extr_h['Hs'].max(),extr_f['Hs'].max()]),decimals=2)

#ax1 = windrose.WindroseAxes.from_ax(ax = ax1)
ax1.bar(extr_h['Dm'], extr_h['Hs'], normed=True, opening=0.8, edgecolor="white", bins=np.linspace(0, max_bin, 8))
ax1.set_legend()
#fig.savefig(os.path.join(dir_out,f'WaveRose_all_Stat{stat_want}.tiff'), dpi=600)
ax1.set_title('Historic')

#ax = windrose.WindroseAxes.from_ax()
ax2.bar(extr_f['Dm'], extr_f['Hs'], normed=True, opening=0.8, edgecolor="white", bins=np.linspace(0, max_bin, 8))
ax2.set_title('Future')

fig.savefig(os.path.join(dir_out,f'WaveRose_extremes_HisFut_Stat{stat_want}.tiff'), dpi=600)


