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
import matplotlib 
import pickle 
import scipy 

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'Y:\WFLOW'
dir_out = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\figures\CDF_Diff\Waves'

SLR_list =['000','025','050','100','150','200','300']
#SLR = '025'

# Station
Stat =337

Mod_list = ['CMCC','CNRM','EcEarth','GFDL','HadGemHH','HadGemHM','HadGemHMsst']
Month_list = ['January','February','March','April','May','June','July','August',
              'September','October','November','December']

#===============================================================================
# %% Define some functions
#===============================================================================


def interpAtQuant(cdf_vals,cdf_quant,data_quant):
    
    # Determine CDF based on the pre-calculated ERA5 cdf 
    interp_data = scipy.interpolate.interp1d(cdf_quant,
                                             cdf_vals, 
                                             fill_value=(cdf_vals.min(),cdf_vals.max()),
                                          copy=False,
                                          assume_sorted=True,
                                          bounds_error=False)
    vals = interp_data(data_quant)

    return vals 


def calc_diff(cdf_H,cdf_F,version):
    import sys
    # Calc CDF correction.
    quants = np.arange(0,1,.001)

                        
    # PUll data for the station and unwrap pandas dataframe to numpy
    cdf_H_stat_cdf = cdf_H['cdf'].to_numpy()
    cdf_H_stat_val = cdf_H['values'].to_numpy()
    cdf_F_stat_cdf = cdf_F['cdf'].to_numpy()
    cdf_F_stat_val = cdf_F['values'].to_numpy()
    
    F = interpAtQuant(cdf_F_stat_val,cdf_F_stat_cdf,quants) 
    H = interpAtQuant(cdf_H_stat_val,cdf_H_stat_cdf,quants)
    
    if version == 'abs':
        diff = F - H

    elif version == 'rel':
        diff = (F-H)/H
        
    elif version == 'per':
        diff = ((F-H)/H)*100
    else:
        sys.exit('Incorrection Version Selected')
    
    diff[~np.isfinite(diff)] = np.nan

    return diff, quants


#===============================================================================
# %% Load the data 
#===============================================================================



diff_mean_save = np.full([1000,12,len(SLR_list)],np.nan)
for SLR_cnt,SLR in enumerate(SLR_list):
    files = glob(os.path.join(dir_in,'LUt_KingPierce_CMIP6_Diff',SLR,'*.nc'))
    ds = xr.open_mfdataset(files, engine='h5netcdf', parallel=True)
    
    ds = ds.isel(station=Stat,drop=True)

    # calculate the monthly quantiles / differences
    diff_save = np.full([1000,12,ds['cmip6'].shape[0]],np.nan)
    
    Mod_list = ds['cmip6'].values
    for Mod_cnt,Mod in enumerate(Mod_list):
        print(f'Processing Model {Mod}, SLR {SLR}')
        
        for month in np.arange(1, 13, 1, dtype=int):       
            
             # Load the CMIP6 historic data
            with open(os.path.join(dir_in,'LUT_output_KingPierce_CMIP6_historical',Mod,
                                    f'CDFmonthly_{month:02d}_{Mod}.pkl'), 'rb') as f: 
                cdf_cmipH = pickle.load(f)
        
                
              # Load the CMIP6 future data
            with open(os.path.join(dir_in,'LUT_output_KingPierce_CMIP6_future',Mod,
                                    f'CDFmonthly_{month:02d}_{Mod}_SLR{SLR}.pkl'), 'rb') as f: 
                cdf_cmipF = pickle.load(f)
            
            # Subset to station of interest
            cdf_cmipH = cdf_cmipH.loc[cdf_cmipH['stat'] == Stat]
            cdf_cmipF = cdf_cmipF.loc[cdf_cmipF['stat'] == Stat]
                        
            diff, quants = calc_diff(cdf_cmipH,cdf_cmipF,'abs')
            
            diff_save[:,month-1,Mod_cnt] = diff

    diff_mean_save[:,:,SLR_cnt] = np.mean(diff_save, axis=2)


#===============================================================================
# %% Plots
#===============================================================================    

################################################################################

    
fig1 = matplotlib.pyplot.subplots(3, 4)
fig1[0].set_size_inches(8, 6)
ax = fig1[0].get_axes()

cmap = matplotlib.colormaps['cividis'].resampled(len(SLR_list)).colors

for SLR_cnt,SLR in enumerate(SLR_list):
    for month in np.arange(1, 13, 1, dtype=int):        
        l1 = ax[month-1].plot(quants,diff_mean_save[:,month-1,SLR_cnt],
                              color = cmap[SLR_cnt])
        

################################################################################
for month in np.arange(1, 13, 1, dtype=int):  

    l3 = ax[month-1].plot([0,1],[0,0],'k--')
      
    ax[month-1].set_xlim([0,1])
    ax[month-1].grid()
    
    ax[month-1].set_title(Month_list[month-1])
    
    if month <= 8:
        ax[month-1].set_xticklabels([])
        
    if month == 5:
        ax[month-1].set_ylabel('Change in Hs by Quantile')
        
        ax[month-1].legend(SLR_list)
    

fig1[0].savefig(os.path.join(dir_out,'Wave_Monthly_BySLR.tiff'), dpi=600)
