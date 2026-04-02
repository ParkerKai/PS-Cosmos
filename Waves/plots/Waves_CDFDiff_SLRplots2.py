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
import numpy as np
import pandas as pd
import matplotlib 
import scipy 
import sys 
#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'X:\PS_Cosmos\LUT\LUT_output'
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

sys.path.append(r'C:\Users\kaparker\Documents\GitHub\Python\Kai_Python\General_Functions')
from Kai_MatlabTools import read_mat


def calc_cdf(data, quants):
    out = scipy.stats.mstats.mquantiles(data, prob=quants, alphap=0.4, betap=0.4)
    
    return out 


def calc_CdfDiff(data_H,data_F,version):
    import sys
    # Calc CDF correction.
    quants = np.arange(0,1,.001)

    H = calc_cdf(data_H,quants)
    F = calc_cdf(data_F,quants)
    
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

diff_save = np.full([1000,len(Mod_list),len(SLR_list)],np.nan)

for SLR_cnt,SLR in enumerate(SLR_list):
    for Mod_cnt,Mod in enumerate(Mod_list):
        print(f'Processing Model {Mod}, SLR {SLR}')
        
        file_in = os.path.join(dir_in, 'LUT_output_KingPierce_CMIP6_historical', f'LUT_wave_timeSeries_{Mod}_his.mat')
        data_h = read_mat(file_in,'Hs',Stat)
        
        file_in = os.path.join(dir_in, 'LUT_output_KingPierce_CMIP6_future',Mod, f'LUT_wave_timeSeries_{Mod}_fut_SLR{SLR}.mat')
        data_f = read_mat(file_in,'Hs',Stat)
        
        
        #ds['cmip_diff'].isel(cmip6=Mod_cnt)
        diff, quants = calc_CdfDiff(data_h,data_f,'abs')
        
        diff_save[:,Mod_cnt,SLR_cnt] = diff
    
        
#===============================================================================
# %% Plots
#===============================================================================    

################################################################################
    
fig1 = matplotlib.pyplot.subplots(1, 1)
fig1[0].set_size_inches(8, 6)
ax = fig1[0].get_axes()
ax = ax[0]

for SLR_cnt,SLR in enumerate(SLR_list):
    data_SLR = diff_save[:,:,SLR_cnt]
    l1 = ax.plot(quants,np.mean(data_SLR,axis=1))  
    
l2 = ax.plot([0,1],[0,0],'k--')
ax.grid()
ax.set_xlim([0,1])
ax.set_title('Quantile Change by SLR')
ax.set_ylabel('Change in Hs (m)')
ax.set_xlabel('Quantile')
ax.legend(SLR_list)

fig1[0].savefig(os.path.join(dir_out,'Wave_Change_BySLR.tiff'), dpi=600)






