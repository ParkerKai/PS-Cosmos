# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:50:56 2024

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
import numpy as np
import xarray as xr
import pickle
import scipy 
import matplotlib 
import pandas as pd 

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'C:\Users\kai\Documents\KaiRuns\DFM'
dir_in_era5 = r'D:\Kai\DFM\ERA5'
dir_out = r'D:\Kai\DFM\ERA5_CMIP6'

# Model to process
Mod = 'CMCC-CM2-VHR4'  # ,'CNRM-CM6-1-HR','GFDL'

# Station
Stat = 562

#SLR_list =['000','025','050','100','150','200','300']
SLR = '000'


#===============================================================================
# %% Define some functions
#===============================================================================



#===============================================================================
# %% Load the ERA5 data and calc quantiles
#===============================================================================


#===============================================================================
# %% Calculate correction for ERA5 based on CMIP6 projections
#===============================================================================    

# fig, [ax1,ax2] = matplotlib.pyplot.subplots(1, 2)
# fig.set_size_inches(8, 8)
# for cnt,Mod in enumerate(Mod_list):
    
#     print(f'Processing CMIP6 Difference for {Mod}')

    
#     for month in np.arange(1, 13, 1, dtype=int):
#         print(f'Processing Month {month:02d}')
        
#          # Load the CMIP6 historic data
#         with open(os.path.join(dir_in,'CMIP6',Mod,'historic','Results_Combined',
#                                'CDFmonthly_{0:02d}.pkl'.format(month)), 'rb') as f: 
#             cdf_cmipH = pickle.load(f)

            
#          # Load the CMIP6 future data
#         with open(os.path.join(dir_in,'CMIP6',Mod,'future','Results_Combined',SLR,
#                                'CDFmonthly_{0:02d}.pkl'.format(month)), 'rb') as f: 
#             cdf_cmipF = pickle.load(f)
        

#         # Subset to station of interest
#         cdf_cmipH = cdf_cmipH.loc[cdf_cmipH['stat'] == Stat]
#         cdf_cmipF = cdf_cmipF.loc[cdf_cmipF['stat'] == Stat]
                    
        
#         l1 = ax1.plot(cdf_cmipH['values'], cdf_cmipH['cdf'])
#         l2 = ax2.plot(cdf_cmipF['values'], cdf_cmipF['cdf'])
        
        
# ax1.legend()
# ax1.legend({'Jan','Feb','Mar','Apr','May','June','July','Aug','Sept','Oct','Nov.','Dec.'})
# ax1.grid(); ax2.grid()
# ax1.set_ylim([0,1]); ax2.set_ylim([0,1])
# #ax1.set_xlim([0,0.5]);ax2.set_xlim([0,0.5])
# ax1.set_title('Historic Water Level CDF by Month '); ax2.set_title('Future Water Level CDF by Month ')
# ax1.set_ylabel('CDF val')
# ax1.set_xlabel('Water Levels')



fig = matplotlib.pyplot.subplots(3, 4)
fig[0].set_size_inches(8, 8)
ax = fig[0].get_axes()


for month in np.arange(1, 13, 1, dtype=int):
    print(f'Processing Month {month:02d}')
    
      # Load the CMIP6 historic data
    with open(os.path.join(dir_in,'CMIP6',Mod,'historic','Results_Combined',
                            'CDFmonthly_{0:02d}.pkl'.format(month)), 'rb') as f: 
        cdf_cmipH = pickle.load(f)

        
      # Load the CMIP6 future data
    with open(os.path.join(dir_in,'CMIP6',Mod,'future','Results_Combined',SLR,
                            'CDFmonthly_{0:02d}.pkl'.format(month)), 'rb') as f: 
        cdf_cmipF = pickle.load(f)
    

    # Subset to station of interest
    cdf_cmipH = cdf_cmipH.loc[cdf_cmipH['stat'] == Stat]
    cdf_cmipF = cdf_cmipF.loc[cdf_cmipF['stat'] == Stat]
                
    
    l1 = ax[month-1].plot(cdf_cmipH['values'], cdf_cmipH['cdf'],color='k')
    l2 = ax[month-1].plot(cdf_cmipF['values'], cdf_cmipF['cdf'],color='r')
    
    ax[month-1].grid()
    
    
    ax[month-1].set_ylim([0,1])
    #ax[month-1].set_xlim([0,0.5])



