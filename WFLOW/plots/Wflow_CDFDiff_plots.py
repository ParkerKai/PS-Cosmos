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
    # Version abs :absolute difference
    # Version rel :relative  difference
    # Version per :relative percent difference 
    



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
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'Y:\WFLOW\20240801_discharges'
dir_out = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\figures\CDF_Diff\WFLOW'

# Station
Stat = 21

# County
cnty = 'pierce'

# Models 
Mod_list = ['CNRM','EcEarth','GFDL','HadGemHH','HadGemHM','HadGemHMsst']
Month_list = ['January','February','March','April','May','June','July','August',
              'September','October','November','December']
#Mod = 'EcEarth'


#===============================================================================
# %% Load the data 
#===============================================================================

files = glob(os.path.join(dir_in,cnty,'cdf_diff','*.nc'))
ds = xr.open_mfdataset(files, engine='h5netcdf', parallel=True)

ds = ds.isel(station=Stat,drop=True)

#===============================================================================
# %% Plots
#===============================================================================    

# fig, ax = matplotlib.pyplot.subplots(1, 1)
# fig.set_size_inches(8, 6)

# l1 = ax.scatter(ds['Q'],ds['hs_quants'],
#                 s = 10, marker = '.' ,color = 'k', )

# ax.grid()
# ax.set_title('Wave Height CDF')
# ax.set_xlabel('Hs (m)')
# ax.set_ylabel('CDF')
# ax.set_ylim([0,1])


fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 6)

l1 = ax.plot(ds['time'], ds['Q'],color = 'k',
              label = 'WaterLevel')

for ii in range(ds['cmip6'].size):
    
    ax.plot(ds['time'], ds['Q']+ds['cmip_diff'].isel(cmip6=ii))


ax.set_xlim(pd.Timestamp('1941-10-01'),pd.Timestamp('1942-01-01'))
ax.set_ylim(000,500)
ax.grid()
ax.set_title('Streamflow')
ax.set_ylabel('Q (m3/s)')
ax.set_xlabel('Date')

ax.legend(['ERA5','CMCC','CNRM','GFDL','HadGemHH','HadGemHM','HadGemHMsst'])
fig.savefig(os.path.join(dir_out,'Q_Ts_AllMods.tiff'), dpi=600)


#######################################################################

fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 6)

l1 = ax.plot(ds['time'], ds['Q'],color = 'k',
              label = 'WaveHeight')

l1 = ax.fill_between(ds['time'],ds['Q']+ds['cmip_diff'].max(dim='cmip6'),
                     ds['Q']+ds['cmip_diff'].min(dim='cmip6'),
                     alpha = 0.5)

l2 = ax.plot(ds['time'],ds['Q']+ds['cmip_diff'].mean(dim='cmip6'),
                     color = 'b')

l3 = ax.plot(ds['time'], ds['Q'],color = 'k',
              label = 'WaveHeight')

# Make the shaded region
#ax.fill(x,y,color='b',alpha = 0.5)

ax.set_xlim(pd.Timestamp('1941-10-01'),pd.Timestamp('1942-01-01'))
ax.set_ylim(000,500)
ax.grid()
ax.set_title('Streamflow')
ax.set_ylabel('Q (m/s)')
ax.set_xlabel('Date')
ax.legend({'ERA5','Cmip6 Model Range','CMIP6 Ave.'})
fig.savefig(os.path.join(dir_out,'Q_Ts_Range.tiff'), dpi=600)

################################################################################


fig = matplotlib.pyplot.subplots(3, 4)
fig[0].set_size_inches(8, 6)
ax = fig[0].get_axes()


diff_save = np.full([1000,12,len(Mod_list)],np.nan)

for Mod_cnt,Mod in enumerate(Mod_list):
    print(f'Processing Model {Mod}')
    
    
    for month in np.arange(1, 13, 1, dtype=int):        
          # Load the CMIP6 historic data
        with open(os.path.join(dir_in,cnty, f'cmip6_{Mod}_historic_bc',
                                'CDFmonthly_{0:02d}.pkl'.format(month)), 'rb') as f: 
            cdf_cmipH = pickle.load(f)
    
            
          # Load the CMIP6 future data
        with open(os.path.join(dir_in,cnty,  f'cmip6_{Mod}_future_bc',
                                'CDFmonthly_{0:02d}.pkl'.format(month)), 'rb') as f: 
            cdf_cmipF = pickle.load(f)
        
    
        # Subset to station of interest
        cdf_cmipH = cdf_cmipH.loc[cdf_cmipH['stat'] == Stat]
        cdf_cmipF = cdf_cmipF.loc[cdf_cmipF['stat'] == Stat]
                    
        diff, quants = calc_diff(cdf_cmipH,cdf_cmipF,'per')
        
        diff_save[:,month-1,Mod_cnt] = diff
        
        l1 = ax[month-1].plot(quants,diff)
        l2 = ax[month-1].plot([0,1],[0,0],'k--')
        
        ax[month-1].set_xlim([0,1])
        
        ax[month-1].set_title(Month_list[month-1])
        
        if month <= 8:
            ax[month-1].set_xticklabels([])
        
        if month == 5:
            ax[month-1].set_ylabel('Percent Change in Q by Quantile')
        
        # if month == 1:
        #     ax[month-1].set_ylim([-100,100])
        # elif month == 2:
        #     ax[month-1].set_ylim([-50,50])
        # elif month == 3:
        #     ax[month-1].set_ylim([-50,50])            
        # elif month == 4:
        #     ax[month-1].set_ylim([-25,25])
        # elif month == 5:
        #     ax[month-1].set_ylim([-20,20])
        # elif month == 6:
        #     ax[month-1].set_ylim([-40,40])
        # elif month == 7:
        #     ax[month-1].set_ylim([-30,30]) 
        # elif month == 8:
        #     ax[month-1].set_ylim([-30,30])
        # elif month == 9:
        #     ax[month-1].set_ylim([-30,30])
        # elif month == 10:
        #     ax[month-1].set_ylim([-40,40])
        # elif month == 11:
        #     ax[month-1].set_ylim([-50,50])     
        # elif month == 12:
        #     ax[month-1].set_ylim([-50,50])
            
            # if month == 1:
            #ax[month-1].legend(['CMCC','CNRM','GFDL','HadGemHH','HadGemHM','HadGemHMsst'])
        
        ax[month-1].grid()


diff_mean = np.mean(diff_save, axis=2)
for month in np.arange(1, 13, 1, dtype=int):        
    l3 = ax[month-1].plot(quants,diff_mean[:,month-1],'k')

fig[0].savefig(os.path.join(dir_out,'Q_Monthly_Models.tiff'), dpi=600)


################################################################################

fig = matplotlib.pyplot.subplots(3, 4)
fig[0].set_size_inches(8, 6)
ax = fig[0].get_axes()


for month in np.arange(1, 13, 1, dtype=int):        

    l1 = ax[month-1].fill_between(quants,np.max(diff_save[:,month-1,:],axis=1),
                         np.min(diff_save[:,month-1,:],axis=1),
                         alpha = 0.5,color = 'k')
    
    l2 = ax[month-1].plot(quants,np.mean(diff_save[:,month-1,:],axis=1),
                         color = 'k')
    
    l3 = ax[month-1].plot([0,1],[0,0],'k--')

    ax[month-1].set_xlim([0,1])
    
    ax[month-1].set_title(Month_list[month-1])
    
    if month <= 8:
        ax[month-1].set_xticklabels([])
        
    if month == 5:
        ax[month-1].set_ylabel('Percent Change in Q by Quantile')
    ax[month-1].grid()

fig[0].savefig(os.path.join(dir_out,'Q_Monthly_Range.tiff'), dpi=600)


