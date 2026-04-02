# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:00:12 2024

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
import numpy as np
import scipy 
import matplotlib 
import sys 
import pickle 
import xarray as xr

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'D:\DFM_Regional\CMIP6_Results'
dir_out = r'Y:\PS_Cosmos\Figures\CDF_Diff\DFM'

# Model to process
Mod_list = ['CNRM-CM6-1-HR','EC-Earth_HR','GFDL','HadGEM_GC31_HH',
            'HadGEM_GC31_HM_highRes','HadGEM_GC31_HM_highResSST',
            'CMCC-CM2-VHR4']

# Station
SLR_list =['000','025','050','100','150','200','300']


Stat = 562

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
# %% Load the data 
#===============================================================================
ds = xr.open_mfdataset('D:\DFM_Regional\ERA5\ERA5_000\ERA5_cdf_1941.nc')
print(f"Station Selected: {ds['station'].isel(station=Stat).values}")


diff_save = np.full([1000,len(Mod_list),len(SLR_list),12],np.nan)

for SLR_cnt,SLR in enumerate(SLR_list):
    for Mod_cnt,Mod in enumerate(Mod_list):
        for month in np.arange(1, 13, 1, dtype=int):  
            print(f'Processing  SLR {SLR}, Model {Mod}, Month {month}')
    
            with open(os.path.join(dir_in,Mod,'historic',
                                    'CDFmonthly_{0:02d}.pkl'.format(month)), 'rb') as f: 
                cdf_cmipH = pickle.load(f)
        
            # Load the CMIP6 future data
            with open(os.path.join(dir_in,Mod,'future',SLR,
                                    'CDFmonthly_{0:02d}.pkl'.format(month)), 'rb') as f: 
                cdf_cmipF = pickle.load(f)
            
            
            # Subset to station of interest
            cdf_cmipH = cdf_cmipH.loc[cdf_cmipH['stat'] == Stat]
            cdf_cmipF = cdf_cmipF.loc[cdf_cmipF['stat'] == Stat]
            
            # Convert to meters
            cdf_cmipH.values = cdf_cmipH.values/10000
            cdf_cmipF.values = cdf_cmipF.values/10000
    
            diff, quants = calc_diff(cdf_cmipH,cdf_cmipF,'abs')
        
            diff_save[:,Mod_cnt,SLR_cnt,month-1] = diff
    
#===============================================================================
# %% Plots
#===============================================================================    


# Get rid of sea levelf to look at nonlinearities 
diff_NoSLR = np.full(diff_save.shape,np.nan)
for SLR_cnt,SLR in enumerate(SLR_list):

    diff_NoSLR[:,:,SLR_cnt] = (diff_save[:,:,SLR_cnt] - (np.array(SLR).astype(float)/100))*100   # Convert SLR to meters (for subtract). And then convert output to centimeters.


################################################################################
#diff_save = [quantiles,months,SLR,Months[]np.full([1000,len(Mod_list),len(SLR_list),12],np.nan)]

month_labels = ['January','February','March','April','May','June','July','August','September','October','November','December']
fig,ax = matplotlib.pyplot.subplots(4, 3)
fig.set_size_inches(10, 10)
ax = ax.flatten()

for cnt,ax_plot in enumerate(ax):
    data_monthly = diff_NoSLR[:,:,:,cnt].squeeze()
    
    for SLR_cnt,SLR in enumerate(SLR_list):
        data_SLR = data_monthly[:,:,SLR_cnt]
        l1 = ax_plot.plot(quants,np.mean(data_SLR,axis=1))  
        
    l2 = ax_plot.plot([0,1],[0,0],'k--')
    ax_plot.grid()
    ax_plot.set_xlim([0,1])
    ax_plot.set_title(month_labels[cnt])
    #ax_plot.set_ylabel('Change in WL (cm)')
    #ax_plot.set_xlabel('Quantile')
    if (cnt != 9) and (cnt != 10) and (cnt != 11):
        ax_plot.set_xticklabels([])
        
    if (cnt ==6):
        ax_plot.set_ylabel('Diff in Water level (cm)')
        
    if (cnt == 10):
        ax_plot.set_xlabel('Quantile')
    
    if (cnt == 2):
        ax_plot.legend(SLR_list)
        
        

fig.savefig(os.path.join(dir_out,'DFM_Change_BySLR_ByMonth.tiff'), dpi=600)



###########  Plot of mistaken SLR runs  ###########
SLR_list_errors = ['000','050','100','200','300']

fig,ax = matplotlib.pyplot.subplots(4, 3)
fig.set_size_inches(10, 10)
ax = ax.flatten()

for cnt,ax_plot in enumerate(ax):
    data_monthly = diff_NoSLR[:,:,:,cnt].squeeze()
    
    for SLR_cnt,SLR in enumerate(SLR_list_errors):
        if SLR == '000':
            data_used = data_monthly[:,:,0]
            data_true = data_monthly[:,:,0]
            data_SLR = data_true - data_used
            
        elif SLR == '050':
            data_used = data_monthly[:,:,1]
            data_true = data_monthly[:,:,2]
            data_SLR = data_true - data_used
            
        elif SLR == '100':
            data_used = data_monthly[:,:,2]
            data_true = data_monthly[:,:,3]
            data_SLR = data_true - data_used
            
        elif SLR == '200':
            data_used = data_monthly[:,:,3]
            data_true = data_monthly[:,:,5]
            data_SLR = data_true - data_used
            
        elif SLR == '300':
            data_used = data_monthly[:,:,4]
            data_true = data_monthly[:,:,6]
            data_SLR = data_true - data_used
            
        
        l1 = ax_plot.plot(quants,np.mean(data_SLR,axis=1),label = f'{SLR}')  
        
    l2 = ax_plot.plot([0,1],[0,0],'k--')
    ax_plot.grid()
    ax_plot.set_xlim([0,1])
    ax_plot.set_title(month_labels[cnt])
    #ax_plot.set_ylabel('Change in WL (cm)')
    #ax_plot.set_xlabel('Quantile')
    if (cnt != 9) and (cnt != 10) and (cnt != 11):
        ax_plot.set_xticklabels([])
        
    if (cnt ==6):
        ax_plot.set_ylabel('Diff from WRong SLR (cm)')
        
    if (cnt == 10):
        ax_plot.set_xlabel('Quantile')
    
    if (cnt == 2):
        ax_plot.legend()
        
        

fig.savefig(os.path.join(dir_out,'DFM_Change_BySLR_ByMonth_ErrorFromSLR.tiff'), dpi=600)





