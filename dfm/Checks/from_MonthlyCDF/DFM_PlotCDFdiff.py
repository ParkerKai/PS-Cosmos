# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:50:56 2024

This script loads in the calculatee cdf for each cmip6 model and then applies 
to the ERA5 period. Specifically for each quantile value for the ERA5 period it finds 
the different predicted from cmip6 historic to future. It does this for each month 
and each CMIP6 model and then saves as a netcdf.  

 
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
Mod_list = ['CMCC-CM2-VHR4','CNRM-CM6-1-HR','GFDL']  # 

# Station
Stat = 562

#SLR_list =['000','025','050','100','150','200','300']
SLR = '000'


#===============================================================================
# %% Define some functions
#===============================================================================

def interpAtQuant(cdf_vals,cdf_quant,data_quant):
    
    # Determine CDF based on the pre-calculated ERA5 cdf 
    interp_era5 = scipy.interpolate.interp1d(cdf_quant,
                                          cdf_vals,
                                          fill_value=(cdf_vals.min(),cdf_vals.max()),
                                          copy=False,
                                          assume_sorted=True,
                                          bounds_error=False)
     
    vals = interp_era5(data_quant)

    return vals 

def calc_diff(cdf_H,cdf_F,data):    
    # PUll data for the station and unwrap pandas dataframe to numpy
    cdf_H_stat_cdf = cdf_H['cdf'].to_numpy()
    cdf_H_stat_val = cdf_H['values'].to_numpy()
    cdf_F_stat_cdf = cdf_F['cdf'].to_numpy()
    cdf_F_stat_val = cdf_F['values'].to_numpy()
    quant_era5     = data['wl_quants'].values
    
    diff  = interpAtQuant(cdf_F_stat_val,cdf_F_stat_cdf,quant_era5) - interpAtQuant(cdf_H_stat_val,cdf_H_stat_cdf,quant_era5)
                
    return diff


def output_yearly(data,dir_out,fname):
    
    year_out = np.unique(data.time.dt.year)
    
    for year in year_out:
        
        print(f'Outputting {year} Chunk')

        out = data.isel(time=data.time.dt.year.isin(year))
        
        out.to_netcdf(os.path.join(dir_out,fname.format(year=year)),engine = 'h5netcdf')

#===============================================================================
# %% Load the ERA5 data and calc quantiles
#===============================================================================
print('loading ERA5 Data')    
files = glob(os.path.join(dir_in_era5,'cdf','*.nc'))
ds_era5 = xr.open_mfdataset(files, engine='h5netcdf', parallel=True)
ds_era5 = ds_era5.isel(station=Stat)

#===============================================================================
# %% Calculate correction for ERA5 based on CMIP6 projections
#===============================================================================    

ds_save  = [i for i in range(len(Mod_list))]
for cnt,Mod in enumerate(Mod_list):
    
    print(f'Processing CMIP6 Difference for {Mod}')

    
    # split by month
    # Final numpy array that will be filled in month by month
    diff_full = np.full(ds_era5['waterlevel'].shape, np.nan, dtype = 'float32')
    
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
        
        # subset ERA5 to the month
        # Index for the specific month we are processing (used to fill in Diff_full later)
        ind_month =  ds_era5.time.dt.month.isin(month)
        data_month = ds_era5.isel(time=ind_month) 
        
        # Calculate the difference between the historic and future. 
        diff = calc_diff(cdf_cmipH,cdf_cmipF,data_month)    
                
        # Add this month chunk into the full set
        diff_full[ind_month] = diff
        
    # Save into the original Xarray dataset
    ds_era5_diff = xr.DataArray(
             data   = diff_full,    # enter data here
             dims   = ['time'],
             coords = {'time': ds_era5['time']},
             attrs  = {
                 '_FillValue': -9999,
                 'units'     : 'meters/10000'
                 })
    
    ds_era5_diff = ds_era5_diff.chunk({ 'time':52560})
    # Save for concatenating later
    ds_save[cnt] = ds_era5_diff

# Concat
ds_diff = xr.concat(ds_save, dim='cmip6')
ds_diff = ds_diff.assign_coords({'cmip6': Mod_list})
ds_diff = ds_diff.chunk({'time':52560, 'cmip6':1})

ds_full =  xr.Dataset({
    'wl':ds_era5['waterlevel']/10000,
    'wl_quant':ds_era5['wl_quants'],
    'cmip_diff':ds_diff/10000,
    'lon':ds_era5['lon'].isel(time=1, drop=True),
    'Lat':ds_era5['lat'].isel(time=1, drop=True),
    'bedlevel':ds_era5['bedlevel'].isel(time=1, drop=True)})

ds_full['wl'] = ds_full['wl'].chunk({'time':52560})

#===============================================================================
# %% Plots
#===============================================================================    

fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 6)

l1 = ax.scatter(ds_full['wl'],ds_full['wl_quant'],
                s = 10, marker = '.' ,color = 'k', )



               
fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 6)

l1 = ax.plot(ds_full['time'], ds_full['wl'],color = 'k',
              label = 'WaterLevel')

l2 = ax.plot(ds_full['time'], ds_full['wl']+ds_full['cmip_diff'].sel(cmip6='CMCC-CM2-VHR4'), color = 'r',
              label = 'WaterLevel')

l3 = ax.plot(ds_full['time'], ds_full['wl']+ds_full['cmip_diff'].sel(cmip6='CNRM-CM6-1-HR'), color = 'b',
              label = 'WaterLevel')


l4 = ax.plot(ds_full['time'], ds_full['wl']+ds_full['cmip_diff'].sel(cmip6='GFDL'), color = 'g',
              label = 'WaterLevel')


ax.set_xlim(pd.Timestamp('1962-10-10'),pd.Timestamp('1962-10-14'))
#ax.set_xlim(100,200)
ax.grid()
ax.set_title('Water Level ')
ax.set_ylabel('WL (m)')
ax.set_xlabel('Date')

ax.legend()


#######################################################################

#x = np.concatenate([ds_full['time'].values,np.flipud(ds_full['time'].values)])
#y = np.concatenate([ds_full['wl']+ds_full['cmip_diff'].max(dim='cmip6'), 
#             np.flipud(ds_full['wl']-ds_full['cmip_diff'].min(dim='cmip6'))])


fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 6)

l2 = ax.plot(ds_full['time'],ds_full['wl']+ds_full['cmip_diff'].mean(dim='cmip6'),
                     color = 'b')

l = ax.fill_between(ds_full['time'],ds_full['wl']+ds_full['cmip_diff'].max(dim='cmip6'),
                     ds_full['wl']+ds_full['cmip_diff'].min(dim='cmip6'),
                     color = 'b',alpha = 0.8)


l1 = ax.plot(ds_full['time'], ds_full['wl'],color = 'k',
              label = 'WaterLevel')

ax.set_xlim(pd.Timestamp('1962-10-10'),pd.Timestamp('1962-10-14'))
#ax.set_xlim(100,200)
ax.grid()
ax.set_title('Water Level ')
ax.set_ylabel('WL (m)')
ax.set_xlabel('Date')

################################################################################

fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 8)

l1 = ax.scatter(ds_full['wl'],ds_full['cmip_diff'].sel(cmip6='CMCC-CM2-VHR4'),
                s = 10, marker = '.' ,color = 'k', label = 'CMCC-CM2-VHR4')

#l2 = ax.scatter(ds_full['wl'],ds_full['cmip_diff'].sel(cmip6='CNRM-CM6-1-HR'),
#             s = 10, marker = '.' ,color = 'r', label = 'CNRM-CM6-1-HR')

#l3 = ax.scatter( ds_full['wl'],ds_full['cmip_diff'].sel(cmip6='GFDL'),
#             s = 10, marker = '.' , color = 'b', label = 'GFDL')


#ax.fill(x,y,color='b',alpha = 0.5)


#ax.set_xlim(100,200)
ax.grid()
ax.set_title('Water Level Diff by Water Level ')
ax.set_ylabel('WL Diff (m)')
ax.set_xlabel('Water level (m)')

ax.legend()

##############################################################################




