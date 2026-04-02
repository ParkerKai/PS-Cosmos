# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:22:20 2024

This script plots the waterlevels for CMIP6 historic-future runs 


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
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib
import pandas as pd
from scipy import interpolate

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
dir_in = r'D:\DFM'
dir_out = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\figures\DFM'

# Model to process
Mod = 'EC-Earth_HR'

# Station to plot
station = 'NOAA_9447130_Seattle'

#===============================================================================
# %% Define some functions
#===============================================================================
def CheckArrayFor(data,string):
    out = []
    for cnt,row in enumerate(data):
        string = str(row)
        # check if string present on a current line
        if string.find(station) != -1:
            if len(out) == 0:
                out = cnt
            else:
                out = out.append(cnt)

    return out

def Load_Station(files,station_string):
    # Test open
    test_open = xr.open_mfdataset(files[0], engine='netcdf4', parallel=False)
    names = test_open['station'].values
    
    ind_grab = CheckArrayFor(names,station)
    
    
    for cnt,file in enumerate(files):
        data = xr.open_mfdataset(file, engine='netcdf4', parallel=False)
        
        #Subset to station of interest
        data = data.isel(station=ind_grab)    
        
        if cnt == 0 :
            data_save = data
        else:
            data_save = xr.concat([data_save,data],dim='time')
    
    # Convert to millimeters 
    data_save['waterlevel'] = data_save['waterlevel']/10
    return data_save

def calc_cdf(data):
    cdf = ECDF(data)
    
    # Deal with inf and extreme values
    ind_keep = np.isfinite(cdf.x)
    cdf.x = cdf.x[ind_keep]
    cdf.y = cdf.y[ind_keep]
    
    return cdf 

#===============================================================================
# %% Load the data 
#===============================================================================

# ERA5
files = glob(os.path.join(dir_in,'ERA5','*.nc'))
ds_era5 = Load_Station(files,station)


# CMIP6 His
files = glob(os.path.join(dir_in,'Cmip6',Mod,'historic','*.nc'))
ds_cmipH = Load_Station(files,station)

ds_cmipH1 = ds_cmipH.sel(time=slice("1951-01-01", "1982-01-01"))
ds_cmipH2 = ds_cmipH.sel(time=slice("1983-01-01", "2014-01-01"))

#===============================================================================
# %% Calculate the statistics
#===============================================================================
cdf_era5 = calc_cdf(ds_era5['waterlevel'])
cdf_cmip1 = calc_cdf(ds_cmipH1['waterlevel'])
cdf_cmip2 = calc_cdf(ds_cmipH2['waterlevel'])

# Difference in Water Levels at all quantiles 
quants = np.arange(0.01, 1, 0.01)

f1 = interpolate.interp1d(cdf_cmip1.y, cdf_cmip1.x)
f2 = interpolate.interp1d(cdf_cmip2.y, cdf_cmip2.x)

diff = f2(quants)-f1(quants)



#===============================================================================
# %% Plot
#===============================================================================


fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 8)

ax.plot(cdf_era5.x,cdf_era5.y, color = 'black',label = 'ERA5')
ax.plot(cdf_cmip1.x,cdf_cmip1.y, color = 'red',label = 'CmipH_1')
ax.plot(cdf_cmip2.x,cdf_cmip2.y, color = 'blue',label = 'CmipH_2')

#ax.set_xlim(pd.Timestamp('2000-08-01'),pd.Timestamp('2001-08-01'))
#ax.set_xlim(100,200)
ax.grid()
ax.set_title('Water Level CDF')
ax.set_xlabel('WL (NAVD88,mm)')
ax.set_ylabel('CDF (emp.)')
ax.legend()


matplotlib.pyplot.show()
fig.savefig(os.path.join(dir_out,'DFM_Cmip6_cdf_HisPer'),  dpi=800,
        bbox_inches='tight')  



fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 8)

ax.plot(cdf_era5.x,cdf_era5.y, color = 'black',label = 'ERA5')
ax.plot(cdf_cmip1.x,cdf_cmip1.y, color = 'red',label = 'CmipH_1')
ax.plot(cdf_cmip2.x,cdf_cmip2.y, color = 'blue',label = 'CmipH_2')

ax.set_ylim(0.9,1)
ax.set_xlim(2250,3750)

ax.grid()
ax.set_title('Water Level CDF')
ax.set_xlabel('WL (NAVD88,mm)')
ax.set_ylabel('CDF (emp.)')
ax.legend()

matplotlib.pyplot.show()
fig.savefig(os.path.join(dir_out,'DFM_Cmip6_cdfZoom_100'),  dpi=800,
        bbox_inches='tight')  


fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 8)

ax.plot(diff,quants, color = 'black',label = 'DiffPeriods')
ax.plot([0,0],[0,1],'--', color = 'black',label = 'DiffPeriods',linewidth=3)
ax.set_ylim(0,1)

ax.grid()
ax.set_title('Difference in CMIP6 CDF')
ax.set_xlabel('Diff in WL (mm,recent-historic)')
ax.set_ylabel('Quantile')

matplotlib.pyplot.show()
fig.savefig(os.path.join(dir_out,'DFM_Cmip6_cdfDiff_HisPer'),  dpi=800,
        bbox_inches='tight')  
