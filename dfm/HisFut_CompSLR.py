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
dir_in = r'Y:\PS_Cosmos\DFM'
dir_out = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\figures\DFM'

# Model to process
Mod = 'HadGEM_GC31_HM_highResSST'

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

# CMIP6 His
print('Loading Historic Data')
files = glob(os.path.join(dir_in,Mod,'historic','*.nc'))
ds_cmipH = Load_Station(files,station)

# Subset the historic run to have equal timeperiods
ds_cmipH = ds_cmipH.sel(time=slice("1980-01-01", "2014-01-01"))

# CMIP6 Fut
print('Loading Future Data 000')
files = glob(os.path.join(dir_in,Mod,'future','000','*.nc'))
ds_cmipF000 = Load_Station(files,station)

print('Loading Future Data 050')
files = glob(os.path.join(dir_in,Mod,'future','050','*.nc'))
ds_cmipF050 = Load_Station(files,station)    

print('Loading Future Data 100')
files = glob(os.path.join(dir_in,Mod,'future','100','*.nc'))
ds_cmipF100 = Load_Station(files,station)

print('Loading Future Data 200')
files = glob(os.path.join(dir_in,Mod,'future','200','*.nc'))
ds_cmipF200 = Load_Station(files,station)

print('Loading Future Data 300')
files = glob(os.path.join(dir_in,Mod,'future','300','*.nc'))
ds_cmipF300 = Load_Station(files,station)


#===============================================================================
# %% Remove some bad events
#===============================================================================

# # A couple of bad events for the ec-Earth runs
# ind_bad = (ds_cmipF000['time'] >= pd.Timestamp('2021-03-15')) & (ds_cmipF000['time'] <= pd.Timestamp('2021-03-20'))
# ds_cmipF000['waterlevel'][ind_bad] = np.nan
# ind_bad = (ds_cmipF000['time'] >= pd.Timestamp('2016-12-25')) & (ds_cmipF000['time'] <= pd.Timestamp('2017-01-01'))
# ds_cmipF000['waterlevel'][ind_bad] = np.nan

# ind_bad = (ds_cmipF050['time'] >= pd.Timestamp('2021-03-15')) & (ds_cmipF050['time'] <= pd.Timestamp('2021-03-20'))
# ds_cmipF050['waterlevel'][ind_bad] = np.nan
# ind_bad = (ds_cmipF100['time'] >= pd.Timestamp('2016-12-25')) & (ds_cmipF050['time'] <= pd.Timestamp('2017-01-01'))
# ds_cmipF050['waterlevel'][ind_bad] = np.nan

# ind_bad = (ds_cmipF100['time'] >= pd.Timestamp('2021-03-15')) & (ds_cmipF100['time'] <= pd.Timestamp('2021-03-20'))
# ds_cmipF100['waterlevel'][ind_bad] = np.nan
# ind_bad = (ds_cmipF100['time'] >= pd.Timestamp('2016-12-25')) & (ds_cmipF100['time'] <= pd.Timestamp('2017-01-01'))
# ds_cmipF100['waterlevel'][ind_bad] = np.nan

# ind_bad = (ds_cmipF200['time'] >= pd.Timestamp('2021-03-15')) & (ds_cmipF200['time'] <= pd.Timestamp('2021-03-20'))
# ds_cmipF200['waterlevel'][ind_bad] = np.nan
# ind_bad = (ds_cmipF200['time'] >= pd.Timestamp('2016-12-25')) & (ds_cmipF200['time'] <= pd.Timestamp('2017-01-01'))
# ds_cmipF200['waterlevel'][ind_bad] = np.nan

# ind_bad = (ds_cmipF300['time'] >= pd.Timestamp('2021-03-15')) & (ds_cmipF300['time'] <= pd.Timestamp('2021-03-20'))
# ds_cmipF300['waterlevel'][ind_bad] = np.nan
# ind_bad = (ds_cmipF300['time'] >= pd.Timestamp('2016-12-25')) & (ds_cmipF300['time'] <= pd.Timestamp('2017-01-01'))
# ds_cmipF300['waterlevel'][ind_bad] = np.nan

#===============================================================================
# %% Calculate the statistics
#===============================================================================
print('Calculating statistics')

    
cdf_cmipH = calc_cdf(ds_cmipH['waterlevel'])

cdf_cmipF000 = calc_cdf(ds_cmipF000['waterlevel'])
cdf_cmipF050 = calc_cdf(ds_cmipF050['waterlevel'])
cdf_cmipF100 = calc_cdf(ds_cmipF100['waterlevel'])
cdf_cmipF200 = calc_cdf(ds_cmipF200['waterlevel'])
cdf_cmipF300 = calc_cdf(ds_cmipF300['waterlevel'])


# Difference in Water Levels at all quantiles 
quants = np.arange(0.01, 1, 0.01)


fH = interpolate.interp1d(cdf_cmipH.y, cdf_cmipH.x)
fF000 = interpolate.interp1d(cdf_cmipF000.y, cdf_cmipF000.x)
fF050 = interpolate.interp1d(cdf_cmipF050.y, cdf_cmipF050.x)
fF100 = interpolate.interp1d(cdf_cmipF100.y, cdf_cmipF100.x)
fF200 = interpolate.interp1d(cdf_cmipF200.y, cdf_cmipF200.x)
fF300 = interpolate.interp1d(cdf_cmipF300.y, cdf_cmipF300.x)


q_diff000 = fF000(quants) - fH(quants) 
q_diff050 = fF050(quants) - fH(quants) 
q_diff100 = fF100(quants) - fH(quants) 
q_diff200 = fF200(quants) - fH(quants) 
q_diff300 = fF300(quants) - fH(quants) 


#===============================================================================
# %% Plot
#===============================================================================

# fig, ax = matplotlib.pyplot.subplots(1, 1)
# fig.set_size_inches(8, 6)

# ax.plot(ds_cmipH['time'],ds_cmipH['waterlevel'], color = 'red',label = 'CmipH')
# ax.plot(ds_cmipF['time'],ds_cmipF['waterlevel'], color = 'blue',label = 'CmipF')


# #ax.set_xlim(pd.Timestamp('2016-12-25'),pd.Timestamp('2017-01-01'))
# #ax.set_xlim(100,200)
# ax.grid()
# ax.set_title('Water Levels')
# ax.set_ylabel('WL (NAVD88,mm)')
# ax.set_xlabel('date')
# ax.legend()
# #ax.set_xlim([pd.Timestamp('2021-03-15'),pd.Timestamp('2021-03-20')])
# matplotlib.pyplot.show()
# fig.savefig(os.path.join(dir_out,'DFM_Cmip6_TS'),  dpi=800,
#         bbox_inches='tight')          

fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 8)
ax.plot(cdf_cmipH.x,cdf_cmipH.y, color = 'black',label = 'CmipH')
ax.plot(cdf_cmipF000.x,cdf_cmipF000.y, color = 'blue',label = 'CmipF000')
ax.plot(cdf_cmipF050.x,cdf_cmipF050.y, color = 'green',label = 'CmipF050')
ax.plot(cdf_cmipF100.x,cdf_cmipF100.y, color = 'red',label = 'CmipF100')
ax.plot(cdf_cmipF200.x,cdf_cmipF200.y, color = 'orange',label = 'CmipF200')
ax.plot(cdf_cmipF300.x,cdf_cmipF300.y, color = 'magenta',label = 'CmipF300')

ax.grid()
ax.set_title('Water Level CDF')
ax.set_xlabel('WL (NAVD88,mm)')
ax.set_ylabel('CDF (emp.)')
ax.legend()
matplotlib.pyplot.show()
#fig.savefig(os.path.join(dir_out,'DFM_Cmip6_cdf_fut'),  dpi=800,
#        bbox_inches='tight')  



fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 8)
ax.plot(cdf_cmipH.x,cdf_cmipH.y, color = 'black',label = 'CmipH')
ax.plot(cdf_cmipF000.x,cdf_cmipF000.y, color = 'blue',label = 'CmipF000')
ax.plot(cdf_cmipF050.x,cdf_cmipF050.y, color = 'green',label = 'CmipF050')
ax.plot(cdf_cmipF100.x,cdf_cmipF100.y, color = 'red',label = 'CmipF100')
ax.plot(cdf_cmipF200.x,cdf_cmipF200.y, color = 'orange',label = 'CmipF200')
ax.plot(cdf_cmipF300.x,cdf_cmipF300.y, color = 'magenta',label = 'CmipF300')

ax.set_ylim(0.9,1)
ax.set_xlim(2250,3750)
ax.grid()
ax.set_title('Water Level CDF')
ax.set_xlabel('WL (NAVD88,mm)')
ax.set_ylabel('CDF (emp.)')
ax.legend()
matplotlib.pyplot.show()
#fig.savefig(os.path.join(dir_out,'DFM_Cmip6_cdfZoom_fut'),  dpi=800,
#        bbox_inches='tight')  


fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 8)
ax.plot(q_diff000,quants, color = 'blue',label = '00')
ax.plot(q_diff050-500,quants, color = 'green',label = '50 (minus SLR)')
ax.plot(q_diff100-1000,quants, color = 'red',label = '100 (minus SLR)')
ax.plot(q_diff200-2000,quants, color = 'orange',label = '200 (minus SLR)')
ax.plot(q_diff300-3000,quants, color = 'magenta',label = '300 (minus SLR)')

ax.grid()
ax.set_title('Difference in CMIP6 CDF')
ax.set_xlabel('Diff in WL (mm)')
ax.set_ylabel('Quantile')
ax.legend()

matplotlib.pyplot.show()
fig.savefig(os.path.join(dir_out,'DFM_Cmip6_{}_cdfDiff_fut'.format(Mod)),  dpi=800,
        bbox_inches='tight')  
