# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:29:25 2024

This script attempts to recreate ERA5 multi-modal spectrums
 
See for ERA5 spectral information:
https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Wavespectra
https://www.ecmwf.int/en/forecasts/documentation-and-support/2d-wave-spectra

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
import xarray as xr 
import math
from glob import glob

import wavespectra



#===============================================================================
# %% User Defined
#===============================================================================
dir_in_param = r'Y:\PNW\ERA5_Waves\WestCoast_SpectralParams'
dir_in_2d = r'Y:\PNW\ERA5_Waves\PNW_Era5Spectral'


dir_out = r'Y:\PNW\ERA5_Waves\PNW_Era5Spectral\ToSwan'

year = 1940

#===============================================================================
# %% Functions
#===============================================================================


#===============================================================================
# %% Read the full 2d Wave spectral information
#===============================================================================
files = glob(os.path.join(dir_in_2d,f'ERA5_SpectralWaves_WC_{year}_*.nc'))

ds = xr.open_mfdataset(files, parallel=True)

ds['dir'] =np.arange(7.5, 352.5 + 15, 15)
ds['freq']=np.full(30, 0.03453) * (1.1 ** np.arange(0, 30))

#See https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Wavespectra
# also need to convert from 1/radians to 1/degrees
ds = (10 ** ds)*(math.pi/180)
ds = ds.fillna(0)

ds_spec = xr.Dataset(
    data_vars=dict(
        efth=(["time", "dir", "freq",'lat','lon'], ds['d2fd'].data),
    ),
    coords=dict(
        time = ds['valid_time'].data,
        dir = ds['dir'].data,
        freq=ds['freq'].data,
        lat=ds['latitude'].data,
        lon=ds['longitude'].data
    ),
    attrs=ds.attrs,
)


ds_spec = wavespectra.SpecDataset(ds_spec)

# Plot the wave significant wave height
hs = ds_spec.isel(lat=0,lon=0).spec.hs()

fig = hs.plot.line(x="time")
ax = matplotlib.pyplot.gca()
ax.grid()


# data.isel(lat=0,lon=0,time=0).spec.plot();
# ax = matplotlib.pyplot.gca()

#===============================================================================
# %% Import parameterized data
#===============================================================================
files = glob(os.path.join(dir_in_param,f'ERA5_WaveParams_{year}_*.nc'))
ds = xr.open_mfdataset(files, parallel=True)





