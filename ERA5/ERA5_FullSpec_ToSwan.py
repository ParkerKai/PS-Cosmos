# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:29:25 2024

This script imports full 2d Spectrum from ERA5, extracts at specific locations, 
and outputs as a SWAN ASCII file. 

Relies on python wavespectra package.
environment: WaveAnalysis
 
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
import geopandas as gpd

import wavespectra


#===============================================================================
# %% User Defined
#===============================================================================
dir_in_2d = r'Y:\PNW\ERA5_Waves\PNW_Era5Spectral'
dir_in_bnd = r'Y:\PNW\mcr\swan\runs\r3\Generic'

dir_out = r'Y:\PNW\mcr\swan\runs\r3'

year = 1980
Bnds =['North','West','South']

#===============================================================================
# %% Functions
#===============================================================================


for Bnd_dir in Bnds:
    print(f'Processing {Bnd_dir} Boundary')
    
    #===============================================================================
    # %% Read boundary location
    #===============================================================================
    
    bnds = pd.read_csv(os.path.join(dir_in_bnd,f'WaveBndLoc_{Bnd_dir}.csv'))
    
    bnds = gpd.GeoDataFrame(geometry=gpd.points_from_xy(bnds['X'], bnds['Y'],crs='EPSG:32610'))
    bnds_latlon = bnds.to_crs("EPSG:4326")
    
    
    #===============================================================================
    # %% Read Wave spectral information
    #===============================================================================
    files = glob(os.path.join(dir_in_2d,f'ERA5_SpectralWaves_WC_{year}_*'))
    
    ds = xr.open_mfdataset(files, parallel=True)
    
    ds['dir'] =np.arange(7.5, 352.5 + 15, 15)
    ds['freq']=np.full(30, 0.03453) * (1.1 ** np.arange(0, 30))
    
    #See https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Wavespectra
    # also need to convert from 1/radians to 1/degrees
    ds = (10 ** ds)*(math.pi/180)
    ds = ds.fillna(0)
    
    ds = xr.Dataset(
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
    
    
    # Export each bound point
    for Bnd_pnt in range(bnds.shape[0]):
        print(f'Boundary Point {Bnd_pnt+1}')

        
        ds_pnt = ds.sel(lat=bnds_latlon['geometry'].y[Bnd_pnt],
                                   lon=bnds_latlon['geometry'].x[Bnd_pnt],
                    method = "nearest")
    
        # Turn into spectral dataset 
        ds_spec = wavespectra.SpecDataset(ds_pnt)
                              
    
        # Plot the wave significant wave height
        # hs = ds_spec.spec.hs()
        
        # fig = hs.plot.line(x="time")
        # ax = matplotlib.pyplot.gca()
        # ax.grid()
        
        
        # data.isel(lat=0,lon=0,time=0).spec.plot();
        # ax = matplotlib.pyplot.gca()
        
        #===============================================================================
        # %% Export to SWAN
        #===============================================================================
        
        ds_spec.to_swan(os.path.join(dir_out,str(year),f'{Bnd_dir}{Bnd_pnt+1}.bnd'))
year
