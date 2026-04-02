# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:09:24 2024

THis functions sets nan raster data to NoData for arcgis pro to read properly.


@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""


import rasterio
import os 
import numpy as np
from glob import glob 

#===============================================================================
# %% User Defined inputs
#===============================================================================

# Directories 
dir_in  = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\PierceKing\20240531_synthetic\02_Pierce\downscaled_2m'
dir_out  = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\PierceKing\20240531_synthetic\02_Pierce\downscaled_2m'


#===============================================================================
# %% Define functions
#===============================================================================

# Files to import 
files = glob(os.path.join(dir_in,'*zsmax*.tif'))


for file in files:
    
    file_name = os.path.basename(os.path.normpath(file))
    out_name  = file_name.replace('_2m.tif','_2m_clean.tif')
    
    dataset = rasterio.open(os.path.join(dir_in,file_name),'r+')
    dataset.nodata = np.nan
    
    with rasterio.open(os.path.join(dir_out, out_name), 'w', **dataset.profile) as dst:
        for ji, window in dataset.block_windows(1):
            data_block = dataset.read(window=window)
                    
            dst.write(data_block, window=window)          
                       
    dataset.close()