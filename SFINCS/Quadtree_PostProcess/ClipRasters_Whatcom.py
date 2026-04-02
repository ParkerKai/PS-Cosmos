# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 10:59:52 2025

This script Loads a raster and then clips to a boundary.

@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"

#===============================================================================
# %% Import Modules
#===============================================================================

import xarray as xr
import numpy as np
import rasterio
import fiona
import zipfile
import os
import shutil
from rasterio.mask import mask

#===============================================================================
# %% User Defined inputs
#===============================================================================
  
# Processing information
dir_base             = r'Y:\PS_Cosmos\06_FinalProducts\Whatcom_DataRelease'
RPs                     = ['0000','0001', '0005', '0010', '0020', '0050', '0100','King']
SLRs                    = ['000', '025', '050', '100', '150', '200', '300']
dir_out              = r'Y:\PS_Cosmos\06_FinalProducts\Whatcom_DataRelease_Trimmed'

# Settings
Variable = 'waterlevel'  #   waterdepth waterlevel                                                  


#===============================================================================
# %% Define some functions
#===============================================================================



#===============================================================================
# %% Load the data 
#===============================================================================
    


# Loop over counties

# Loop over SLRs
for SLR in SLRs:
    
    # unzip directory
    zip_file = os.path.join(dir_base,Variable,f'{Variable}_D1_SLR{SLR}.zip')
    zip_out = os.path.join(dir_base,Variable,f'{Variable}_D1_SLR{SLR}')
    
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        print(f'unzipping: {os.path.basename(zip_file)}')
        zip_ref.extractall(zip_out)
    
    # Create destout directory if it doesn't exist
    destout = os.path.join(dir_out,Variable,f'{Variable}_D1_SLR{SLR}')
    if not os.path.exists(destout):
        os.makedirs(destout)
      
    
    # Go over sub_categories (standard, low and high)
    for RP in RPs:
        print(f'Processing SLR {SLR}, RP {RP}')
        
        # File to process    
        file_in = os.path.join(dir_base,Variable,f'{Variable}_D1_SLR{SLR}',f'SLR{SLR}',f'{Variable}_D1_SLR{SLR}_RP{RP}.tif')
        file_out = os.path.join(destout,f'{Variable}_D1_SLR{SLR}_RP{RP}_trimmed.tif')
                    
        # Read clipping polygon
        with fiona.open(os.path.join(dir_out,'FloodExtent',f'Projections_FloodHazard_D1_SLR{SLR}',f'floodtype_D1_SLR{SLR}_RP{RP}_connected.shp'), "r") as shapefile:                        
                clip = [feature["geometry"] for feature in shapefile]
         
         # Load raster (clipped by feature geometry)
        with rasterio.open(file_in) as src:
                data, out_transform = mask(src, clip,
                                           crop=False,
                                           nodata=-9999,
                                           filled=True)
        data = data.squeeze()
        # Export the new raster.
        # Define writing
        kwargs2 = dict(
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype='float32',
            crs=src.crs,  # Assuming 'src' is defined from previous operations
            transform=out_transform,  # Assuming 'src' is defined from previous operations
            tiled=True,
            blockxsize=128,                 # reduced this from 256
            blockysize=128,
            compress="deflate",
            predictor=2,                    # Adjust based on your data's nature (floating-point or integer)
            zlevel=6,                       # reduced to 6 from 9
            profile="COG",
            nodata = -9999)
        
        # Do actual writing
        with rasterio.open(file_out, 'w', **kwargs2) as dst:
            dst.write(data, 1)  # Write the first (and only) band
        
    print(f'Deleting Zip Directory: {os.path.basename(zip_file)}')
    shutil.rmtree(zip_out)


