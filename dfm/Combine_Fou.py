# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:11:14 2024


This script tries to figure out how to combined DFM .fou files into a single timeseries

@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"

# Import Modules
import os
import xarray as xr
import numpy as np
import pandas as pd
#import datetime
import geopandas as gpd 
import matplotlib
#import netcdf4
import contextily as cx
from glob import glob


#===============================================================================
# %% User Defined inputs
#===============================================================================

# Directory where DFM .fou Data is stored
dir_in = r"Y:\PS_Cosmos\02_models\DFM_Regional\CMIP6\EC-Earth_HR\future\Results_Combined\000\Fou"

# Output directory
dir_out = r"C:\Users\kaparker\Documents\Projects\PS_Cosmos\DFM_Temp\Cmip6_Fou"
dir_out_plt = r"C:\Users\kaparker\Documents\Projects\PS_Cosmos\DFM_Temp\figures"

# Model
Mod = 'EcEarth'

# Number of subdirectories for the DFM run 
num_dirs = 40

os.environ['REQUEST_CA_BUNDLE'] = r'C:\Users\kaparker\Documents\Software\Certificate'

#===============================================================================
# %% Define functions
#===============================================================================

def get_files_nc(directory):
    listing = os.listdir(directory)
    
    # Create a list of all netcdfs in the directory
    # Number of files
    num_files = 0
    for cnt,file in enumerate(listing):
        if file.endswith('.nc'):
            num_files = num_files +1 
    
    cnt = 0
    files = ['empty']*num_files
    for file in listing:
        if file.endswith('.nc'):
            files[cnt] = os.path.join(directory,file) 
            cnt = cnt +1
    # or use the glob strategy
    # from glob import glob
    # files = glob(os.path.join(directory,'*.nc'))

    return files

def Read_Fou(file_in):
    
    # Read the file
    data = xr.open_mfdataset(file_in, parallel=True, engine='netcdf4')
    
    # Subset to variables of interest
    out = data[['mesh2d_fourier003_max_depth','mesh2d_fourier003_max',
                'mesh2d_fourier002_min_depth', 'mesh2d_fourier002_min',
                'mesh2d_fourier001_mean', 'mesh2d_face_x', 'mesh2d_face_y']]
    
    return out

def Combine_Fou(files_in):
    year = np.full(len(files_in),0,dtype=int)
    for cnt,file_in in enumerate(files_in):
        # Find the year
        ind = file_in.find('WY_')
        year[cnt] = np.array(file_in[ind+3:ind+7],dtype=int)
        
        #print('Processing {}'.format(year[cnt]))
        
        ds = Read_Fou(file_in)
        
        if cnt == 0:
            data_out = ds
        else:
            data_out = xr.concat([data_out,ds],'year')
            
        
    data_out = data_out.assign_coords({"year": year})
    
    return data_out

def Combine_Fou_SubDir(dir_in,num_dirs):
    # Wrapper to combine FOu with multiple sub-domains.
    
    # Load the storm surge Netcdfs
    for subdir in range(num_dirs):
        print('Processing SubDomain {}'.format(str(subdir).zfill(4)))

        files_in = glob(os.path.join(dir_in,'*{}_fou.nc'.format(str(subdir).zfill(4))))

        data = Combine_Fou(files_in)
        if subdir == 0:
            data_out = data
        else:
             data_out = xr.concat([data_out,data],dim = 'mesh2d_nFaces') 
             
    return data_out

def convert2gpd(data):

    # Turn into a dataframe
    d = {'max_depth': data['mesh2d_fourier003_max_depth'].max(dim='year').values,
         'max_wl': data['mesh2d_fourier003_max'].max(dim='year').values,
         'min_depth': data['mesh2d_fourier002_min_depth'].min(dim='year').values,
         'min_wl': data['mesh2d_fourier002_min'].min(dim='year').values,
         'mean': data['mesh2d_fourier002_min'].mean(dim='year').values}
    
    # Convert to pandas dataframe
    stats_out = pd.DataFrame(data=d)
    
    # add geometry and turn into geopandas dataset
    geometry = gpd.points_from_xy(x=data['mesh2d_face_x'],
                                  y=data['mesh2d_face_y'],
                                  crs = 'EPSG:4326')
    
    out = gpd.GeoDataFrame(data =stats_out, geometry=geometry)
    
    return out 

#===============================================================================
# %% Load the  data
#===============================================================================

data = Combine_Fou_SubDir(dir_in,num_dirs)

#===============================================================================
# %% Load the  data
#===============================================================================


gpd_data = convert2gpd(data)

gpd_data.to_file(os.path.join(dir_out,'Fou_{}.shp'.format(Mod)))


#===============================================================================
# %% Plot the data
#===============================================================================

fig, ax = matplotlib.pyplot.subplots()
ax2 = gpd_data[['max_wl','geometry']].plot(ax = ax,
                    marker='.', markersize=1,
                    cmap='OrRd', alpha=.4, figsize=(8, 7),
                    legend=True, legend_kwds={"label": "max Wl", "orientation": "vertical"})

#cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,crs=gpd_data.crs)
ax.grid()
ax.set_title('Fourier Max WL')
ax.set_ylabel('Elevation (NAVD88,m)')

# add colorbar
# cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
# sm = matplotlib.pyplot.cm.ScalarMappable(cmap='OrRd', norm=matplotlib.pyplot.Normalize(vmin=850, vmax=1500))
# # fake up the array of the scalar mappable. Urgh...
# sm._A = []
# fig.colorbar(sm, cax=cax)

fig.savefig(os.path.join(dir_out_plt,'MaxWl_{}.png'.format(Mod)),  dpi=800,
        bbox_inches='tight')        
