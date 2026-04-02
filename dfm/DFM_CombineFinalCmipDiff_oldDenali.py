# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:54:34 2024

This script loads a specific year of data and combines into a single year file.
Also makes plots
 
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
import dask.distributed

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_out = r'C:\Users\kai\Documents\KaiRuns\DFM\cdf_diff_combined'
dir_in_era5 = r'C:\Users\kai\Documents\KaiRuns\DFM\ERA5_050'
dir_in_diff = r'C:\Users\kai\Documents\KaiRuns\DFM\cdf_diff\050'

# Model to process
Mod_list = ['CMCC-CM2-VHR4','CNRM-CM6-1-HR','GFDL','EC-Earth_HR','HadGEM_GC31_HH','HadGEM_GC31_HM_highRes','HadGEM_GC31_HM_highResSST']

#SLR_list =['000','025','050','100','150','200','300']

year_list  = np.arange(1941,1950,1)


#===============================================================================
# %% Define some functions
#===============================================================================


def convert2int(DataArray,nan_val):
    
    out = DataArray.fillna(nan_val)
    out = out.astype(dtype = 'int32')

    return out    



def Chunk2Yearly(data,dir_out,fname):
    
    year_out = np.unique(data.time.dt.year)
    
    out_data = []
    out_file = []
    
    for year in year_out:
        out_data.append(data.isel(time=data.time.dt.year.isin(year)))
        out_file.append(os.path.join(dir_out,fname.format(year=year)))

    return out_data, out_file

    
    
    
#===============================================================================
# %% Main
#===============================================================================

def main():    

    #===============================================================================
    # %% Load the ERA5 data and calc quantiles
    #===============================================================================  
    #files = glob(os.path.join(dir_in_era5,'Results_Combined','ERA5_cdf*'))
    files = [os.path.join(dir_in_era5,'Results_Combined',f'ERA5_cdf_{year}.nc') for year in year_list]

    ds_era5 = xr.open_mfdataset(files, engine='h5netcdf', parallel=True,
                                combine='by_coords',
                                chunks = {'station':1, 'time':-1})
    
    # Reset station as unicdoe
    # THis is so adding ds_diff can match stations
    
    station_id = ds_era5['station'].to_numpy().astype('unicode')
    ds_era5 = ds_era5.assign_coords({'station':station_id})
    
    #===============================================================================
    # %% Load the CMIP6 difference data
    #===============================================================================   

    # Load the cmip6 difference data (monthly)
    #files = glob(os.path.join(dir_in_diff,'*.nc'))
    files = []
    for year in year_list:
        files.extend(glob(os.path.join(dir_in_diff,f'*{year}*.nc')))
    
    ds_diff = xr.open_mfdataset(files, engine='h5netcdf', parallel=True,
                                combine='by_coords',
                                chunks = {'station':1, 'time':-1, 'cmip6':1})
    
    #===============================================================================
    # %% Modify the file 
    #===============================================================================   
    
    # Add to create a new file
    ds_diff = ds_diff.chunk({'station':1, 'time':-1, 'cmip6':1})
    
    # Convert some of the data to integers for filesize savings 
    ds_era5['waterlevel'] = convert2int(ds_era5['waterlevel'],-9999)
    
    ds_diff = convert2int(ds_diff,-9999)
    ds_diff = ds_diff.assign_coords({'station':station_id})
    ds_era5['cmip_diff'] = ds_diff['cmip_diff']
    
    
    # SEt some attributes to the varialbes 
    ds_era5['cmip6'].attrs = {'long_name':'Cmip6 Model (HighResMIP)'}
    ds_era5['waterlevel'].attrs = {'units': 'm',
                                   'standard_name': 'sea_surface_height',
                                   'long_name': 'water level',
                                   'reference': 'NAVD88'}
    ds_era5['wl_quants'].attrs = {'units': 'None',
                                  'long_name':'Waterlevel Quantile (Monthly)',
                                  'Desc':'Quantiles determined monthly for all data in timeseries within specific month'}
    ds_era5['cmip_diff'].attrs = {'long_name': 'Cmip6 Difference in water levels',
                                  'units': 'meters',
                                  'Desc':'Cmip6 difference for each ERA5 Waterlevel value (as determined by monthly quantile)'}

    ds_era5['time'].attrs = {}


    datasets,paths = Chunk2Yearly(ds_era5,dir_out,'ERA5_{year}_Diff.nc')
    
    # ds_era5.to_netcdf(os.path.join(dir_out,f'ERA5wl_Diff_{year}.nc'), engine = 'netcdf4',
    #                   encoding={'waterlevel':{'dtype': 'int32'},
    #                             'wl_quants':{'dtype':'float32'},
    #                             'cmip_diff':{'dtype': 'int32'},
    #                             'lon':{'dtype': 'float32'},
    #                             'lat':{'dtype':'float32'},
    #                             'bedlevel':{'dtype':'float32'}})
    
    #===============================================================================
    # %% Save as modified file
    #===============================================================================      
    xr.save_mfdataset(datasets=datasets, paths=paths)


if __name__ == '__main__':
    main()

