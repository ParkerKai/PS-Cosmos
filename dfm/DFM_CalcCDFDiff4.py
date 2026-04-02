# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:58:59 2024

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
import pandas as pd
import dask.distributed
import pickle
import scipy 

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'C:\Users\kai\Documents\KaiRuns\DFM'
dir_in_era5 = r'D:\Kai\DFM\ERA5'
dir_out = r'D:\Kai\DFM\ERA5_CMIP6'

# Model to process
Mod_list = ['CMCC-CM2-VHR4','CNRM-CM6-1-HR','EC-Earth_HR','GFDL','HadGEM_GC31_HH','HadGEM_GC31_HM_highRes','HadGEM_GC31_HM_highResSST']

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


def calc_diff(cdf_H,cdf_F,data_month):  
    diff = np.full(data_month['wl_quants'].shape, -9999, dtype = 'int32')
    
    for stat in range(data_month.dims['station']):      
        cdf_H_stat_cdf = cdf_H['cdf'].loc[cdf_H['stat'] == stat].to_numpy()
        cdf_H_stat_val = cdf_H['values'].loc[cdf_H['stat'] == stat].to_numpy()
        cdf_F_stat_cdf = cdf_F['cdf'].loc[cdf_F['stat'] == stat].to_numpy()
        cdf_F_stat_val = cdf_F['values'].loc[cdf_F['stat'] == stat].to_numpy()
        quant_era5     = data_month['wl_quants'].isel(station=stat).values
                      
        if (quant_era5.size == 0) or (cdf_H_stat_cdf.size == 0) or (cdf_F_stat_cdf.size == 0):
            print(f'No Data for Station {stat}')
                                                                                          
        else:
            diff_stat = interpAtQuant(cdf_F_stat_val,cdf_F_stat_cdf,quant_era5) - interpAtQuant(cdf_H_stat_val,cdf_H_stat_cdf,quant_era5)
            diff_stat  = np.nan_to_num(diff_stat,nan=-9999, posinf=-9999, neginf=-9999)
            diff_stat = np.round(diff_stat).astype(dtype='int32')
            diff[:,stat] = diff_stat

    return diff


def output_yearly(data,dir_out,fname):
    
    year_out = np.unique(data.time.dt.year)
    
    for year in year_out:
        
        print(f'Outputting {year} Chunk')

        out = data.isel(time=data.time.dt.year.isin(year))
        
        out.to_netcdf(os.path.join(dir_out,fname.format(year=year)),engine = 'h5netcdf')
 
def convert2unicode(data,axis):
    # data is a matrix of characters that need to be combined into a row of strings
    if (axis == 0):
        data_string = np.full(data.shape[0],'',dtype='<U{}'.format(data.shape[1]))

    elif (axis == 1):
        data_string = np.full(data.shape[1],'',dtype='<U{}'.format(data.shape[0]))

    for ii in range(data.shape[axis]):
        if (axis == 0):
            data_string[ii] = "".join(data[ii,:].astype('unicode'))

        elif (axis == 1):
            data_string[ii] = "".join(data[:,ii].astype('unicode'))

    return data_string

def main():    
        
    cl= dask.distributed.LocalCluster()
    client=dask.distributed.Client(cl)
    
    print(client.dashboard_link)
    
    #===============================================================================
    # %% Load the ERA5 data and calc quantiles
    #===============================================================================
    print('loading ERA5 Data')    
    files = glob(os.path.join(dir_in_era5,'cdf','ERA5_cdf*'))
    
    for file in files:
        
        ds_era5 = xr.open_mfdataset(file, engine='h5netcdf', parallel=True,
                                      chunks = {'station':1, 'time':52560})
        
        year = np.unique(ds_era5.time.dt.year)[0]
        print(f'Processing Year {year}')
        
        #===============================================================================
        # %% Calculate correction for ERA5 based on CMIP6 projections
        #===============================================================================    
  
                                              
        # split by month
        for month in np.arange(1, 13, 1, dtype=int):
            print(f'Processing Month {month:02d}')
                            
        
            # subset ERA5 to the month
            # Index for the specific month we are processing (used to fill in Diff_full later)
            ind_month =  ds_era5.time.dt.month.isin(month)
            data_month = ds_era5.isel(time=ind_month) 
            
            diff = np.full([data_month['time'].size, data_month['station'].size, len(Mod_list)],-9999, dtype='int32')
            for cnt,Mod in enumerate(Mod_list):
                print(f'Processing CMIP6 Difference for {Mod} {SLR}',flush=True)
                    
                 # Load the CMIP6 historic data
                with open(os.path.join(dir_in,'CMIP6',Mod,'historic','Results_Combined',
                                       'CDFmonthly_{0:02d}.pkl'.format(month)), 'rb') as f: 
                    cdf_cmipH = pickle.load(f)
             
                 # Load the CMIP6 future data
                with open(os.path.join(dir_in,'CMIP6',Mod,'future','Results_Combined',SLR,
                                       'CDFmonthly_{0:02d}.pkl'.format(month)), 'rb') as f: 
                    cdf_cmipF = pickle.load(f)
                
                
                # Calculate the difference between the historic and future. 
                diff[:,:,cnt] = calc_diff(cdf_cmipH,cdf_cmipF,data_month)    
            
                
            # Convert station_id to a unicode string
            station_id = data_month['station'].to_numpy().astype('unicode')
            
            ds_diff = xr.DataArray(
                data = diff,
                     dims   = ['time','station','cmip6'],
                     coords = {'time': data_month['time'],
                               'station':station_id,
                               'cmip6': Mod_list},
                     attrs  = {'long_name':'Cmip6 Difference in water levels',
                               'Date': f'year {year}, month {month}',
                         'units' : 'meters/10000'
                         })
                
            ds_diff = ds_diff.chunk({'cmip6':1,'time':-1, 'station':1})

            # Post process some variables for export
            wl_out = data_month['waterlevel'].fillna(-9999)
            wl_out = wl_out.round(0).astype(dtype='int32')
            wl_out = wl_out.chunk({'time':-1, 'station':1})
            
            ds_wl = xr.DataArray(
                data = wl_out,
                     dims   = ['time','station'],
                     coords = {'time': data_month['time'],
                               'station':station_id},
                     attrs  = {'long_name':'ERA5 water levels',
                               'Date': f'year {year}, month {month}',
                         'units' : 'meters/10000'
                         })
            

           # temp = data_month.assign_coords({'station':station_id})
            
            ds_full=  xr.Dataset({
                'wl':ds_wl,
                'cmip_diff':ds_diff})
            
            
            # ds_full=  xr.Dataset({
            #     'wl':wl_out,
            #     'wl_quant':data_month['wl_quants'],
            #     'cmip_diff':ds_diff,
            #     'lon':data_month['lon'],
            #     'lat':data_month['lat']})
            
            # Rewrite the station ID with the new variable
            ds_full = ds_full.assign_coords({'station':station_id})

            ds_full.to_netcdf(os.path.join(dir_out,f'ERA5wl_Diff_{year}_{month:02d}.nc'), engine = 'netcdf4',
                              encoding={'wl':{'dtype': 'int32'},
                                        'cmip_diff':{'dtype': 'int32'}})    

            ds_full.to_netcdf(os.path.join(dir_out,f'ERA5wl_Diff_{year}_{month:02d}.nc'), engine = 'netcdf4',
                              encoding={'wl':{'dtype': 'int32'},
                                        'wl_quant':{'dtype':'float32'},
                                        'cmip_diff':{'dtype': 'int32'}})                    
                
                
    client.shutdown()

if __name__ == '__main__':
    main()