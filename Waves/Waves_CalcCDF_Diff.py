# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 15:19:18 2024

This script loads in the calculatee cdf for each cmip6 model and then finds the difference as applies to  
to the ERA5 period. Specifically for each quantile value for the ERA5 period it finds 
the different predicted from cmip6 historic to future. It does this for each month 
and each CMIP6 model and then saves as a netcdf.  


# NOTE: this scirpt incorreclty doesn't fix the fact that CMIP6 and ERA5 station are out of order.
        THis is fixed with Waves_CdfDiff_StationFix.py 

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
import pickle
import scipy 

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'X:\PS_Cosmos\LUT\LUT_output'
dir_out = r'X:\PS_Cosmos\LUT\LUT_output\LUt_KingPierce_CMIP6_Diff_DONTUSE'   # Fixed after to show this script incorrectly doesn't fix stations.

# Model to process
Mod_list = ['CMCC','CNRM','EcEarth','GFDL','HadGemHH','HadGemHM','HadGemHMsst']

SLR_list =['300']
#SLR = '000'


#===============================================================================
# %% Define some functions
#===============================================================================

@dask.delayed()
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
    # Calc CDF correction.
    diff = np.full(data_month['hs_quants'].shape, np.nan, dtype = 'float32')  # data_month['wl_quants'].shape
    
    for stat in range(data_month.dims['station']):   #data.dims['station']
        print(f'processing Station: {stat}')
                        
        # PUll data for the station and unwrap pandas dataframe to numpy
        cdf_H_stat_cdf = dask.delayed(cdf_H['cdf'].loc[cdf_H['stat'] == stat].to_numpy())
        cdf_H_stat_val = dask.delayed(cdf_H['values'].loc[cdf_H['stat'] == stat].to_numpy())
        cdf_F_stat_cdf = dask.delayed(cdf_F['cdf'].loc[cdf_F['stat'] == stat].to_numpy())
        cdf_F_stat_val = dask.delayed(cdf_F['values'].loc[cdf_F['stat'] == stat].to_numpy())
        quant_era5     = dask.delayed(data_month['hs_quants'].isel(station=stat).values)
        
        delayedResult = interpAtQuant(cdf_F_stat_val,cdf_F_stat_cdf,quant_era5) - interpAtQuant(cdf_H_stat_val,cdf_H_stat_cdf,quant_era5)
        
        diff[:,stat] = delayedResult.compute()
        
    return diff


def output_yearly(data,dir_out,fname):
    
    year_out = np.unique(data.time.dt.year)
    
    for year in year_out:
        
        print(f'Outputting {year} Chunk')

        out = data.isel(time=data.time.dt.year.isin(year))
        
        out.to_netcdf(os.path.join(dir_out,fname.format(year=year)),engine = 'netcdf4')
 

def main():    
        
    cl= dask.distributed.LocalCluster()
    client=dask.distributed.Client(cl)
    
    print(client.dashboard_link)
    
    
    #===============================================================================
    # %% Load the ERA5 data and calc quantiles
    #===============================================================================
    print('loading ERA5 Data')    
    files = glob(os.path.join(dir_in,'LUT_output_KingPierce_ERA5','netcdf','*.nc'))
    
    concat_list = [i for i in range(len(files))]
    for cnt,file in enumerate(files):
        raw = xr.open_mfdataset(file, engine='h5netcdf', parallel=True, decode_cf=False)

        del raw.Tp.attrs['units']
        
        concat_list[cnt] = xr.decode_cf(raw)
        
    ds_era5 = xr.concat(concat_list, dim='time')

    #===============================================================================
    # %% Calculate correction for ERA5 based on CMIP6 projections
    #===============================================================================    
    for SLR in SLR_list:
        ds_save  = [i for i in range(len(Mod_list))]
        for cnt,Mod in enumerate(Mod_list):
            
            print(f'Processing CMIP6 Difference for {Mod} {SLR}')
            
            # split by month
            # Final numpy array that will be filled in month by month
            diff_full = np.full(ds_era5['Hs'].shape, np.nan, dtype = 'float32')
            
            for month in np.arange(1, 13, 1, dtype=int):
                print(f'Processing Month {month:02d}')
                
                 # Load the CMIP6 historic data
                with open(os.path.join(dir_in,'LUT_output_KingPierce_CMIP6_historical',Mod,
                                       f'CDFmonthly_{month:02d}_{Mod:s}.pkl'), 'rb') as f: 
                    cdf_cmipH = pickle.load(f)
        
                 # Load the CMIP6 future data
                with open(os.path.join(dir_in,'LUT_output_KingPierce_CMIP6_future',Mod,
                                       f'CDFmonthly_{month:02d}_{Mod:s}_SLR{SLR:s}.pkl'), 'rb') as f: 
                    cdf_cmipF = pickle.load(f)
                
                # subset ERA5 to the month
                # Index for the specific month we are processing (used to fill in Diff_full later)
                ind_month =  ds_era5.time.dt.month.isin(month)
                data_month = ds_era5.isel(time=ind_month) 
                
                # Calculate the difference between the historic and future. 
                diff = calc_diff(cdf_cmipH,cdf_cmipF,data_month)    
                        
                # Add this month chunk into the full set
                diff_full[ind_month,:] = diff
                
            # Save into the original Xarray dataset
            ds_era5_diff = xr.DataArray(
                     data   = diff_full,    # enter data here
                     dims   = ['time','station'],
                     coords = {'time': ds_era5['time'],
                               'station':ds_era5['station']},
                     attrs  = {
                         '_FillValue': -9999,
                         'units'     : 'meters'
                         })
            
            ds_era5_diff = ds_era5_diff.chunk({'station':1, 'time':52560})
    
            # Save for concatenating later
            ds_save[cnt] = ds_era5_diff
        
        # Concat
        ds_diff = xr.concat(ds_save, dim='cmip6')
        ds_diff = ds_diff.assign_coords({'cmip6': Mod_list})
        ds_diff = ds_diff.chunk({'station':1, 'time':52560, 'cmip6':1})
        
        ds_full =  xr.Dataset({
            'Hs':ds_era5['Hs'],
            'Dm':ds_era5['Dm'],
            'Tp':ds_era5['Tp'],
            'hs_quants':ds_era5['hs_quants'],
            'cmip_diff':ds_diff,
            'lon':ds_era5['Lon'].isel(time=1, drop=True),
            'Lat':ds_era5['Lat'].isel(time=1, drop=True)},
            attrs = {'DataSource': 'Y:\PS_Cosmos\PS_Cosmos\09_wave_lut_predictions\LUT_output',
                     'ProducedBy': 'Anita Englestad and Kai Parker',
                     'General': 'Tm,Tp and Dm are NaN for Hs < 0.05m. Output is for lat_10mIso/lon_10mIso locations. lat_ncPoint and lon_ncPoint are the original nc points to which the closest gridpoints on the 10m isobath were found'}
        )
        
        output_yearly(ds_full,os.path.join(dir_out,SLR),'ERA5_{year}_Diff.nc')
    
                
    client.shutdown()

if __name__ == '__main__':
    main()