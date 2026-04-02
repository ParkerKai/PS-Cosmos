# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 07:54:34 2024


This script patches runs that didn't run on the hpc
calcCDF_diff4.py fails for certain months so this goes through, finds
the runs that need to be re-done, and then does them.

 
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
from dask.distributed import Client
import pickle
import scipy 

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'Y:\PS_Cosmos\DFMts/usgs/hazards/pcmsc/cosmos/PS_CoSMoS/cmip6'
dir_in_era5 = r'Y:\PS_Cosmos\DFM\ERA5'
dir_out = r'Y:\PS_Cosmos\DFM\cdf_diff'

# Model to process
Mod_list = ['CMCC-CM2-VHR4','CNRM-CM6-1-HR','GFDL','EC-Earth_HR','HadGEM_GC31_HH','HadGEM_GC31_HM_highRes','HadGEM_GC31_HM_highResSST']

#SLR_list =['000','025','050','100','150','200','300']
#SLR = '000'

#===============================================================================
# %% Define some functions
#===============================================================================

def getfile_date(files):
    year = np.full(len(files),-9999,dtype='int32')
    mnth = np.full(len(files),-9999,dtype='int32')
    for cnt,file in enumerate(files):
        file = os.path.basename(file)
        year[cnt] = np.asarray(file[-10:-6],dtype='int32')
        mnth[cnt] = np.asarray(file[-5:-3],dtype='int32')
        
    return year,mnth

def get_MissingRuns(files):
    
    # Return date/month for runs that completed
    file_yr,file_mn = getfile_date(files)
    file_date = np.transpose(np.vstack((file_yr,file_mn)))
    
    # Compile runs that should have completed
    yr_strt = file_yr.min()
    yr_end  = file_yr.max()
    years = np.arange(yr_strt,yr_end+1,dtype='int32')
    
    want_yr = np.full(years.size*12,-9999,dtype='int32')
    for cnt,yr in enumerate(years):    
        want_yr[cnt*12:cnt*12+12] = np.full(12,yr,dtype='int32')
        
    want_mn = np.full(years.size*12,-9999,dtype='int32')
    for cnt in range(years.size):    
        want_mn[cnt*12:cnt*12+12] = np.arange(1,12+1,dtype='int32')    
    
    want_date = np.transpose(np.vstack((want_yr,want_mn)))
    
    # Check if year,month runs are in the wanted array
    run_exists = np.full(want_date.shape[0],False)
    for ii in range(want_date.shape[0]):
        run_exists[ii] = any(np.all(file_date == want_date[ii,:], axis=1))
    
    # We also don't want any files that are too small
    for file in files:
        size = os.path.getsize(file)
        if size < 3e+8:
            run_exists[ii] = False
        
    runs_want = want_date[run_exists==False,:]
    
    #num_runs = run_exists.shape[0]
    #num_failed = runs_want.shape[0]
    #print(f'{num_failed} Runs out of {num_runs} need to be re-run')
    
    return runs_want


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
    diff = np.full(data_month['wl_quants'].shape, -9999, dtype = 'int32')  # data_month['wl_quants'].shape
    
    for stat in range(data_month.dims['station']):   #data.dims['station']
        #print(f'processing Station: {stat}')
                        
        # PUll data for the station and unwrap pandas dataframe to numpy
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



#===============================================================================
# %% Main
#===============================================================================

def main():    
    
    # Parse inputs
    # import argparse
    
    # parser = argparse.ArgumentParser(description='CDF_Diff_Yearly')
    # parser.add_argument("year", nargs=1,default="2000")
    # parser.add_argument("SLR", nargs=1,default="000")
    # args=parser.parse_args()
    
    # SLR =  args.SLR[0]
    # year = args.year[0]
    
    # print(f'Processing {year} for {SLR}')
    # print(f'SLURMD_NODENAME: {os.environ.get("SLURMD_NODENAME")}')

    # # Start up the cluster
    # dask.config.set({'distributed.scheduler.active-memory-manager.measure': 'optimistic'})
    # dask.config.set({'distributed.worker.memory.recent-to-old-time': 60})
    # dask.config.set({'distributed.worker.memory.rebalance.measure': 'managed'})
    # dask.config.set({'distributed.worker.memory.spill': False})
    # dask.config.set({'distributed.worker.memory.pause': False})
    # dask.config.set({'distributed.worker.memory.terminate': False})
    # dask.config.set({'distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_': 1})

    # client = Client(n_workers=16, threads_per_worker=2)  # , diagnostics_port=None)
    # print(f'Number of workers: {len(client.ncores())}') 
    
    SLR= '000'
    year = '1980'
    
    #===============================================================================
    # %% Load the ERA5 data and calc quantiles
    #===============================================================================
    print('loading ERA5 Data')    
    #files = glob(os.path.join(dir_in_era5,'ERA5_cdf*'))
    files = os.path.join(dir_in_era5,SLR,'cdf',f'ERA5_cdf_{year}.nc')

    ds_era5 = xr.open_mfdataset(files, engine='netcdf4', parallel=True,
                                  chunks = {'station':1, 'time':-1})
    
    #===============================================================================
    # %% Figure out missing months
    #===============================================================================
    
    # get file list
    files = glob(os.path.join(dir_out,SLR,'*.nc'))
    
    # Get files that need to be run
    runs_want = get_MissingRuns(files)

    months_want = runs_want[(runs_want[:,0] == np.array(year).astype('int32')),1]
    print(f'Needed months for {year}: {months_want}') 
        
    #===============================================================================
    # %% Calculate correction for ERA5 based on CMIP6 projections
    #===============================================================================   

    # split by month
    for month in months_want:
        print(f'Processing Month {month:02d}')

        # subset ERA5 to the month
        # Index for the specific month we are processing (used to fill in Diff_full later)
        ind_month =  ds_era5.time.dt.month.isin(month)
        data_month = ds_era5.isel(time=ind_month)

        diff = np.full([data_month['time'].size, data_month['station'].size, len(Mod_list)],-9999, dtype='int32')
        for cnt,Mod in enumerate(Mod_list):
            print(f'Processing CMIP6 Difference for {Mod} {SLR}',flush=True)

             # Load the CMIP6 historic data
            with open(os.path.join(dir_in,Mod,'historic','Results_Combined',
                                   'CDFmonthly_{0:02d}.pkl'.format(month)), 'rb') as f:
                cdf_cmipH = pickle.load(f)

             # Load the CMIP6 future data
            with open(os.path.join(dir_in,Mod,'future','Results_Combined',SLR,
                                   'CDFmonthly_{0:02d}.pkl'.format(month)), 'rb') as f:
                cdf_cmipF = pickle.load(f)


            # Calculate the difference between the historic and future.
            diff[:,:,cnt] = calc_diff(cdf_cmipH,cdf_cmipF,data_month)

        ds_diff = xr.DataArray(
            data = diff,
                 dims   = ['time','station','cmip6'],
                 coords = {'time': data_month['time'],
                           'station':data_month['station'],
                           'cmip6': Mod_list},
                 attrs  = {'long_name':'Cmip6 Difference in water levels',
                           'Date': f'year {year}, month {month}',
                           'units' : 'meters/10000'})

        ds_diff = ds_diff.chunk({'cmip6':1,'time':-1, 'station':1})

        # Post process some variables for output
        #wl_out = data_month['waterlevel'].fillna(-9999)
        #wl_out = wl_out.round(0).astype(dtype='int32')
        #wl_out = wl_out.chunk({'time':-1, 'station':1})
        station_id = data_month['station'].to_numpy().astype('unicode')

        #ds_full =  xr.Dataset({
        #    'wl':wl_out,
        #    'wl_quant':data_month['wl_quants'],
        #    'cmip_diff':ds_diff,
        #    'lon':data_month['lon'],
        #    'Lat':data_month['lat']})
        
        ds_full =  xr.Dataset({'cmip_diff':ds_diff})
        ds_full = ds_full.assign_coords({'station':station_id})

        ds_full.to_netcdf(os.path.join(dir_out,SLR,f'ERA5wl_Diff_{year}_{month:02d}.nc'), engine = 'netcdf4',
                              encoding={'cmip_diff':{'dtype': 'int32'}})

        #ds_full.to_netcdf(os.path.join(dir_out,SLR,f'ERA5wl_Diff_{year}_{month:02d}.nc'), engine = 'netcdf4',
        #                  encoding={'wl':{'dtype': 'int32'},
        #                            'wl_quant':{'dtype':'float32'},
        #                            'cmip_diff':{'dtype': 'int32'},
        #                            'Lon':{'dtype': 'float64'},
        #                            'Lat':{'dtype':'float64'}})


    client.shutdown()

if __name__ == '__main__':
    main()

