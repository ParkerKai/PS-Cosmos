# -*- coding: utf-8 -*-
"""
Created on Mon Jan  13:14:39 2024

Process Kitsap streamgauge data

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
import scipy 
import datetime
import dataretrieval.nwis as nwis

#===============================================================================
# %% User Defined inputs
#===============================================================================

dir_in = r'Y:\PS_Cosmos\01_data\Hydrology\Kitsap_Gauges\KisapPlus'
dir_out = r'Y:\PS_Cosmos\01_data\Hydrology\Kitsap_Gauges'

#===============================================================================
# %% Define some functions
#===============================================================================
def get_stationID(file_name):
    parts = name.split("_")
    
    # Use list comprehension to separate alphabets and numbers
    alphabets_list = [char for char in parts[0] if char.isalpha()]
    station = ''.join(alphabets_list)
    
    return station 


#===============================================================================
# %% Load the usgs site
#===============================================================================

# specify the USGS site code for which we want data.
site = '12073500'

# get instantaneous values (iv)
df = nwis.get_record(sites=site, parameterCd='00060', start='1989-10-01', end='2025-01-22', access='3')

# get basic info about the site
site_info = nwis.get_record(sites=site, service='site')



data_vars = {'Q':(['time','station'],  np.expand_dims(df['00060'].values, axis=1), 
                         {'units': 'cubic feet per second (CFS)', 
                          'standard_name':'Discharge'}),
             'Lat':(['station'],site_info['dec_lat_va'], 
                             {'long_name':'Latitude',
                              'Datum':'NAD83'}),
             'lon':(['station'],site_info['dec_long_va'], 
                             {'long_name':'Longitude',
                              'Datum':'NAD83'}),
             }

# define coordinates
coords= { 'station':(['station'], site_info['site_no'].values, 
                         {'long_name':'Site Code'}),
             'time':(['time'], df.index, 
                             {'standard_name':'time',
                              'TimeZone':'?'})}

# define global attributes
attrs = dict(
    processing_date=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
    description= 'Streamflow data from USGS',
    dataSource = 'NWIS retrieved with dataretrieval',
    author='Kai Parker: kaparker@usgs.gov')

ds_out = xr.Dataset(data_vars=data_vars, 
                coords=coords, 
                attrs=attrs)

ds_out.to_netcdf(os.path.join(dir_out,f"USGS_{site_info['site_no'].values}.nc"))


#===============================================================================
# %% Load the datastation information
#===============================================================================    
files = glob(os.path.join(dir_in,'Hydrology_*'))


Discharge = []
site_code = np.full(len(files),'  ',dtype='U3')
Lat       = np.full(len(files),np.nan)
Lon       = np.full(len(files),np.nan)
for cnt,file in enumerate(files):
    name = os.path.basename(file)
    print(f'Processing: {name}')
    
    
    data =  pd.read_csv(os.path.join(dir_in,file),index_col='Collect Date (UTC)',
                          dtype={'Site_Code': str,
                                 'Collect Date (local)': str,
                                 'Stage (ft)': np.float32,
                                 'Discharge (cfs)': np.float32,
                                 'E = Estimate; W = Warning; P = Provisional': str},
                          on_bad_lines = 'warn',
                          na_values=[" ", "e"])
    
    site_code[cnt] = data['Site_Code'].iloc[0]
    data = data['Discharge (cfs)']
    
    # Get location information
    # Found here: https://green2.kingcounty.gov/hydrology/GaugeTextSearch.aspx
    if site_code[cnt] == '28a':
        Lat[cnt] = 47.4034
        Lon[cnt] = -122.46882
        
    elif site_code[cnt] == '65A':
        Lat[cnt] = 47.3345
        Lon[cnt] = -122.5089
        
    elif site_code[cnt] == '65B':
        Lat[cnt] = 47.3841
        Lon[cnt] = -122.4815
        
    elif site_code[cnt] == '43a':
        Lat[cnt] = 47.4783	
        Lon[cnt] = -122.48173
        
    elif site_code[cnt] == '17B':
        Lat[cnt] = 47.45368	
        Lon[cnt] = -122.44465	 
        
    elif site_code[cnt] == '36A':
        Lat[cnt] = 47.3883		
        Lon[cnt] = -122.4274	
    else:
        print('Station Lat Lon not found')
        
        	
    Discharge.append(data)

Q   = pd.concat(Discharge, axis = 1)


#===============================================================================
# %% Turn into a netcdf
#===============================================================================    

# define data with variable attributes
data_vars = {'Q':(['time','station'],  Q, 
                         {'units': 'cubic feet per second (CFS)', 
                          'standard_name':'Discharge'}),
             'Lat':(['station'],Lat, 
                             {'long_name':'Latitude'}),
             'lon':(['station'],Lon, 
                             {'long_name':'Longitude'}),
             }

# define coordinates
coords= { 'station':(['station'], site_code, 
                         {'long_name':'Site Code'}),
             'time':(['time'],  Q.index, 
                             {'standard_name':'time',
                              'TimeZone':'UTC'})}

# define global attributes
attrs = dict(
    processing_date=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
    description= 'Streamflow data for Vashon Island (King County)',
    dataSource = 'https://green2.kingcounty.gov/hydrology/',
    author='Kai Parker: kaparker@usgs.gov')

# create dataset
ds_out = xr.Dataset(data_vars=data_vars, 
                coords=coords, 
                attrs=attrs)

ds_out.to_netcdf(os.path.join(dir_out,'Vashon_gauges.nc'))





