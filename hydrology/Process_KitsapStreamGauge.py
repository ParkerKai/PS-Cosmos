# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:14:39 2024

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

#===============================================================================
# %% User Defined inputs
#===============================================================================

dir_in = r'Y:\PS_Cosmos\01_data\Hydrology\Kitsap_Gauges'
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
# %% Load the datastation information
#===============================================================================    

stat = pd.read_excel(os.path.join(dir_in,'KPUD station list.xlsx'))

files = glob(os.path.join(dir_in,'Gauges','*.xlsx'))

Observation = []
Staff = []
Discharge = []
Stat_info = []
for file in files:
    name = os.path.basename(file)
    print(f'Processing: {name}')
    
    data =  pd.read_excel(os.path.join(dir_in,'Gauges',file),index_col='Date',
                          dtype={'Instrument_Reading': np.float32,
                                 'Staff_ft': np.float32,
                                 'Discharge_cfs': np.float32,
                                 'Estimated': np.float32},
                          na_values=["*.**", " ", "e"])
    

    
    # Get station name
    stat_name  = get_stationID(name)
    
    info = stat[stat['SITE_CODE'] == stat_name]
    if info.size == 0:
        print(f'No Station information found for: {name}')
    
    else:
        Observation.append(data['Instrument_Reading'])
        Staff.append(data['Staff_ft'])
        Discharge.append(data['Discharge_cfs'])
        Stat_info.append(info)

# Concatenate into datasets 
Obs = pd.concat(Observation, axis = 1)
Stf = pd.concat(Staff, axis = 1)
Q   = pd.concat(Discharge, axis = 1)
Stat = pd.concat(Stat_info, axis = 0)


#===============================================================================
# %% Turn into a netcdf
#===============================================================================    

# define data with variable attributes
data_vars = {'Obs':(['time','station'], Obs, 
                         {'units': '?',
                          'long_name':'Instrument_Reading'}),
            'Stf':(['time','station'], Stf, 
                         {'units': 'feet', 
                          'long_name':'Staff'}),
            'Q':(['time','station'],  Q, 
                         {'units': 'cubic feet per second (CFS)', 
                          'standard_name':'Discharge'}),
             'Name':(['station'],Stat['SITE_NAME'], 
                             {'long_name':'SITE_NAME'}),
             'Lat':(['station'],Stat['Latitude'], 
                             {'long_name':'Latitude'}),
             'lon':(['station'],Stat['Longitude'], 
                             {'long_name':'Longitude'}),
             'elev':(['station'],Stat['ELEVATION'], 
                            {'long_name':'Longitude'}),
             'Tag':(['station'],Stat['GAGETAG'], 
                             {'long_name':'GAUGETAG'}),
             'Status':(['station'],Stat['STATUS'], 
                             {'long_name':'STATUS'}),
             'Date_Start':(['station'],Stat['DATE_INSTA'].values.astype('U30'), 
                             {'long_name':'DATE_INSTA'}),
             'Date_End':(['station'],Stat['DATE_REMOV'].values.astype('U30'), 
                             {'long_name':'DATE_REMOV'}),
             'Comment':(['station'],Stat['COMMENT'], 
                             {'long_name':'COMMENT'}),
             'DrnArea':(['station'],Stat['DrnArea'], 
                             {'long_name':'DrainageArea'}),
             }

# define coordinates
coords= { 'station':(['station'], Stat['SITE_CODE'], 
                         {'long_name':'Site Code'}),
             'time':(['time'],  Q.index, 
                             {'standard_name':'time'})}

# define global attributes
attrs = dict(
    processing_date=datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
    description= 'Streamflow data for Kitsap county processed from KPUDs database',
    dataSource = 'http://kpudhydrodata.kpud.org/RainMap.html',
    citation = 'Kitsap Public Utility District, 2024, KPUD Kitsap County Water Resources Precipitation and Streamflow database,  Rainfall & Streamflow Data – Kitsap Public Utility District',
    contact = 'Joel Purdy: jpurdy@kpud.org',
    author='Kai Parker: kaparker@usgs.gov')

# create dataset
ds_out = xr.Dataset(data_vars=data_vars, 
                coords=coords, 
                attrs=attrs)

ds_out.to_netcdf(os.path.join(dir_in,'Kitsap_gauges.nc'))

