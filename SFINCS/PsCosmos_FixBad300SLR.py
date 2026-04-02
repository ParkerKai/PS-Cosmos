# -*- coding: utf-8 -*-
"""
Created on Wed May 14 09:40:31 2025

@author: kai
"""

import xarray as xr
import os 
import matplotlib 

dir_in = r'C:\Users\kai\Documents\KaiRuns\20250422_tideonly_MHHW\02_Pierce'
dir_out = r'C:\Users\kai\Documents\KaiRuns\20250422_tideonly_MHHW_New300\02_Pierce'
Scenario = '300_high'

data = xr.open_mfdataset(os.path.join(dir_in,Scenario,'SY1940','sfincs_bndbzs.nc'))


data_new = data.copy(deep=True) 
data_new['zs'] = data['zs']+3


file_out = os.path.join(dir_out,Scenario,'SY1940','sfincs_bndbzs.nc')

data_new.to_netcdf(file_out)

