# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:28:01 2023


This script Load Precip. data for CMIP6 and ERA5 regions and then compares to
see if there is any bias. 

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
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
import matplotlib
import datetime
import sys
import pandas as pd
import geopandas as gpd
from glob import glob

#===============================================================================
# %% Define some functions
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



def wrapTo360(lon):
    lon_360 = lon % 360
    return lon_360

def wrapTo180(lon):
    lon_180 = (lon % 360 + 540) % 360 - 180
    return lon_180



def to_datetime(d):
    import datetime as dt
    import cftime
    import pandas as pd
    
    if isinstance(d, dt.datetime):
        return d
    if isinstance(d, cftime.DatetimeNoLeap):
        return dt.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    elif isinstance(d, cftime.DatetimeGregorian):
        return dt.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    elif isinstance(d, cftime.DatetimeJulian):
        return dt.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    elif isinstance(d, cftime.Datetime360Day):
        return dt.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    elif isinstance(d, str):
        errors = []
        for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return dt.datetime.strptime(d, fmt)
            except ValueError as e:
                errors.append(e)
                continue
        raise Exception(errors)
    elif isinstance(d, np.datetime64):
        temp = d.astype(dt.datetime)
        if isinstance(temp,dt.datetime):
            out = temp
        elif isinstance(temp,int):
            out = pd.Timestamp(d).to_pydatetime()
        return out
    else:
        raise Exception("Unknown value: {} type: {}".format(d, type(d)))


def interp_datetime(date_q,date,data):
    import calendar
   
    def toTimestamp(t):
        out = [calendar.timegm(val.timetuple()) for val in t]
        out = np.array(out)
        return out
  

    result = np.interp(toTimestamp(date_q),toTimestamp(date),data)
    return result
    

def Watershed_Extract(dataset,huc):
    
    # Extracts data within a watershed 
    # Huc is a polygon defining the huc watershed
    # dataset is an xarray.DataArray  (can't be a Xarray.dataset)  
    # Extracts and then averages for the points in the watershed
    Coord_names = np.array(dataset.coords.dims)
    
    ###################### Pull out the location info ####################
    # Assumes format of lat and lon coordinate is the same 
    if sum(Coord_names == 'longitude') == 1:
        lat  = dataset['latitude'].values 
        lon  = dataset['longitude'].values 
        
    elif sum(Coord_names == 'lon') == 1:
        lat  = dataset['lat'].values 
        lon  = dataset['lon'].values 
        
    elif sum(Coord_names == 'Lon') == 1:
        lat  = dataset['Lat'].values 
        lon  = dataset['Lon'].values 
        
    elif sum(Coord_names == 'Longitude') == 1:
        lat  = dataset['Latitude'].values 
        lon  = dataset['Longitude'].values 
        
    else: 
        print('Couldnt parse coordinates')
        sys.exit 
    
    # Deal with wrapping
    wrap_switch = 0
    if np.max(lon) > 180:
        wrap_switch = 1
        lon  = wrapTo180(lon)
    
    # Mesh so can create a full matrix of values 
    lon, lat = np.meshgrid(lon, lat, indexing='xy')
    
    data_loc = gpd.GeoSeries(gpd.points_from_xy(x = lon.flatten(), y = lat.flatten()),crs = "EPSG:4326") 
    data_loc = gpd.GeoDataFrame(geometry=data_loc)
    
    
    #### Pull out coordinates for dataset locations within the huc polygon ####
    temp = gpd.sjoin(huc,data_loc,how='left')
    lon_pull = data_loc['geometry'][temp['index_right']].x.values
    lat_pull = data_loc['geometry'][temp['index_right']].y.values
    
    if wrap_switch == 1:
        lon_pull = wrapTo360(lon_pull)
    
    # pull data for these locations
    data_pull = np.empty([dataset['time'].shape[0],lon_pull.shape[0]])
    for ii in range(lon_pull.shape[0]):
        
        # Assumes format of lat and lon coordinate is the same 
        # Add method nearest because the wrapto180 and wrapto 360 loose precision
        if sum(Coord_names == 'longitude') == 1:
            data_pull[:,ii] = dataset.sel(longitude = lon_pull[ii],latitude=lat_pull[ii],method='nearest').values
        elif sum(Coord_names == 'lon') == 1:
            data_pull[:,ii] = dataset.sel(lon = lon_pull[ii],lat=lat_pull[ii],method='nearest').values
        elif sum(Coord_names == 'Lon') == 1:
            data_pull[:,ii] = dataset.sel(Lon = lon_pull[ii],Lat=lat_pull[ii],method='nearest').values
        elif sum(Coord_names == 'Longitude') == 1:
            data_pull[:,ii] = dataset.sel(Longitude = lon_pull[ii],Latitude=lat_pull[ii],method='nearest').values
        else: 
            print('Couldnt parse coordinates')
            sys.exit 
        
    # Average acroos all points within the polygon
    data_out = np.mean(data_pull,axis=1)
        
    return data_out
    
def Average_quarterly(data,time):
    # Extract quarterly data
    # created using xarray and  Dataset.resample(time='QS-DEC').mean(dim="time"))
    #time is a pandas datatime index 
    
    ind_q0 = time.month == 12
    ind_q1 = time.month == 3
    ind_q2 = time.month == 6
    ind_q3 = time.month == 9
    
    DataOut_AveQtr    = np.empty([4])
    DataOut_AveQtr[0] = np.mean(data[ind_q0])
    DataOut_AveQtr[1] = np.mean(data[ind_q1])
    DataOut_AveQtr[2] = np.mean(data[ind_q2])
    DataOut_AveQtr[3] = np.mean(data[ind_q3])

    return DataOut_AveQtr

def Extract_Monthly(data,time):
    # Extract Monthly data
    # created using xarray and  Dataset.resample(time='MS').mean(dim="time"))
    #time is a pandas datatime index 
    
    DataOut_0  = data[time.month == 1]
    DataOut_1  = data[time.month == 2]
    DataOut_2  = data[time.month == 3]
    DataOut_3  = data[time.month == 4]
    DataOut_4  = data[time.month == 5]
    DataOut_5  = data[time.month == 6]
    DataOut_6  = data[time.month == 7]
    DataOut_7  = data[time.month == 8]
    DataOut_8  = data[time.month == 9]
    DataOut_9  = data[time.month == 10]
    DataOut_10 = data[time.month == 11]
    DataOut_11 = data[time.month == 12]

    return DataOut_0,DataOut_1,DataOut_2,DataOut_3, \
        DataOut_4,DataOut_5,DataOut_6,DataOut_7,\
        DataOut_8,DataOut_9,DataOut_10,DataOut_11


#===============================================================================
# %% User Defined inputs
#===============================================================================

#### Directories #####
dir_era5  = r'Y:\PS_Cosmos\ERA5'
dir_cmip6 = r'Y:\PS_Cosmos\CMIP6' 
dir_wbd   = r'Y:\PS_Cosmos\GIS\Hydrology\WBD_17_HU2_Shape\Shape'

dir_out  = r'C:\Users\kaparker\Documents\Projects\PS_Cosmos\figures\Precip'

# CMIP6 models
mod = ['CMCC-CM2-VHR4','CNRM-CM6-1-HR','EC-Earth_HR','GFDL',
       'HadGEM_GC31_HH','HadGEM_GC31_HM_highRes','HadGEM_GC31_HM_highResSST']

# Watersheds to aggregate precip to
# selected by importing shapefile into arc and choosing names for area of interest.
#huc_names = ['Puyallup','Nisqually','Duwamish','Lake Washington','Snoqualmie']
huc_name  = 'Duwamish'

# Variable 
Var = 'temp'   # 'pr' 'temp'

#===============================================================================
# %% Figure out name shortening
#===============================================================================
mod_short = [i for i in range(len(mod))]
for cnt,model in enumerate(mod):
    if model == 'CMCC-CM2-VHR4':
        mod_short[cnt] = 'CMCC'
        
    elif model == 'CNRM-CM6-1-HR':
        mod_short[cnt] = 'CNRM'
        
    elif model == 'EC-Earth_HR':
        mod_short[cnt] = 'EcEarth'
    
    elif model == 'GFDL':
        mod_short[cnt] = 'GFDL'
    
    elif model == 'HadGEM_GC31_HH':
        mod_short[cnt] = 'HadGemHH'
    
    elif model == 'HadGEM_GC31_HM_highRes':
        mod_short[cnt] = 'HadGemHM'
    
    elif model == 'HadGEM_GC31_HM_highResSST':
        mod_short[cnt] = 'HadGemHMsst'
    
    else:
         print('Model Choice not found')   
         sys.exit()


#===============================================================================
# %% Load the WBD  Data
#===============================================================================
 
# load the WBD
WBDHU8 = gpd.read_file(os.path.join(dir_wbd,'WBDHU8.shp'))

# project to utm for analysis
WBDHU8 = WBDHU8.to_crs('EPSG:4326')

huc = WBDHU8.loc[WBDHU8['name']==huc_name]

# add a small buffer to grab points who's cells are in the HUC but centroid isn't
huc = gpd.GeoDataFrame(geometry= huc.buffer(.1))  # .1 degrees is approximately 11 kilometers (~half a grid cell)

#===============================================================================
# %% Load the ERA5 Data
#===============================================================================
print('Processing ERA5')

if Var == 'pr':  # Precipitation

    # ERA5 files to read and Open all
    files = glob(os.path.join(dir_era5,'Precipitation','*.nc'))
    ds_era5 = xr.open_mfdataset(files, engine='netcdf4', parallel=True)
    
    ds_era5 = ds_era5['mtpr']

elif Var == 'temp': # Temperature

    # ERA5 files to read and Open all
    files = glob(os.path.join(dir_era5,'Download','*.nc'))
    ds_era5 = xr.open_mfdataset(files, engine='netcdf4', parallel=True)
    
    ds_era5 = ds_era5['t2m']

# Pull out the time variable
time = ds_era5['time'].values
    
# Convert to a datetime variable
date = [to_datetime(val) for val in time]
time = np.array(date)

# Cut to only time period that overlaps with CMIP6 historic period
ds_era5 = ds_era5.sel(time=slice("1950-01-01", "2015-01-01"))

# Process the watershed scale precip

# resample precip to quarterly
ds_era5_seasonal = ds_era5.resample(time='MS').mean(dim="time")

# Extract for watershed
ds_era5_huc = Watershed_Extract(ds_era5_seasonal,huc)

if Var == 'pr':  # Precipitation
    # Convert to mm/hr
    # kg        m3   60 s   60 min    1000 mm
    # m2*s   1000kg   min    hr       1m
    conv  = (60*60*1000)/(1000)
    ds_era5_huc = ds_era5_huc*conv    

if Var == 'temp':  # Precipitation
    # Convert from Kelvin to Celcius
    ds_era5_huc = ds_era5_huc- 273.15  


time_pds = ds_era5_seasonal.indexes['time']

pr_q0,pr_q1,pr_q2,pr_q3,pr_q4,pr_q5,pr_q6,pr_q7,pr_q8,pr_q9,pr_q10,pr_q11 = \
    Extract_Monthly(ds_era5_huc,time_pds)

pr_means = [np.mean(pr_q0),np.mean(pr_q1),np.mean(pr_q2),np.mean(pr_q3), \
            np.mean(pr_q4),np.mean(pr_q5),np.mean(pr_q6),np.mean(pr_q7), \
            np.mean(pr_q8),np.mean(pr_q9),np.mean(pr_q10),np.mean(pr_q11)]

fig, ax1 = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(8, 5))
ax1.plot(np.arange(1,13,1,dtype=int),pr_means,'-ko')

#===============================================================================
# %% Load the Cmip6 Data
#===============================================================================
# Convert to m of precipitation per hour
# ERA5 is hourly
#  kg   1000 g   1 cm3     1   m3    60 s  60 min  |  m   
#  m2s  1 kg     1 g     100^3 cm3   1 min  1 hr   |  hr

#conv = (1000*60*60)/(100^3)
#ds_cmip = ds_cmip*conv
# Now looking at precipation flux rather than precip/hour.

precip_cmip_ave_qtr = np.empty([len(mod_short),4],dtype = 'float64')
for cnt,model in enumerate(mod_short):
    print('Processing Cmip6: {}'.format(model))
    
    if Var == 'pr':  # Precipitation
        file_in = os.path.join(dir_cmip6,'Precipitation','historic','psfm_{}_historic_precip.nc'.format(model))
        ds_cmip = xr.open_mfdataset(file_in, engine='netcdf4')
        ds_cmip = ds_cmip.to_array()
        
    elif Var == 'temp': # Temperature
        file_in = os.path.join(dir_cmip6,'Temperature','historic','psfm_{}_historic_temp.nc'.format(model))
        ds_cmip = xr.open_mfdataset(file_in, engine='netcdf4')
        ds_cmip = ds_cmip.to_array()
    
    # Process the watershed scale precip
    ds_cmip_seasonal = ds_cmip.resample(time='MS').mean(dim="time")
    ds_cmip_huc = Watershed_Extract(ds_cmip_seasonal,huc)
    
    if Var == 'pr':  # Precipitation
        # Convert to mm/hr
        conv  = (60*60*1000)/(1000)
        ds_cmip_huc = ds_cmip_huc*conv
        
    if Var == 'temp':  # Precipitation
        # Convert from Kelvin to Celcius
        ds_cmip_huc = ds_cmip_huc- 273.15  
    
    
    date = [to_datetime(val) for val in ds_cmip_seasonal.indexes['time'].values]
    time_pds = pd.DatetimeIndex(date) 
        
    pr_q0,pr_q1,pr_q2,pr_q3,pr_q4,pr_q5,pr_q6,pr_q7,pr_q8,pr_q9,pr_q10,pr_q11 = \
        Extract_Monthly(ds_cmip_huc,time_pds)
    
    pr_means = [np.mean(pr_q0),np.mean(pr_q1),np.mean(pr_q2),np.mean(pr_q3), \
                np.mean(pr_q4),np.mean(pr_q5),np.mean(pr_q6),np.mean(pr_q7), \
                np.mean(pr_q8),np.mean(pr_q9),np.mean(pr_q10),np.mean(pr_q11)]
    
    ax1.plot(np.arange(1,13,1,dtype=int),pr_means,'-o')

ax1.grid()

ax1.legend(['ERA5','CMCC','CNRM','ECEarth','GFDL','HadGemHH','HadGemHM','HadGemSST'])

ax1.set_xticks(np.arange(1,13,1,dtype=int))
ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], \
                    rotation = 45, ha="right")

if Var == 'pr':
    ax1.set_title('Monthly Precip')
    ax1.set_ylabel('Precip Flux (mm/hr)')
    fig.savefig(os.path.join(dir_out,'Precip_Compare_Monthly{}.png'.format(huc_name)),dpi = 400)

elif Var == 'temp':
    ax1.set_title('Monthly Temperature')
    ax1.set_ylabel('Temp (deg. C)')
    fig.savefig(os.path.join(dir_out,'Temp_Compare_Monthly{}.png'.format(huc_name)),dpi = 400)   
    
    