# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 10:51:34 2025

This script post-processes the Tmax_Zs product from sfincs.
It takes in the tmax_Zs product produced by the first 3 post-processing steps.
as well as the index which tracks which simulation year each return period cam from.

It then clusters the times when maximums occurs, assignes an integer identifier
and then tries to sort out when teh maximums occurs. 

It outputs a raster and shapefile.

@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"


# Import Modules
import os
import rasterio
import numpy as np
from hydromt_sfincs import SfincsModel
import pandas as pd
from sklearn.cluster import DBSCAN
import geopandas as gpd
import xarray as xr 
import rioxarray

#===============================================================================
# %% User Defined inputs
#===============================================================================

# Directories 
dir_base = r'C:\Users\kai\Documents\KaiRuns'


# User Options
RP_list = ['100']
SLR_list = ['000']
county_list = ['01_King']

year_base = 1941


#===============================================================================
# %% Define functions
#===============================================================================
def make_timestamp(date_string, year_base):
    import datetime
    
    # Case where read in as a datetime
    if isinstance(date_string,datetime.datetime):
        date_string = date_string.strftime(("%Y%m%d %H%M%S"))
   
    # Replace the year with the run year (per the folder). Or preceeding year. 
    date_string =str(int(date_string[0:4])+year_base)+date_string[4:]
   
    return pd.Timestamp(date_string)

def raster_to_shape_rasterio(raster_file, vector_file):
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape
    
    with rasterio.open(raster_file) as src:
        image = src.read(1)
        transform = src.transform
        results = shapes(image, transform=transform)
        
        geoms = []
        for geom, value in results:
            geoms.append({
                'geometry': shape(geom),
                'properties': {'ID': int(value)}
            })
        gdf = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)  
        
    gdf.to_file(vector_file)

def timestamp_to_seconds_since(timestamp,year_base):
    
    """Converts a pandas Timestamp to seconds since year_base-01-01 00:00:00."""
    reference_timestamp = pd.Timestamp(f'{year_base}-01-01 00:00:00')
    time_delta = timestamp - reference_timestamp
    seconds_since = time_delta.total_seconds()
    
    out = np.array(seconds_since).astype('int64')
    return out


def seconds_since_to_timestamp(seconds,year_base):
    
    """Converts a pandas Timestamp to seconds since year_base-01-01 00:00:00."""
    reference_timestamp = pd.Timestamp(f'{year_base}-01-01 00:00:00')
    date = reference_timestamp + pd.Timedelta(seconds,'s')
    
    return date

#===============================================================================
# %% Work through the processing lists
#===============================================================================

for county in county_list:
    for SLR in SLR_list:
        
        # Sort out directories
        dir_in          = os.path.join(dir_base,'PostProcess',county,SLR, )
        dir_out         = os.path.join(dir_base,'PostProcess',county,SLR,'Tmax_Zs')
        junk_folder     = os.path.join(dir_base, 'PostProcess',county,SLR,'Tmax_Zs','junk')
        
        for RP in RP_list:
            
            #===============================================================================
            # %% Load the data 
            #===============================================================================
            print(f'Processing county:{county},  SLR:{SLR},  RP:{RP}')

            file_in = os.path.join(dir_in,'downscaled_2m',f'tmax_zs_{RP}_2m_masked.tif')
            
            # Load raster
            data = rioxarray.open_rasterio(file_in)            
            t_max = data.values.squeeze()
            
            #Turn to integer
            t_max = np.nan_to_num(t_max, nan=-999)
            t_max = t_max.astype('int64')

            # Load the .nc to figure out the year.
            ds = xr.open_mfdataset(os.path.join(dir_in,f'processed_SFINCS_output_RP{RP}.nc'))
            year_max = ds['year_max'].values 
            
            
        
            #===============================================================================
            # %% Convert to calendar times
            #===============================================================================
            # Read the Year_max value
            year_max                    = ds['year_max']  # Assuming var_data is accessible this way
            year_max.attrs['crs']       = 'EPSG:6339'
            year_max.attrs['_FillValue']= -999
            
            # Project to subgrid resoultion (so matches output data)
            year_max                    = year_max.raster.reproject_like(data, method="nearest")  
            
            # get data as np.array
            year_max = year_max.values

            # Convert to integers
            year_max = np.nan_to_num(year_max, copy=True, nan=-999)
            year_max = year_max.astype('int64')
            
            # Mask by t_max 
            year_max[t_max==-999] = -999
            
            
            #Open file to get rasterio information.
            with rasterio.open(file_in) as src:
                src_transform = src.transform
                src_crs       = src.crs

            
            ####################### Write Raster (Year Max) ######################
            # Define writing
            kwargs2                 = dict(
                driver="GTiff",
                height=year_max.shape[0],
                width=year_max.shape[1],
                count=1,
                dtype='int64',
                crs=src_crs,  # Assuming 'src' is defined from previous operations
                transform=src_transform,  # Assuming 'src' is defined from previous operations
                tiled=True,
                blockxsize=128,                 # reduced this from 256
                blockysize=128,
                compress="deflate",
                predictor=2,                    # Adjust based on your data's nature (floating-point or integer)
                zlevel=6,                       # reduced to 6 from 9
                nodata=-999,
                profile="COG"
            )
            
            # Do actual writing
            file_out_Raster_storms = os.path.join(dir_out,f'YearMax_{RP}.tif')
            with rasterio.open(file_out_Raster_storms, 'w', **kwargs2) as dst:
                dst.write(year_max, 1)  # Write the first (and only) band
            
            # Build dataframe of simulation year specific information
            years = np.unique(year_max)
            years = years[years>0]
            
            modTimes = []
            for cnt,year in enumerate(years):
                
                # Open the Model configuration files 
                dir_sfincsMod = os.path.join(dir_base,county,SLR,f'SY{year:03d}')
                mod                     = SfincsModel(dir_sfincsMod, mode="r")
                mod.read_config(os.path.join(dir_sfincsMod,'sfincs.inp'))
                
                d = {'Year_ID': year,
                     'tref' : make_timestamp(mod.config['tref'],year_base), 
                    'tstart' : make_timestamp(mod.config['tstart'],year_base),
                    'tstop' :  make_timestamp(mod.config['tstop'],year_base)}
                
                modTimes.append(d)
                
            modTimes = pd.DataFrame(modTimes)
            modTimes = modTimes.set_index('Year_ID')
            
            # Go through the Year_max matrix and turn into seconds since reference data 
            # for each simulation year.   
            
            
            # Convert to seconds since refdate for sim year start 
            # So SecSince_year is seconds from reference data until start of the simulation year.
            print('Creating array of simulation year start dates')
            SecSince_year = np.full(year_max.shape,-999,dtype='int64')
            ind_real = np.argwhere(year_max>0)
            for ii in range(len(ind_real)):
                
                year_id = year_max[ind_real[ii,0],ind_real[ii,1]]
                SimStart = modTimes['tref'][year_id]
                
                
                SecSince_year[ind_real[ii,0],ind_real[ii,1]] = timestamp_to_seconds_since(SimStart,
                                                                                          year_base)
                
             
            
            max_date = SecSince_year + t_max
            max_date[max_date < 0 ] = -999
            
            #===============================================================================
            # %% cluster 
            #===============================================================================
        
            # Cluster the events
            max_distance = 4*(24*60*60)  # 4 Days
            
            in2clusters = max_date.flatten()
            in2clusters = in2clusters[in2clusters>=0]
            in2clusters = np.unique(in2clusters)
            in2clusters = in2clusters.reshape(-1, 1)
            
            clustering = DBSCAN(eps=max_distance, min_samples=1).fit(in2clusters)
            
            labels = clustering.labels_
            
            # Number of clusters in labels, ignoring noise if present.
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            print(f"Number of Event clusters: {n_clusters}")
            
            print('Labeling matrix')
            data_labeled = np.full(max_date.shape,-1,dtype = 'int16')
            for cnt,val in enumerate(in2clusters):
                data_labeled[max_date==val] = labels[cnt].astype('int16')
            
            
            # Figure out the timestamp
            events = np.unique(labels)
            events = events[events >= 0]
            t_ave = np.full(events.shape,np.nan)
            t_storm = []
            t_storm_dt = []
            for cnt,event in enumerate(events):
                ind = (labels == event)
                t_ave = np.nanmean(in2clusters[ind])
                t_storm.append(seconds_since_to_timestamp(t_ave,year_base).round(freq='min'))
                t_storm_dt.append(in2clusters[ind].max() - in2clusters[ind].min())
            d = {'time':t_storm,
                 'duration':t_storm_dt}
            Storms = pd.DataFrame(data=d,index=events)
            
            
            #===============================================================================
            # %% Write Data
            #===============================================================================
            print('Outputing data')
            
            #Open file to get rasterio information.
            with rasterio.open(file_in) as src:
                src_transform = src.transform
                src_crs       = src.crs

            
            ####################### Write Raster (Storm Groups) ######################
            # Define writing
            kwargs2                 = dict(
                driver="GTiff",
                height=data_labeled.shape[0],
                width=data_labeled.shape[1],
                count=1,
                dtype='int16',
                crs=src_crs,  # Assuming 'src' is defined from previous operations
                transform=src_transform,  # Assuming 'src' is defined from previous operations
                tiled=True,
                blockxsize=128,                 # reduced this from 256
                blockysize=128,
                compress="deflate",
                predictor=2,                    # Adjust based on your data's nature (floating-point or integer)
                zlevel=6,                       # reduced to 6 from 9
                profile="COG"
            )
            
            # Do actual writing
            file_out_Raster_storms = os.path.join(dir_out,f'tmax_zs_{RP}_Storms.tif')
            with rasterio.open(file_out_Raster_storms, 'w', **kwargs2) as dst:
                dst.write(data_labeled, 1)  # Write the first (and only) band
            
            
            
            ####################### Write Raster (Seconds Since Ref.Date) ######################
            # Define writing
            kwargs2                 = dict(
                driver="GTiff",
                height=max_date.shape[0],
                width=max_date.shape[1],
                count=1,
                dtype='int64',
                crs=src_crs,  # Assuming 'src' is defined from previous operations
                transform=src_transform,  # Assuming 'src' is defined from previous operations
                tiled=True,
                blockxsize=128,                 # reduced this from 256
                blockysize=128,
                compress="deflate",
                predictor=2,                    # Adjust based on your data's nature (floating-point or integer)
                zlevel=6,                       # reduced to 6 from 9
                profile="COG"
            )    

            # Do actual writing
            file_out_Raster_Secs = os.path.join(dir_out,f'tmax_zs_{RP}_SecSinceRef.tif')
            with rasterio.open(file_out_Raster_Secs, 'w', **kwargs2) as dst:
                dst.write(data_labeled, 1)  # Write the first (and only) band
            
            
      
            ####################### Write Shapefile ######################
            
            # First write a shapefile with no metadata
            file_out =  os.path.join(junk_folder,f'tmax_zs_{RP}_Storms.shp')
            print(f'Writing Intermediate shapefile from raster: {file_out}')
            raster_to_shape_rasterio(file_out_Raster_storms, file_out)
            
            # Then write a shapefile with dates added 
            
            #### Add attributes to the file ######
            Storm_data = gpd.read_file(file_out)
            
            file_out = os.path.join(dir_out,f'tmax_zs_{RP}_Storms.shp')
            print(f'Writing Final shapefile from raster: {file_out}')
            
            #Storm_data = Storm_data.rename({'ID':'StormID'})
            # Remove the no data case
            Storm_data = Storm_data.loc[(Storm_data['ID'] != -1)]
            
            
            dates = np.full(Storm_data['ID'].shape,' ',dtype='U20')
            for ID_sel in np.unique(Storm_data['ID']):
                dates[(Storm_data['ID'] == ID_sel)] = Storms.loc[ID_sel].time.strftime('%Y-%m-%d %H:%M:%S')
            
            Storm_data['Dates'] = dates
             
            Storm_data.to_file(file_out)




                                  