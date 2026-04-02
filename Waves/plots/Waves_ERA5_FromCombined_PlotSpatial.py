# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:52:38 2024

This script Plots the spatial variablity in CMIP6 Diff
 
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
import matplotlib
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import scipy.stats as stats
import pyextremes
import scipy.optimize as optimize
import scipy

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory where the DFM data resides
#dir_in = r'D:\DFM'
dir_in = r'D:\Combined_DFM\ERA5'
dir_out = r'Y:\PS_Cosmos\GIS\Waves\ERA5'
dir_out_fig = r'Y:\PS_Cosmos\Figures\Waves'

#SLR_list =['000','025','050','100','150','200','300']
SLR ='000'

# Metric to plot
metric = 'RP_50'   


#===============================================================================
# %% Define some functions
#===============================================================================


def CalcStats(files):
    # Data: Xarray dataset 
    
    stats = np.full((len(files),11),np.nan)
    lon   = np.full((len(files)),np.nan)
    lat   = np.full((len(files)),np.nan)
    for cnt,file in enumerate(files):
        ds  = xr.open_mfdataset(file, engine='h5netcdf', parallel=True,
                                decode_timedelta =False)

        # Load TWL values into memeory as a pandas series
        data = pd.Series( ds['Hs'].values,
                        index= ds['time'].values)
        
        print(file)
        # only process if more than 60% the record is real values (not nans)
        if (data.isna().sum() < data.shape[0]*0.5):
            
            try:
                # Find extremes
                 
                # Get number of exceedances to grab
                num_years = np.unique(data.index.year).shape[0]
                Npryr  = 1
                Num_Exce = (num_years -1) * Npryr
                
                selected_thresh = POT_theshold_SetNum(
                    data,          # dataset
                    'Hs',              # column name of pd series
                    '72h',              # independence time delta r
                    Num_Exce)           # number of exceedences we want
                
                extremes = pyextremes.extremes.get_extremes(
                    data,
                    method="POT",
                    extremes_type='high',
                    threshold=selected_thresh,
                    r='72h')
            
                return_periods = pyextremes.get_return_periods(
                    ts=pd.DataFrame(data),
                    extremes=extremes,
                    extremes_method="POT",
                    extremes_type="high",
                    return_period_size="365.2425D",
                    plotting_position="weibull")
                

                return_periods.sort_values("return period", ascending=True,inplace=True)
                
                
                interp_rp = scipy.interpolate.interp1d(return_periods['return period'].to_numpy(),
                                                    return_periods['extreme values'].to_numpy(),
                                                    copy=False,
                                                    assume_sorted=True,
                                                    fill_value=(return_periods['extreme values'].min(),np.nan),
                                                    bounds_error=False) 
                RIs = interp_rp([1,10,25,50,80])


                # Turn into a dataframe
                d = {'Mean': np.nanmean(data, axis=0),
                    'Max': np.nanmax(data, axis=0),
                    'Std': np.nanstd(data, axis=0),
                    'Q99': np.nanquantile(data,.99, axis=0),
                    'Q95': np.nanquantile(data,.95, axis=0),
                    'RI_1': RIs[0],
                    'RP_10': RIs[1],
                    'RP_25': RIs[2],
                    'RP_50': RIs[3],
                    'RP_80': RIs[4],
                    'NumNan': np.count_nonzero(np.isnan(data),axis=0)}


                d_array = np.array(list(d.values())) 

                stats[cnt,:] = d_array


            except:
                print(f'{file} has Maximum finding issues')
                                
        lon[cnt] = ds['lon_wl'].values
        lat[cnt] = ds['lat_wl'].values
                
        

    # Convert to pandas dataframe
    stats_out = pd.DataFrame(data= stats, columns=['Mean','Max','Std','Q99','Q95','RI_1','RP_10','RP_25','RP_50','RP_80','NumNan'])
        
    # add geometry and turn into geopandas dataset
    geometry = gpd.points_from_xy(x=lon,
                                  y=lat,
                                  crs = 'EPSG:4326')
    
    out = gpd.GeoDataFrame(data =stats_out, geometry=geometry)
    
    return out



# Use a function minimizer to figure out the actual threshold we want
def threshold_min_fun(thresh,Num_Exce,filt,data):
    from pypot.threshold_selection import get_extremes_peaks_over_threshold

    pks = get_extremes_peaks_over_threshold(data, thresh, r=filt)
    
    # Number of Peaks for this threshold
    num_peaks = pks.shape[0]

    # What is the difference between this and the amount we want
    # This is the function we are trying to minimize to zero
    Diff = np.absolute(Num_Exce-num_peaks)

    return Diff


def POT_theshold_SetNum(data,y_lab,r,Num_Exce):
    """ select threshold for PoT analysis
    using a set number of exceedences.

    args:
        data (pd.DataFrame): data
        y_lab: (str) label of y column        
        r (str): time delta string to define independence
        Num_Exce (np.float): number of exceedences we want

    returns:
        threshold (np.float)
    """

    if isinstance(data,pd.Series) or isinstance(data,pd.DataFrame):
    
        if isinstance(data,pd.Series):
            data = data.to_frame(name=y_lab) 
        
    else:
        print(f'Unrecognized input data type {type(data)}')
        print('Input must be a pandas dataframe or series')

    # Guess for the threshold
    B = data.sort_values(by=y_lab,na_position='last',ascending=False)
    thresh_guess = B[y_lab].iloc[Num_Exce]
        
    # Minimize
    bnds = optimize.Bounds(lb =B[y_lab].mean(), ub = B[y_lab].iloc[1])
    Optim_out = optimize.minimize(threshold_min_fun,
                                       x0 = thresh_guess,
                                       args=(Num_Exce,r,data[y_lab]),
                                       bounds =bnds,
                                       method = 'Nelder-Mead')
    
    
    if Optim_out['success']:
        threshold = Optim_out['x']
    else:
        print('Something went wrong!')
    
    return threshold


#===============================================================================
# %% Load the  data
#===============================================================================

# Load state shapefiles.
cnty = gpd.read_file(r'Y:\PS_Cosmos\GIS\general\Washington_Counties_with_Natural_Shoreline___washsh_area.shp')

# Load in the landmass file
lm = gpd.read_file(r"Y:\PS_Cosmos\GIS\general\PoliticalBoundaries_Shapefile\NA_PoliticalDivisions\data\bound_p\boundaries_p_2021_v3.shp")



files = glob(os.path.join(dir_in,'*.nc'))
data = CalcStats(files)


# Save as a shapefile. 
data.to_file(os.path.join(dir_out,f'Wave_Hs_ERA5Stats_{SLR}.shp'))

#===============================================================================
# %% Load the  data
#===============================================================================
lm = lm.to_crs(crs=data.crs)


ax1 = lm.plot(figsize=(10, 10), color='0.8', edgecolor='black', alpha=0.4)
s1 = data.plot(ax=ax1,column=metric, marker='.', markersize=10,
                    legend=True, cmap='OrRd', legend_kwds={"label": "Hs (m)"},
                    label=f'{metric} Hs')

ax1.set_xlim([-124.75,-122])
ax1.set_ylim([47,49.5])
ax1.grid()
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latittude')
ax1.set_title(f'ERA5 Modelled {metric} Hs')
ax1.legend()

fig =ax1.get_figure()
fig.savefig(os.path.join(dir_out_fig,f'ERAf_Map_{metric}.tiff'), dpi=300)


