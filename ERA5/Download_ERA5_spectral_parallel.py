# -*- coding: utf-8 -*-
"""
Created on Oct 13th 12:17:00 2024

based on: https://confluence.ecmwf.int/display/CEMS/EWDS+-+Best+Practices


@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"

# Import Modules
import time 
import os
import cdsapi
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime 
import warnings
warnings.filterwarnings("ignore")

#===============================================================================
# %% User Defined inputs
#===============================================================================
# Directory to save the downloaded files
dir_out = r'Y:\PNW\ERA5_Waves\PNW_Era5Spectral'

# Variables to download 
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form


# Locational limits of download
area_lims = [50, -130, 30, -115]  # North, West, South, East   (Limits for PNW)

# nino_poly_lon = wrapTo360([-170 -120 -120 -170]);
# nino_poly_lat = [-5 -5 5 5];


# Years to download
years = [1980,2023+1]
months = [1,12+1]



#===============================================================================
# %% Functions
#===============================================================================


def last_day_of_month(any_day):
    # The day 28 exists in every month. 4 days later, it's always next month
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)
    # subtracting the number of the current day brings us back one month
    return next_month - datetime.timedelta(days=next_month.day)

 
def get_months(years, months):
    
    years = ["%d" % (y) for y in range(years[0], years[1])]
    months = ["%d" % (m) for m in range(months[0], months[1])]
    
    dates = []
    for yy in years:
        for mm in months:
            dates.append([yy,mm])

    return dates
 
 
def retrieve(client, request, date):

    yy = date[0]
    mm = date[1]
    
    first_day = datetime.date(int(yy), int(mm), 1)
    last_day  = last_day_of_month(first_day)
    
    print(f"requesting year: {yy}, month: {mm} /n")
    request.update({"date": f'{first_day}/to/{last_day}'})
    client.retrieve(
        "reanalysis-era5-complete", request, f'ERA5_SpectralWaves_WC_{yy}_{mm}.nc'
    )
    return f"retrieved year: {yy}, month: {mm}"
 
 
def main(request):
    "concurrent request using 10 threads"
    client = cdsapi.Client()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(retrieve, client, request.copy(), date) for date in DATES
        ]
        for f in as_completed(futures):
            try:
                print(f.result())
                
            except:
                print("could not retrieve")
 
 
 
DATES = get_months(years,months)

 
if __name__ == "__main__":
 
    request = {
        "class"     : "ea",
        "direction" : "1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24",
        "domain"    : "g",
        "expver"    : "1",
        "frequency" : "1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30",
        "param"     : "251.140",
        "stream"    : "wave",
        "date"      : "",
        "time"      : "00:00:00/01:00:00/02:00:00/03:00:00/04:00:00/05:00:00/06:00:00/07:00:00/08:00:00/09:00:00/10:00:00/11:00:00/12:00:00/13:00:00/14:00:00/15:00:00/16:00:00/17:00:00/18:00:00/19:00:00/20:00:00/21:00:00/22:00:00/23:00:00",
        "type"      : 'an',
        "area"      : f'{area_lims[0]}/{area_lims[1]}/{area_lims[2]}/{area_lims[3]}',          # North, West, South, East. Default: global
        "grid"      : '0.36 / 0.36',  # Latitude/longitude. Default: spherical harmonics or reduced Gaussian grid
        "format"    : 'netcdf',     # Output needs to be regular lat-lon, so only works in combination with 'grid'!
        }
 
    main(request)

    
    