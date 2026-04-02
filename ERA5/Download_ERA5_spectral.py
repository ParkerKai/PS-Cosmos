# -*- coding: utf-8 -*-
"""
Created on Oct 6th 16:17:00 2022

@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"

# Import Modules
import cdsapi
import time 
import os

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
years = range(1980,2023+1)   # Need +1 as range is non-inclusive for the end 
months = range(1,12+1)



#===============================================================================
# %% Functions
#===============================================================================
import datetime

def last_day_of_month(any_day):
    # The day 28 exists in every month. 4 days later, it's always next month
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)
    # subtracting the number of the current day brings us back one month
    return next_month - datetime.timedelta(days=next_month.day)
    

#===============================================================================
# %% Download
#===============================================================================

c = cdsapi.Client(debug=False, wait_until_complete=False)

for yy in years:
    for mm in months:
        print(f'Processing year {yy}, Month {mm}')
        
        
        first_day = datetime.date(yy, mm, 1)
        last_day  = last_day_of_month(first_day)
        
        dataset = 'reanalysis-era5-complete'
        request = {
            "class"     : "ea",
            "direction" : "1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24",
            "domain"    : "g",
            "expver"    : "1",
            "frequency" : "1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30",
            "param"     : "251.140",
            "stream"    : "wave",
            "date"      : f'{first_day}/to/{last_day}',
            "time"      : "00:00:00/01:00:00/02:00:00/03:00:00/04:00:00/05:00:00/06:00:00/07:00:00/08:00:00/09:00:00/10:00:00/11:00:00/12:00:00/13:00:00/14:00:00/15:00:00/16:00:00/17:00:00/18:00:00/19:00:00/20:00:00/21:00:00/22:00:00/23:00:00",
            "type"      : 'an',
            "area"      : f'{area_lims[0]}/{area_lims[1]}/{area_lims[2]}/{area_lims[3]}',          # North, West, South, East. Default: global
            "grid"      : '0.36 / 0.36',  # Latitude/longitude. Default: spherical harmonics or reduced Gaussian grid
            "format"    : 'netcdf',     # Output needs to be regular lat-lon, so only works in combination with 'grid'!
            }
        
        
        ### Get the download ready by retrieiving the file information from the API 
        r = c.retrieve(dataset, request)
        
        
        # Download the File.
        # Code from here:  https://github.com/ecmwf/cdsapi/blob/master/examples/example-era5-update.py
        print('Downloading file')
        sleep = 30
        while True:
            r.update()
            reply = r.reply
            r.info("Request ID: %s, state: %s" % (reply["request_id"], reply["state"]))

            if reply["state"] == "completed":
                break
            elif reply["state"] in ("queued", "running"):
                r.info("Request ID: %s, sleep: %s", reply["request_id"], sleep)
                time.sleep(sleep)
            elif reply["state"] in ("failed",):
                r.error("Message: %s", reply["error"].get("message"))
                r.error("Reason:  %s", reply["error"].get("reason"))
                for n in (
                    reply.get("error", {}).get("context", {}).get("traceback", "").split("\n")
                ):
                    if n.strip() == "":
                        break
                    r.error("  %s", n)
                raise Exception(
                    "%s. %s." % (reply["error"].get("message"), reply["error"].get("reason"))
                )


        file_out = os.path.join(dir_out, f'ERA5_SpectralWaves_WC_{yy}_{mm}.nc')
        r.download(file_out)
        print(f'Downloading of ERA5_SpectralWaves_PNW_{yy}_{mm}.nc Complete')
        
        

