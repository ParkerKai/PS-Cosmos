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
dir_out = r'Y:\PNW\ERA5_Waves\WestCoast_SpectralParams'

# Variables to download 
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form

Var = ["mean_direction_of_total_swell",
        "mean_direction_of_wind_waves",
        "mean_period_of_total_swell",
        "mean_period_of_wind_waves",
        "mean_wave_direction_of_first_swell_partition",
        "mean_wave_direction_of_second_swell_partition",
        "mean_wave_direction_of_third_swell_partition",
        "mean_wave_period_based_on_first_moment_for_swell",
        "mean_wave_period_based_on_first_moment_for_wind_waves",
        "mean_wave_period_based_on_second_moment_for_swell",
        "mean_wave_period_based_on_second_moment_for_wind_waves",
        "mean_wave_period_of_first_swell_partition",
        "mean_wave_period_of_second_swell_partition",
        "mean_wave_period_of_third_swell_partition",
        "significant_height_of_total_swell",
        "significant_height_of_wind_waves",
        "significant_wave_height_of_first_swell_partition",
        "significant_wave_height_of_second_swell_partition",
        "significant_wave_height_of_third_swell_partition",
        "wave_spectral_directional_width_for_swell",
        "wave_spectral_directional_width_for_wind_waves",
        "mean_wave_direction",
        "mean_wave_period",
        "significant_height_of_combined_wind_waves_and_swell",
        "mean_wave_period_based_on_first_moment",
        "mean_zero_crossing_wave_period",
        "model_bathymetry",
        "peak_wave_period",
        "wave_spectral_directional_width",
        "wave_spectral_kurtosis",
        "wave_spectral_peakedness",
        "wave_spectral_skewness"]


# Locational limits of download
area_lims = [49, -130, 41, -124]  # North, West, South, East   (Limits for West Coast)

# nino_poly_lon = wrapTo360([-170 -120 -120 -170]);
# nino_poly_lat = [-5 -5 5 5];


# Years to download
years = range(1987,2023+1)   # Need +1 as range is non-inclusive for the end 
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
    

def ListOfDays(year,month):

    first_day = datetime.date(year, month, 1)
    last_day  = last_day_of_month(first_day)
    
    delta = last_day - first_day   # returns timedelta
    
    print(first_day)
    print(last_day)
    
    out = [f'{(first_day + datetime.timedelta(days=i)).day:02d}' for i in range(delta.days + 1)]
    
    return out 

#===============================================================================
# %% Download
#===============================================================================

c = cdsapi.Client(debug=False, wait_until_complete=False)

for yy in years:
    for mm in months:
        print(f'Processing year {yy}, Month {mm}')  
        
        
        days = ListOfDays(yy,mm)
        
        
        ### Get the download ready by retrieiving the file information from the API 
        r = c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': Var,
                'year': str(yy),
                'month': f'{mm:02d}',
                'day': days,
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'area': area_lims,
            },
        )


        # Download the File.
        # Code from here:  https://github.com/ecmwf/cdsapi/blob/master/examples/example-era5-update.py

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



        file_out = os.path.join(dir_out,f'ERA5_WaveParams_{yy}_{mm}.nc')
        r.download(file_out)


