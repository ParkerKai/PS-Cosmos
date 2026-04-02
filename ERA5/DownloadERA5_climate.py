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
#dir_out = r'Y:\PS_Cosmos\01_data\Climate\ERA5\SST_Nino'
dir_out = r'X:\PNW\ERA5_Waves'
#dir_out = r'Y:\PS_Cosmos\01_data\Climate\ERA5\SST_PNW'
#dir_out = r'Y:\PS_Cosmos\01_data\Climate\ERA5\Precipitation'

# Variables to download 
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form

Var = ['mean_wave_direction',]

# 'mean_direction_of_total_swell', 'mean_period_of_total_swell',
        # 'mean_wave_direction', 'mean_wave_direction_of_first_swell_partition',
         # 'mean_wave_period', 'mean_wave_period_of_first_swell_partition',
         # 'peak_wave_period', 'significant_height_of_combined_wind_waves_and_swell',
         # 'significant_height_of_wind_waves', 'significant_wave_height_of_first_swell_partition',
         # 'wave_spectral_directional_width', 'wave_spectral_directional_width_for_swell']
         
         

#Var = ['sea_surface_temperature']
#Var = ['maximum_total_precipitation_rate_since_previous_post_processing',
#       'mean_total_precipitation_rate',
#        'minimum_total_precipitation_rate_since_previous_post_processing',
#        'precipitation_type',
#         'total_precipitation']



# Locational limits of download
area_lims = [65, 102, -65, -66]  # North, West, South, East
#area_lims = [5, -170, -5, -120]  # North, West, South, East

# nino_poly_lon = wrapTo360([-170 -120 -120 -170]);
# nino_poly_lat = [-5 -5 5 5];


# Years to download
years = range(1966,2023)

#===============================================================================
# %% Download
#===============================================================================

c = cdsapi.Client(debug=False, wait_until_complete=False)

for yy in years:
    
    ### Get the download ready by retrieiving the file information from the API 
    r = c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': Var,
            'year': str(yy),
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
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



    file_out = os.path.join(dir_out,'download_%s.nc') % (str(yy))
    r.download(file_out)


