# -*- coding: utf-8 -*-
"""
Created on Oct 6th 16:17:00 2022

This script downloads ERA5 data from copernicus.
User tells the script which variables to download and the area limits you would like.
It then downloads the data in year chunks

See here for example: 
https://cds.climate.copernicus.eu/api-how-to

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
dir_out = r'X:\PS_Cosmos\01_data\Climate\ERA5\test'


# Variables to download 
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form

Var = ['10m_u_component_of_wind', '10m_v_component_of_wind',
        '2m_temperature', 'mean_sea_level_pressure',
         'total_precipitation']

# Locational limits of download
area_lims = [52, -130, 46, -121]  # North, West, South, East


# Years to download
years = range(2022,2024)

#===============================================================================
# %% Download
#===============================================================================

c = cdsapi.Client(debug=False, wait_until_complete=False)

# Go through each year and download 
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


