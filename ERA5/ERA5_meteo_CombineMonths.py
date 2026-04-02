#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unpack ERA5 downloaded data

This package unpacks downloaded ERA5 data.
It is designed to handle the messed up download file names
CDSAPI downloads the files as .zip instead of .nc



__author__ = Kai Parker (USGS)
__email__ = kaparker@usgs.gov
__status__ = Dev
__created__ = 2025-12-18
"""

# ===============================================================================
# %% Import Modules
# ===============================================================================

import xarray as xr
import os
from pathlib import Path
import shutil
from glob import glob
import numpy as np


# ===============================================================================
# %% User Input
# ===============================================================================


dir_in = r"Y:\PNW\data\ERA5Land_Meteo"
dir_temp = r"Y:\PNW\data\temp"  # Temporary directory for unpacking files
dir_out = r"Y:\PNW\data\ERA5_Land_Winds_Combined"


# ===============================================================================
# %% Unpack Files
# ===============================================================================

# Get all .nc files in the directory
files = glob(os.path.join(dir_in,'ERA5*'))

# Get years
# Split by underscore and pick the 4th element (index 3)
years = [np.int32(os.path.basename(f).split("_")[3]) for f in files]
years = np.unique(years)  # Get unique years

# Now for each year get the files
for year in years:
    print(f"Processing files for year: {year}")

    # Get all files for this year
    files = glob(os.path.join(dir_in, f"*_{year}_*"))

    # Run through each file, unpack it, and save as a list of xr.Datasets
    ds_save = []
    for file in files:
        
        #REname file to have .nc extension
        if not file.endswith(".nc"):
            file_new = file + ".nc"
            os.rename(file, file_new)
        else:
            file_new = file
        
        with xr.open_mfdataset(
            file_new,
            combine="by_coords",
            parallel=True,
            data_vars="minimal",
            coords="minimal",
            compat="override",  # if attrs differ across files; try 'no_conflicts' first
        ) as ds:
            # Ensure a clean, sorted time axis if present
            if "time" in ds.dims:
                ds = ds.sortby("time")

            # Change variable names a bit
            ds2 = (
                ds.rename_dims({"valid_time": "time"})
                .rename_vars({"valid_time": "time"})
                .drop_vars(["expver", "number"], errors="ignore")
            )
            ds_save.append(ds2.load())

    ds_out = xr.concat(ds_save, dim="time")

    # Write out to a single NetCDF
    ds_out.to_netcdf(
        os.path.join(dir_out, f"ERA5Land_Meteo_PNw_{year}.nc"),
        format="NETCDF4",
        engine="netcdf4",
    )

    print(f"Wrote combined dataset to {dir_out}")
