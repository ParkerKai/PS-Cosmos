#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Custom module name or brief description.

This script modifies a SFINCS input files to test how incorrect wave forcing
effected model results.

__author__ = Kai Parker (USGS)
__email__ = kaparker@usgs.gov
__status__ = Dev
__created__ = 2026-01-22
"""

# ===============================================================================
# %% Import Modules
# ===============================================================================

import xarray as xr
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# ===============================================================================
# %% User Input
# ===============================================================================

folder_in = r"Y:\PS_Cosmos\02_models\SFINCS\20250122_synthetic_future_meanchange_100yr_Intel\01_King\TestWaves_000_SY000"


# ===============================================================================
# %% Read in the original SFINCS file
# ===============================================================================

ds1 = xr.open_dataset(os.path.join(folder_in, "sfincs_bndbzs_orig.nc"), decode_times=False)
ds2 = xr.open_dataset(os.path.join(folder_in, "sfincs_bndbzs2.nc"),decode_times=False)

zs1 = ds1['zs'].values
zs2 = ds2['zs'].values


fig, ax1 = plt.subplots(1, 1, figsize=[8, 5])

ax1.plot(
    ds1["time"],
    zs1[:,0],
    "k",
    label="ds1")



ax1.plot(
    ds2["time"],
    zs2[:,0],
    "r",
    label="ds2")


ax1.grid()

ax1.set_xlim([100000, 200000])
ax1.set_ylabel("Wave Height (m)")
ax1.legend()
