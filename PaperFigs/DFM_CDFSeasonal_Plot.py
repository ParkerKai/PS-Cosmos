# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:00:12 2024

This script Plots the Seasonal cdf corrections

@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"

# ===============================================================================
# %% Import Modules
# ===============================================================================
import os
from glob import glob
import numpy as np
import xarray as xr
import pickle
import scipy
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the DFM data resides
# dir_in = r'D:\DFM'
dir_in = r"Y:\PS_Cosmos\02_models\DFM_Regional\CMIP6_Results"
dir_out = r"Y:\PS_Cosmos\Figures\Paper"

# Model to process
Mod_list = [
    "CNRM-CM6-1-HR",
    "EC-Earth_HR",
    "GFDL",
    "HadGEM_GC31_HH",
    "HadGEM_GC31_HM_highRes",
    "HadGEM_GC31_HM_highResSST",
    "CMCC-CM2-VHR4",
]

# Station
Stat = 562

# SLR_list =['000','025','050','100','150','200','300']
SLR = "000"

Month_list = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

# ===============================================================================
# %% Define some functions
# ===============================================================================


def interpAtQuant(cdf_vals, cdf_quant, data_quant):
    # Determine CDF based on the pre-calculated ERA5 cdf
    interp_data = scipy.interpolate.interp1d(
        cdf_quant,
        cdf_vals,
        fill_value=(cdf_vals.min(), cdf_vals.max()),
        copy=False,
        assume_sorted=True,
        bounds_error=False,
    )
    vals = interp_data(data_quant)

    return vals


def calc_diff(cdf_H, cdf_F, version):
    import sys
    # Version abs :absolute difference
    # Version rel :relative  difference
    # Version per :relative percent difference

    # Calc CDF correction.
    quants = np.arange(0, 1, 0.00001)

    # PUll data for the station and unwrap pandas dataframe to numpy
    cdf_H_stat_cdf = cdf_H["cdf"].to_numpy()
    cdf_H_stat_val = cdf_H["values"].to_numpy()
    cdf_F_stat_cdf = cdf_F["cdf"].to_numpy()
    cdf_F_stat_val = cdf_F["values"].to_numpy()

    F = interpAtQuant(cdf_F_stat_val, cdf_F_stat_cdf, quants)
    H = interpAtQuant(cdf_H_stat_val, cdf_H_stat_cdf, quants)

    if version == "abs":
        diff = F - H

    elif version == "rel":
        diff = (F - H) / H

    elif version == "per":
        diff = ((F - H) / H) * 100

    else:
        sys.exit("Incorrection Version Selected")

    diff[~np.isfinite(diff)] = np.nan

    return diff, quants


# ===============================================================================
# %% Load the ERA5 data and calc quantiles
# ===============================================================================


# ===============================================================================
# %% Calculate correction for ERA5 based on CMIP6 projections
# ===============================================================================


diff_save = np.full([100000, 12, len(Mod_list)], np.nan)

for Mod_cnt, Mod in enumerate(Mod_list):
    for month in np.arange(1, 13, 1, dtype=int):
        print(f"Loading Model {Mod}, Month {month}, historic")

        with open(
            os.path.join(
                dir_in, Mod, "historic", "CDFmonthly_{0:02d}.pkl".format(month)
            ),
            "rb",
        ) as f:
            cdf_cmipH = pickle.load(f)

        print(f"Loading Model {Mod}, Month {month}, future")
        # Load the CMIP6 future data
        with open(
            os.path.join(
                dir_in, Mod, "future", SLR, "CDFmonthly_{0:02d}.pkl".format(month)
            ),
            "rb",
        ) as f:
            cdf_cmipF = pickle.load(f)

        print("Processing...")
        # Subset to station of interest
        cdf_cmipH = cdf_cmipH.loc[cdf_cmipH["stat"] == Stat]
        cdf_cmipF = cdf_cmipF.loc[cdf_cmipF["stat"] == Stat]

        # Calc difference
        diff, quants = calc_diff(cdf_cmipH, cdf_cmipF, "abs")

        # Convert to meters
        diff = diff / 10000

        diff_save[:, month - 1, Mod_cnt] = diff


# ===============================================================================
# %% Average to seasons
# ===============================================================================
# Seasons to average
Seasons = {'Winter':[11,0,1],
           'Spring':[2,3,4],
           'Summer':[5,6,7],
           'Fall':[8,9,10]}



Seasons_Agg = ['Winter','Spring','Summer','Fall']
diff_save_season = np.full([100000, 4, len(Mod_list)], np.nan)
for cnt,Season in enumerate(Seasons_Agg):
    print(f"Calculating {Season} average")
    months = Seasons[Season]
    diff_save_season[:,cnt,:] = np.mean(diff_save[:, months, :], axis=1)



# ===============================================================================
# %% Plot
# ===============================================================================

mod_labels = [
    "CNRM",
    "EC-Earth",
    "GFDL",
    "HadGEM_HH",
    "HadGEM_HM",
    "HadGEM_HM_SST",
    "CMCC",
]


fig = plt.subplots(2, 4, figsize=(10, 6), layout="tight")

ax = fig[0].get_axes()


for Mod_cnt, Mod in enumerate(Mod_list):
    for season_cnt in np.arange(1, 4+1, 1, dtype=int):
        diff_pull = diff_save_season[:, season_cnt - 1, Mod_cnt]

        l1 = ax[season_cnt - 1].plot(quants, diff_pull, linestyle='--', dashes=(5, 7), label=mod_labels[Mod_cnt])
        l2 = ax[season_cnt - 1].plot([0, 1], [0, 0], "k")

        ax[season_cnt - 1].set_xlim([0, 1])

        ax[season_cnt - 1].set_title(Seasons_Agg[season_cnt - 1])
        ax[season_cnt - 1].grid()


diff_mean = np.mean(diff_save_season, axis=2)
for season_cnt in np.arange(1, 4+1, 1, dtype=int):
    l3 = ax[season_cnt - 1].plot(
        quants, diff_mean[:, season_cnt - 1], "b", linewidth=2, label="Ensemble Mean"
    )

    ax[season_cnt - 1].set_ylim([-0.15, 0.15])
    ax[season_cnt - 1].set_xlim([0, 1])



###################################################################

for Mod_cnt, Mod in enumerate(Mod_list):
    for season_cnt in np.arange(1, 4+1, 1, dtype=int):
        diff_pull = diff_save_season[:, season_cnt - 1, Mod_cnt]

        l1 = ax[season_cnt - 1+4].plot(quants, diff_pull,linestyle='--' , label=mod_labels[Mod_cnt])
        l2 = ax[season_cnt - 1+4].plot([0, 1], [0, 0], "k")

        ax[season_cnt - 1+4].set_xlim([0, 1])


diff_mean = np.mean(diff_save_season, axis=2)
for season_cnt in np.arange(1, 4+1, 1, dtype=int):
    l3 = ax[season_cnt - 1+4].plot(
        quants, diff_mean[:, season_cnt - 1], "b", linewidth=2, label="Ensemble Mean"
    )

    ax[season_cnt - 1+4].set_ylim([-0.20, 0.20])
    ax[season_cnt - 1+4].set_xlim([0.98, 1])
    ax[season_cnt - 1+4].grid()


ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
ax[3].set_yticklabels([])
ax[5].set_yticklabels([])
ax[6].set_yticklabels([])
ax[7].set_yticklabels([])


ax[4].set_ylabel("Change in WL by Quantile (m)")
ax[5].set_xlabel('Quantile')


#ax[3].legend(facecolor='white', framealpha=1)

# ax[11].legend(loc="upper right")

fig[0].savefig(os.path.join(dir_out, "DFM_Seasonal_Models.tiff"), dpi=400)

