# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:14:13 2024

This script plots differences to wave CDFS


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
import numpy as np
import xarray as xr
import scipy
import matplotlib
import pickle
import matplotlib.pyplot as plt

# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the DFM data resides
# dir_in = r'D:\DFM'
dir_in = r"Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries"
dir_out = r"C:\Users\kaparker\Documents\Projects\PS_Cosmos\figures\CDF_Diff"

# Station
Stat = 0

SLR = "000"

Mod_list = ["CMCC", "CNRM", "EcEarth", "GFDL", "HadGemHH", "HadGemHM", "HadGemHMsst"]

Month_list = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

county = "King"

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
    quants = np.arange(0, 1, 0.001)

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
# %% Plots
# ===============================================================================


fig = plt.subplots(3, 4)
fig[0].set_size_inches(10, 6)
ax = fig[0].get_axes()

diff_save = np.full([1000, 12, len(Mod_list)], np.nan)

for Mod_cnt, Mod in enumerate(Mod_list):
    for month in np.arange(1, 13, 1, dtype=int):
        print(f"Processing {Mod}, Month {month:02d}")

        # Load the CMIP6 historic data
        with open(
            os.path.join(
                dir_in,
                f"LUT_output_{county}_CMIP6_historical",
                Mod,
                f"CDFmonthly_{month:02d}_{county}_{Mod:s}.pkl",
            ),
            "rb",
        ) as f:
            cdf_cmipH = pickle.load(f)

        # Load the CMIP6 future data
        with open(
            os.path.join(
                dir_in,
                f"LUT_output_{county}_CMIP6_future",
                Mod,
                f"CDFmonthly_{month:02d}_{Mod:s}_{county}_SLR{SLR:s}.pkl",
            ),
            "rb",
        ) as f:
            cdf_cmipF = pickle.load(f)

        # Subset to station of interest
        cdf_cmipH = cdf_cmipH.loc[cdf_cmipH["stat"] == Stat]
        cdf_cmipF = cdf_cmipF.loc[cdf_cmipF["stat"] == Stat]

        # Calc difference
        diff, quants = calc_diff(cdf_cmipH, cdf_cmipF, "abs")

        diff_save[:, month - 1, Mod_cnt] = diff

        l1 = ax[month - 1].plot(quants, diff)
        l2 = ax[month - 1].plot([0, 1], [0, 0], "k--")

        ax[month - 1].set_xlim([0, 1])

        ax[month - 1].set_title(Month_list[month - 1])

        if month <= 8:
            ax[month - 1].set_xticklabels([])

        if month == 5:
            ax[month - 1].set_ylabel("Change in Hs by Quantile (m)")

        # if month == 1:
        #     ax[month-1].set_ylim([-100,100])
        # elif month == 2:
        #     ax[month-1].set_ylim([-50,50])
        # elif month == 3:
        #     ax[month-1].set_ylim([-50,50])
        # elif month == 4:
        #     ax[month-1].set_ylim([-25,25])
        # elif month == 5:
        #     ax[month-1].set_ylim([-20,20])
        # elif month == 6:
        #     ax[month-1].set_ylim([-40,40])
        # elif month == 7:
        #     ax[month-1].set_ylim([-30,30])
        # elif month == 8:
        #     ax[month-1].set_ylim([-30,30])
        # elif month == 9:
        #     ax[month-1].set_ylim([-30,30])
        # elif month == 10:
        #     ax[month-1].set_ylim([-40,40])
        # elif month == 11:
        #     ax[month-1].set_ylim([-50,50])
        # elif month == 12:
        #     ax[month-1].set_ylim([-50,50])

        # if month == 1:
        # ax[month-1].legend(['CMCC','CNRM','GFDL','HadGemHH','HadGemHM','HadGemHMsst'])

        ax[month - 1].grid()


diff_mean = np.mean(diff_save, axis=2)
for month in np.arange(1, 13, 1, dtype=int):
    l3 = ax[month - 1].plot(quants, diff_mean[:, month - 1], "k")


diff_mean = np.mean(diff_save, axis=2)
for month in np.arange(1, 13, 1, dtype=int):
    l3 = ax[month - 1].plot(quants, diff_mean[:, month - 1], "k")

fig[0].savefig(os.path.join(dir_out, "Wave_Monthly_Models.tiff"), dpi=600)


fig = plt.subplots(3, 4)
fig[0].set_size_inches(10, 6)
ax = fig[0].get_axes()


for month in np.arange(1, 13, 1, dtype=int):
    l1 = ax[month - 1].fill_between(
        quants,
        np.max(diff_save[:, month - 1, :], axis=1),
        np.min(diff_save[:, month - 1, :], axis=1),
        alpha=0.5,
        color="k",
    )

    l2 = ax[month - 1].plot(
        quants, np.mean(diff_save[:, month - 1, :], axis=1), color="k"
    )

    l3 = ax[month - 1].plot([0, 1], [0, 0], "k--")

    ax[month - 1].set_xlim([0, 1])

    ax[month - 1].set_title(Month_list[month - 1])

    if month <= 8:
        ax[month - 1].set_xticklabels([])

    if month == 5:
        ax[month - 1].set_ylabel(" Change in Q by Quantile")
    ax[month - 1].grid()

fig[0].savefig(os.path.join(dir_out, "Wave_Monthly_Range.tiff"), dpi=600)
