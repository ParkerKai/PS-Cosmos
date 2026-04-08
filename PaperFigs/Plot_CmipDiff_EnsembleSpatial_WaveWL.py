# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 11:22:32 2025

This script Plots maps of wave and water level differences
Fig XX for the paper

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
import matplotlib
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the DFM data resides
# dir_in = r'D:\DFM'
dir_in_wl = r"Y:\PS_Cosmos\GIS\DFM\DFM_CmipDiff_byModel"
dir_in_wv = r"Y:\PS_Cosmos\GIS\Waves\Wave_CmipDiff_byModel"

dir_out = r"Y:\PS_Cosmos\Figures\Paper"


# SLR_list =['000','025','050','100','150','200','300']
SLR_list = ["000"]

# Metric to plot
metric = "RP_30"

# ===============================================================================
# %% Define some functions
# ===============================================================================


# ===============================================================================
# %% Load general data
# ===============================================================================

# Load state shapefiles.
cnty = gpd.read_file(
    r"Y:\PS_Cosmos\GIS\general\Washington_Counties_with_Natural_Shoreline___washsh_area.shp"
)

# Load in the landmass file
lm = gpd.read_file(
    r"Y:\PS_Cosmos\GIS\general\PoliticalBoundaries_Shapefile\NA_PoliticalDivisions\data\bound_p\boundaries_p_2021_v3.shp"
)


# Load in the basin file
mask  = gpd.read_file(
    r"Y:\PS_Cosmos\GIS\Shapefiles\general\SalishSea_Basins.shp"
)

# Dissolve into a single (multi)polygon to avoid edges between parts
mask_union = mask.dissolve()  # single row with unified geometry


# ===============================================================================
# %% Load Water level (DFM) data
# ===============================================================================
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

# REad in a single file to get dimentions
file_in = os.path.join(dir_in_wl, f"Dfm_diff{SLR_list[0]}_{Mod_list[0]}.shp")
pull = gpd.read_file(file_in)


data = np.full([pull.shape[0], pull.shape[1] - 1, len(Mod_list)], np.nan)
for slr in SLR_list:
    for cnt, Mod in enumerate(Mod_list):
        print(f"Processing: {Mod}")
        file_in = os.path.join(dir_in_wl, f"Dfm_diff{slr}_{Mod}.shp")

        pull = gpd.read_file(file_in)
        data[:, :, cnt] = pull.drop(columns="geometry").to_numpy()


mean_data = np.nanmean(data, axis=2)
mean_data = pd.DataFrame(mean_data, columns=pull.drop(columns="geometry").columns)

plot_data_wl = gpd.GeoDataFrame(
    mean_data, geometry=pull["geometry"], crs=pull.crs
)  # Set the coordinate reference system)

# find index for metric of interest
columns = pull.drop(columns="geometry").columns
ind_want = np.ravel(np.argwhere(columns == metric))

# One sample t-test if mean isn't zero
p_vals = np.full([data.shape[0]], np.nan)
sign = np.full([data.shape[0]], False)
for cnt in range(data.shape[0]):
    # Sample data
    sample_pull = np.ravel(data[cnt, ind_want, :])

    # Population mean to test against
    population_mean = 0

    # Perform the one-sample t-test

    _, p_vals[cnt] = stats.ttest_1samp(sample_pull, population_mean, nan_policy="omit")

    if p_vals[cnt] < 0.05:
        sign[cnt] = True

plot_data_sig_wl = plot_data_wl.iloc[sign, :]


# ===============================================================================
# %% Load Water level (Wave) data
# ===============================================================================

Mod_list = ["CMCC", "CNRM", "EcEarth", "GFDL", "HadGemHH", "HadGemHM", "HadGemHMsst"]


# REad in a single file to get dimentions
file_in = os.path.join(dir_in_wv, f"Wave_diff{SLR_list[0]}_{Mod_list[0]}.shp")
pull = gpd.read_file(file_in)

data = np.full([pull.shape[0], pull.shape[1] - 2, len(Mod_list)], np.nan)
for slr in SLR_list:
    for cnt, Mod in enumerate(Mod_list):
        print(f"Processing: {Mod}")
        file_in = os.path.join(dir_in_wv, f"Wave_diff{slr}_{Mod}.shp")

        pull = gpd.read_file(file_in)
        data[:, :, cnt] = pull.drop(columns=["geometry", "county"]).to_numpy()


mean_data = np.nanmean(data, axis=2)
mean_data = pd.DataFrame(
    mean_data, columns=pull.drop(columns=["geometry", "county"]).columns
)

plot_data_wv = gpd.GeoDataFrame(
    mean_data, geometry=pull["geometry"], crs=pull.crs
)  # Set the coordinate reference system)


# find index for metric of interest
columns = pull.drop(columns=["geometry", "county"]).columns
ind_want = np.ravel(np.argwhere(columns == metric))

# One sample t-test if mean isn't zero
p_vals = np.full([data.shape[0]], np.nan)
sign = np.full([data.shape[0]], False)
for cnt in range(data.shape[0]):
    # Sample data
    sample_pull = np.ravel(data[cnt, ind_want, :])

    # Population mean to test against
    population_mean = 0

    # Perform the one-sample t-test

    _, p_vals[cnt] = stats.ttest_1samp(sample_pull, population_mean, nan_policy="omit")

    if p_vals[cnt] < 0.05:
        sign[cnt] = True

plot_data_sig_wv = plot_data_wv.iloc[sign, :]


# ===============================================================================
# %% Plot the  data
# ===============================================================================

# Filter geographically 

plot_data_wl = gpd.clip(plot_data_wl, mask_union)
plot_data_wv = gpd.clip(plot_data_wv, mask_union)

plot_data_sig_wl = gpd.clip(plot_data_sig_wl, mask_union)
plot_data_sig_wv = gpd.clip(plot_data_sig_wv, mask_union)



lm = lm.to_crs(crs=plot_data_sig_wl.crs)


if metric == "RI_1":
    vmin = -0.1
    vmax = 0.1
elif metric == "RP_30":
    vmin = -0.25
    vmax = 0.25
    

fig, [ax1, ax2] = plt.subplots(
    1, 2, figsize=(10, 7), layout="tight", sharex="row", sharey="row"
)

lm.plot(ax=ax1, figsize=(11, 6), color="0.8", edgecolor="black", alpha=0.4)
s1 = plot_data_sig_wl.plot(
    ax=ax1, column=metric, markersize=10, color="k", label="Statistically Significant"
)
s2 = plot_data_wl.plot(
    ax=ax1,
    column=metric,
    marker=".",
    markersize=10,
    cmap="coolwarm",
    label="Mean Change",
    vmin=vmin,
    vmax=vmax,
)

ax1.set_xlim([-124.75, -122])
ax1.set_ylim([47, 49.5])
ax1.grid()
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latittude")
ax1.set_title("CMIP6 Ensemble Change in Water levels")
ax1.legend(loc="lower left")


lm.plot(ax=ax2, color="0.8", edgecolor="black", alpha=0.4)
s1 = plot_data_sig_wv.plot(
    ax=ax2, column=metric, markersize=10, color="k", label="Statistically Significant"
)
s2 = plot_data_wv.plot(
    ax=ax2,
    column=metric,
    marker=".",
    markersize=10,
    cmap="coolwarm",
    label="Mean Change",
    vmin=vmin,
    vmax=vmax)
    #legend=True, legend_kwds={"label": "Ensemble Mean Change (m)"})
#)  #                    legend=True, cmap='coolwarm', legend_kwds={"label": "Ensemble Mean Change (m)"},


ax2.set_xlim([-124.75, -122])
ax2.set_ylim([47, 49.25])
ax2.grid()
ax2.set_title("CMIP6 Ensemble Change in Wave Height")
ax2.set_yticklabels([])

# elems = ax2.get_children()
# divider = make_axes_locatable(ax2)
# cax = divider.append_axes("right", size="5%", pad=0.1)
# matplotlib.pyplot.colorbar(elems[2], cax=cax)

fig.savefig(os.path.join(dir_out, f"MeanDiff_Map_{metric}_fixed.tiff"), dpi=400)
