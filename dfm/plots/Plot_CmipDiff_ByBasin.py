# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:56:45 2025

This script Plots the spatial variablity in changes to a specific metric

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
import matplotlib
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import scipy.stats as stats

from matplotlib import pyplot as plt 

# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the DFM data resides
# dir_in = r'D:\DFM'
dir_in = r"Y:\PS_Cosmos\GIS\DFM\DFM_CmipDiff_byModel"
dir_in_gis = r"Y:\PS_Cosmos\GIS\Shapefiles\general"
dir_out = r"Y:\PS_Cosmos\Figures\DFM\basin_Aggregate"

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


SLR_list = ["000", "050", "100", "200", "300"]

# Metric to plot
metric = "RP_30"


# ===============================================================================
# %% Define some functions
# ===============================================================================


# ===============================================================================
# %% Read in the Basin File and find basin location
# ===============================================================================

basins = gpd.read_file(os.path.join(dir_in_gis, "SalishSea_Basins.shp"))
basins = basins.to_crs(crs="EPSG:4326")


# REad in a single file to get dimentions
file_in = os.path.join(dir_in, f"Dfm_diff{SLR_list[0]}_{Mod_list[0]}.shp")
pnts = gpd.read_file(file_in)


# For each basin find the DFM_Pnts within
Index_DFM = gpd.sjoin(basins, pnts, how="right")
Index_DFM = Index_DFM.rename(columns={"index_left": "BasinID"})
Index_DFM["BasinID"] = Index_DFM["BasinID"].fillna(-999)
Index_DFM["BasinID"] = Index_DFM["BasinID"].values.astype("int32")

Index_DFM = Index_DFM.drop(
    [
        "Id",
        "Mean",
        "Max",
        "Std",
        "Q99",
        "Q95",
        "RI_1",
        "RP_5",
        "RP_10",
        "RP_15",
        "RP_30",
    ],
    axis=1,
)


save = []
for cnt_slr, slr in enumerate(SLR_list):
    for cnt_mod, Mod in enumerate(Mod_list):
        print(f"Processing: {slr},  {Mod}")
        file_in = os.path.join(dir_in, f"Dfm_diff{slr}_{Mod}.shp")

        pull = gpd.read_file(file_in)

        pull["BasinID"] = Index_DFM["BasinID"]

        # Remove NaN Stations (no Basin)
        pull = pull.drop(pull[pull["BasinID"] == -999].index)
        pull = pull.drop(["geometry"], axis=1)

        pull["Mod"] = np.full((pull.shape[0]), cnt_mod)
        pull["slr"] = np.full((pull.shape[0]), cnt_slr)

        save.append(pull)


data = pd.concat(save)


fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))


# plot violin plot

Basin_labels = [
    "S. Strait Georgia",
    "Central Sound",
    "South Sound",
    "Hood Canal",
    "Admiralty Inlet",
    "Bellingham Bay",
    "Whidbey Basin",
    "San Juan Islands",
    "Western S.J.D.F",
    "Eastern S.J.D.F",
]

for basin in np.unique(pull["BasinID"]):
    axs.violinplot(
        data.loc[(data["BasinID"] == basin) & (data["slr"] == 0)][metric].dropna()
        * 100,
        positions=[basin],
        showmeans=True,
        showmedians=False,
    )


axs.plot([-0.5, 9.5], [0, 0], color="black", linewidth=2, linestyle="--")
axs.set_title(f"Change in {metric}")
axs.grid()
axs.set_xlim([-0.5, 9.5])
plt.xticks(
    ticks=np.unique(pull["BasinID"]), labels=Basin_labels, rotation=20
)
axs.set_ylabel("Ensemble Change in Water Levels (cm)")

fig.savefig(
    os.path.join(dir_out, f"EnsembleChangeWL_Basin_{metric}.tif"),
    dpi=300,
    bbox_inches="tight",
)

############################################################################

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
cnt = 0
for basin in np.unique(pull["BasinID"]):
    vp1 = axs.violinplot(
        data.loc[(data["BasinID"] == basin) & (data["slr"] == 0)]["RI_1"].dropna()
        * 100,
        positions=[basin * 5 + cnt + 1],
        showmeans=True,
        showmedians=False,
        widths=1,
    )
    vp1["bodies"][0].set_facecolor("red")
    vp1["cmeans"].set_edgecolor("red")
    vp1["cmaxes"].set_edgecolor("red")
    vp1["cmins"].set_edgecolor("red")
    vp1["cbars"].set_edgecolor("red")

    vp2 = axs.violinplot(
        data.loc[(data["BasinID"] == basin) & (data["slr"] == 0)]["RP_5"].dropna()
        * 100,
        positions=[basin * 5 + cnt + 2],
        showmeans=True,
        showmedians=False,
        widths=1,
    )
    vp2["bodies"][0].set_facecolor("blue")
    vp2["cmeans"].set_edgecolor("blue")
    vp2["cmaxes"].set_edgecolor("blue")
    vp2["cmins"].set_edgecolor("blue")
    vp2["cbars"].set_edgecolor("blue")

    vp3 = axs.violinplot(
        data.loc[(data["BasinID"] == basin) & (data["slr"] == 0)]["RP_10"].dropna()
        * 100,
        positions=[basin * 5 + cnt + 3],
        showmeans=True,
        showmedians=False,
        widths=1,
    )
    vp3["bodies"][0].set_facecolor("green")
    vp3["cmeans"].set_edgecolor("green")
    vp3["cmaxes"].set_edgecolor("green")
    vp3["cmins"].set_edgecolor("green")
    vp3["cbars"].set_edgecolor("green")

    vp4 = axs.violinplot(
        data.loc[(data["BasinID"] == basin) & (data["slr"] == 0)]["RP_15"].dropna()
        * 100,
        positions=[basin * 5 + cnt + 4],
        showmeans=True,
        showmedians=False,
        widths=1,
    )
    vp4["bodies"][0].set_facecolor("cyan")
    vp4["cmeans"].set_edgecolor("cyan")
    vp4["cmaxes"].set_edgecolor("cyan")
    vp4["cmins"].set_edgecolor("cyan")
    vp4["cbars"].set_edgecolor("cyan")

    vp5 = axs.violinplot(
        data.loc[(data["BasinID"] == basin) & (data["slr"] == 0)]["RP_30"].dropna()
        * 100,
        positions=[basin * 5 + cnt + 5],
        showmeans=True,
        showmedians=False,
        widths=1,
    )
    vp5["bodies"][0].set_facecolor("orange")
    vp5["cmeans"].set_edgecolor("orange")
    vp5["cmaxes"].set_edgecolor("orange")
    vp5["cmins"].set_edgecolor("orange")
    vp5["cbars"].set_edgecolor("orange")

    cnt = cnt + 2

axs.plot([-0.5, 70], [0, 0], color="black", linewidth=2, linestyle="--")
axs.set_title("Change in Water Levels")
axs.grid()
axs.set_xlim([0, 69])
plt.xticks(
    ticks=[2.5, 9.5, 16.5, 23.5, 30.5, 37.5, 44.5, 51.5, 58.5, 65.5],
    labels=Basin_labels,
    rotation=20,
    ha="right",              # right-justify so the end of text sits on the tick
    rotation_mode="anchor",  # keep the anchor on the tick when rotated
)

axs.set_ylabel("Ensemble Change in Water Levels (cm)")
# axs.legend(
#     [
#         vp1["bodies"][0],
#         vp2["bodies"][0],
#         vp3["bodies"][0],
#         vp4["bodies"][0],
#         vp5["bodies"][0],
#     ],
#     ["RP1", "RP5", "RP10", "RP15", "RP30"],
#     loc="lower right",
# )


fig.savefig(
    os.path.join(dir_out, "EnsembleChangeWL_Basin_AcrossRP.tif"),
    dpi=300,
    bbox_inches="tight",
)

############################################################################

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
cnt = 0
for basin in np.unique(pull["BasinID"]):
    vp1 = axs.violinplot(
        data.loc[(data["BasinID"] == basin) & (data["slr"] == 0)][metric].dropna()[lambda x: x != 0]
        * 100
        - 0,
        positions=[basin * 5 + cnt + 1],
        showmeans=True,
        showmedians=False,
        widths=1,
    )
    vp1["bodies"][0].set_facecolor("red")
    vp1["cmeans"].set_edgecolor("red")
    vp1["cmaxes"].set_edgecolor("red")
    vp1["cmins"].set_edgecolor("red")
    vp1["cbars"].set_edgecolor("red")

    vp2 = axs.violinplot(
        data.loc[(data["BasinID"] == basin) & (data["slr"] == 1)][metric].dropna()[lambda x: x != 0]
        * 100
        - 50,
        positions=[basin * 5 + cnt + 2],
        showmeans=True,
        showmedians=False,
        widths=1,
    )
    vp2["bodies"][0].set_facecolor("blue")
    vp2["cmeans"].set_edgecolor("blue")
    vp2["cmaxes"].set_edgecolor("blue")
    vp2["cmins"].set_edgecolor("blue")
    vp2["cbars"].set_edgecolor("blue")

    vp3 = axs.violinplot(
        data.loc[(data["BasinID"] == basin) & (data["slr"] == 2)][metric].dropna()[lambda x: x != 0]
        * 100
        - 100,
        positions=[basin * 5 + cnt + 3],
        showmeans=True,
        showmedians=False,
        widths=1,
    )
    vp3["bodies"][0].set_facecolor("green")
    vp3["cmeans"].set_edgecolor("green")
    vp3["cmaxes"].set_edgecolor("green")
    vp3["cmins"].set_edgecolor("green")
    vp3["cbars"].set_edgecolor("green")

    vp4 = axs.violinplot(
        data.loc[(data["BasinID"] == basin) & (data["slr"] == 3)][metric].dropna()[lambda x: x != 0]
        * 100
        - 200,
        positions=[basin * 5 + cnt + 4],
        showmeans=True,
        showmedians=False,
        widths=1,
    )
    vp4["bodies"][0].set_facecolor("cyan")
    vp4["cmeans"].set_edgecolor("cyan")
    vp4["cmaxes"].set_edgecolor("cyan")
    vp4["cmins"].set_edgecolor("cyan")
    vp4["cbars"].set_edgecolor("cyan")

    vp5 = axs.violinplot(
        data.loc[(data["BasinID"] == basin) & (data["slr"] == 4)][metric].dropna()[lambda x: x != 0]
        * 100
        - 300,
        positions=[basin * 5 + cnt + 5],
        showmeans=True,
        showmedians=False,
        widths=1,
    )
    vp5["bodies"][0].set_facecolor("orange")
    vp5["cmeans"].set_edgecolor("orange")
    vp5["cmaxes"].set_edgecolor("orange")
    vp5["cmins"].set_edgecolor("orange")
    vp5["cbars"].set_edgecolor("orange")

    cnt = cnt + 2

axs.plot([-0.5, 70], [0, 0], color="black", linewidth=2, linestyle="--")
axs.set_title("Change in Water Levels by SLR Scenario")
axs.grid()
axs.set_xlim([0, 69])
plt.xticks(
    ticks=[2.5, 9.5, 16.5, 23.5, 30.5, 37.5, 44.5, 51.5, 58.5, 65.5],
    labels=Basin_labels,
    rotation=20,
    ha="right",              # right-justify so the end of text sits on the tick
    rotation_mode="anchor",  # keep the anchor on the tick when rotated
)
axs.set_ylabel("Ensemble Change in Water Levels (cm)")

axs.legend(
    [
        vp1["bodies"][0],
        vp2["bodies"][0],
        vp3["bodies"][0],
        vp4["bodies"][0],
        vp5["bodies"][0],
    ],
    ["SLR 000", "SLR 050", "SLR 100", "SLR 200", "SLR 300"],
    loc="upper right",
)


fig.savefig(
    os.path.join(dir_out, f"EnsembleChangeWL_Basin_AcrossSLR_{metric}.tif"),
    dpi=300,
    bbox_inches="tight",
)
