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
import matplotlib
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import scipy.optimize as optimize
import pyextremes
import matplotlib.pyplot as plt

# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the DFM data resides
# dir_in = r'D:\DFM'
dir_in = r"D:\Combined_DFM\ERA5"
dir_in_gis = r"Y:\PS_Cosmos\GIS\Shapefiles\general"
dir_out = r"Y:\PS_Cosmos\Figures\DFM\basin_Aggregate"


# ===============================================================================
# %% Define some functions
# ===============================================================================


# Use a function minimizer to figure out the actual threshold we want
def threshold_min_fun(thresh, Num_Exce, filt, data):
    from pypot.threshold_selection import get_extremes_peaks_over_threshold

    pks = get_extremes_peaks_over_threshold(data, thresh, r=filt)

    # Number of Peaks for this threshold
    num_peaks = pks.shape[0]

    # What is the difference between this and the amount we want
    # This is the function we are trying to minimize to zero
    Diff = np.absolute(Num_Exce - num_peaks)

    return Diff


def POT_theshold_SetNum(data, y_lab, r, Num_Exce):
    """select threshold for PoT analysis
    using a set number of exceedences.

    args:
        data (pd.DataFrame): data
        y_lab: (str) label of y column
        r (str): time delta string to define independence
        Num_Exce (np.float): number of exceedences we want

    returns:
        threshold (np.float)
    """

    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        if isinstance(data, pd.Series):
            data = data.to_frame(name=y_lab)

    else:
        print(f"Unrecognized input data type {type(data)}")
        print("Input must be a pandas dataframe or series")

    # Guess for the threshold
    B = data.sort_values(by=y_lab, na_position="last", ascending=False)
    thresh_guess = B[y_lab].iloc[Num_Exce]

    # Minimize
    bnds = optimize.Bounds(lb=B[y_lab].mean(), ub=B[y_lab].iloc[1])
    Optim_out = optimize.minimize(
        threshold_min_fun,
        x0=thresh_guess,
        args=(Num_Exce, r, data[y_lab]),
        bounds=bnds,
        method="Nelder-Mead",
    )

    if Optim_out["success"]:
        threshold = Optim_out["x"]
    else:
        print("Something went wrong!")

    return threshold


def CalcStats(files):
    # Data: Xarray dataset

    stats = np.full((len(files), 4), np.nan)
    lon = np.full((len(files)), np.nan)
    lat = np.full((len(files)), np.nan)
    for cnt, file in enumerate(files):
        data = xr.open_mfdataset(
            file, engine="netcdf4", parallel=True, decode_timedelta=True
        )

        # Load TWL values into memeory as a pandas series
        twl = pd.Series(
            data["waterlevel"].values + data["Hs"] * 0.20, index=data["time"].values
        )

        print(file)
        # only process if more than 60% the record is real values (not nans)
        if twl.isna().sum() < twl.shape[0] * 0.5:
            try:
                # FInd extremes

                # Get number of exceedances to grab
                num_years = np.unique(twl.index.year).shape[0]
                Npryr = 1
                Num_Exce = (num_years - 1) * Npryr

                selected_thresh = POT_theshold_SetNum(
                    twl,  # dataset
                    "twl",  # column name of pd series
                    "72h",  # independence time delta r
                    Num_Exce,
                )  # number of exceedences we want

                extremes = pyextremes.extremes.get_extremes(
                    twl,
                    method="POT",
                    extremes_type="high",
                    threshold=selected_thresh,
                    r="72h",
                )

                extremes = extremes.to_frame()

                # Get data associated with extremes
                extremes.insert(1, "ntr", data["ntr"].loc[extremes.index].values)
                extremes.insert(2, "tide", data["tide"].loc[extremes.index].values)
                extremes.insert(3, "eta", data["Hs"].loc[extremes.index].values * 0.20)

                extremes.rename(columns={"extreme values": "twl"}, inplace=True)

                # Calculate statistics on data (mean of extremes)
                stats[cnt, :] = np.transpose(extremes.mean(axis=0).to_numpy())

            except Exception as e:
                print(f"{file} has Maximum finding issues: {e}")

        lon[cnt] = data["lon_wl"].values
        lat[cnt] = data["lat_wl"].values

        # Plot
        # fig, (ax1,ax2) = matplotlib.pyplot.subplots(2, 1)
        # ax1.plot(out['time'],data['tide'],'k',label = 'wl(tide)')
        # ax1.plot(out['time'],data['trend'],'r',label = 'trend')
        # ax1.legend()
        # ax1.grid()
        # ax1.set_title('Water Levels')
        # ax1.set_ylabel('WL (NAVD88,m)')

        # ax2.plot(out['time'],out.values,'k',label = 'wl(tide)')

        # # #ax.set_xlim(pd.Timestamp('2000-08-01'),pd.Timestamp('2001-08-01'))
        # # #ax.set_xlim(100,200)
        # ax2.set_ylabel('WL (NAVD88,m)')
        # ax2.set_xlabel('Time')
        # ax2.grid()
        # ax2.legend()

    # Convert to pandas dataframe
    stats_out = pd.DataFrame(data=stats, columns=["twl", "ntr", "tide", "eta"])

    # add geometry and turn into geopandas dataset
    geometry = gpd.points_from_xy(x=lon, y=lat, crs="EPSG:4326")

    out = gpd.GeoDataFrame(data=stats_out, geometry=geometry)

    return out


# ===============================================================================
# %% Read in the match file
# ===============================================================================

Index_DFM = pd.read_csv(
    r"Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries\dfm_waveIndex_DFM.csv"
)
Index_LUT = pd.read_csv(
    r"Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries\dfm_waveIndex_LUT.csv"
)

Index_DFM = gpd.GeoDataFrame(
    Index_DFM,
    geometry=gpd.points_from_xy(Index_DFM["lon"], Index_DFM["lat"]),
    crs="EPSG:6318",
)
Index_LUT = gpd.GeoDataFrame(
    Index_LUT,
    geometry=gpd.points_from_xy(Index_LUT["lon"], Index_LUT["lat"]),
    crs="EPSG:6318",
)

Index_DFM.sindex
Index_LUT.sindex


# ===============================================================================
# %% Read in the Basin File and find basin location
# ===============================================================================

basins = gpd.read_file(os.path.join(dir_in_gis, "SalishSea_Basins.shp"))
basins = basins.to_crs(crs="EPSG:6318")


Index_LUT = gpd.sjoin(basins, Index_LUT, how="right")
Index_LUT = Index_LUT.rename(columns={"index_left": "BasinID"})
Index_LUT["BasinID"] = Index_LUT["BasinID"].fillna(-999)
Index_LUT["BasinID"] = Index_LUT["BasinID"].values.astype("int32")

Index_LUT = Index_LUT.drop(["Id"], axis=1)

Index_LUT["BasinID"].max()


IDs = np.unique(Index_LUT["BasinID"])
IDs = IDs[IDs >= 0]

Components = np.full((len(IDs), 4), np.nan)
for cnt, Id in enumerate(IDs):
    ind_basin = np.where(Index_LUT["BasinID"] == Id)
    ind_basin = ind_basin[0]
    files = [os.path.join(dir_in, f"CombinedTWL_{ii:04d}.nc") for ii in ind_basin]

    data = CalcStats(files)

    # Average the contribution across the basin
    Components[cnt, :] = data.drop(["geometry"], axis=1).mean(axis=0).to_numpy()

Components = pd.DataFrame(
    data=Components, columns=["twl", "ntr", "tide", "eta"], index=IDs
)


# ===============================================================================
# %% Plot
# ===============================================================================
# plt.style.use('dark_background')

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

n_basins = len(IDs)
fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(8, 4))
bottom = np.zeros(n_basins)

comp_plot = ["tide", "ntr", "eta"]
color_list = ['blue', 'red', 'green']
for cnt, comp in enumerate(comp_plot):
    pull = Components[comp].to_numpy()
    p = ax.bar(IDs, pull, width=0.8, 
               label=comp, bottom=bottom,
               zorder=3,color=color_list[cnt],
               edgecolor='black',alpha=0.8)
    bottom += pull

    # ax.bar_label(p, label_type="center", fmt="%.2f")


matplotlib.pyplot.xticks(ticks=np.arange(0, 10, 1), labels=Basin_labels, rotation=20)

ax.set_ylabel("Average Extreme Water Level (m)")

ax.grid(zorder=0)
ax.legend(["Tide", "NTR", "Setup"])
ax.set_title("Extreme Water Level Components Across Salish Sea Basins")

fig.savefig(
    os.path.join(dir_out, "WLcomponents_Basin.tif"), dpi=300, bbox_inches="tight"
)
