# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:45:11 2024

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
from glob import glob
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
import pickle
import scipy
import h5py
import sys

# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the DFM data resides
# dir_in = r'D:\DFM'
dir_in = r"Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries"
dir_out = r"C:\Users\kaparker\Documents\Projects\PS_Cosmos\figures\CDF_Diff\Waves"

# SLR_list =['000','025','050','100','150','200','300']
SLR = "000"

county = "King"

# Station
Stat_Lat = float(47) + (float(36.2) / 60)
Stat_Lon = -(float(122) + (float(20.4) / 60))

Mod_list = ["CMCC", "CNRM", "EcEarth", "GFDL", "HadGemHH", "HadGemHM", "HadGemHMsst"]
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
sys.path.append(
    r"C:\Users\kaparker\Documents\GitHub\Python\Kai_Python\General_Functions"
)
from Kai_MatlabTools import matlab2datetime
from Kai_GeoTools import distance_ll


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


def LoadWaveLUTmats(file_in):
    # Load the .mat file using h5py
    with h5py.File(file_in, "r") as f:
        group = f["LUTout"]
        hs = group["Hs"][()]
        dm = group["Dm"][()]
        tp = group["Tp"][()]
        t = group["t"][:, 0][()]
        lat = np.squeeze(group["lat"][()])
        lon = np.squeeze(group["lon"][()])
        depth = np.squeeze(group["depth"][()])
        stat = np.arange(0, len(lon), dtype="int32")
        dfm_id = np.squeeze(group["DFMid"][()])

        # Convert from matlab to pandas datetimeIndex.
        # timeseries is in hours so round to hours to clean up conversion error.
        t_dt = matlab2datetime(t, "h")

    # Turn into an Xarray dataset
    ds = xr.Dataset(
        {
            "Hs": xr.DataArray(
                data=hs,  # enter data here
                dims=["time", "station"],
                coords={"time": t_dt, "station": stat},
                attrs={"_FillValue": -9999, "units": "meters"},
            ),
            "Dm": xr.DataArray(
                data=dm,  # enter data here
                dims=["time", "station"],
                coords={"time": t_dt, "station": stat},
                attrs={"_FillValue": -9999, "units": "degrees"},
            ),
            "Tp": xr.DataArray(
                data=tp,  # enter data here
                dims=["time", "station"],
                coords={"time": t_dt, "station": stat},
                attrs={"_FillValue": -9999, "units": "seconds"},
            ),
            "Lat": xr.DataArray(
                data=lat,  # enter data here
                dims=["station"],
                coords={"station": stat},
                attrs={"_FillValue": -9999, "units": "Degree"},
            ),
            "Lon": xr.DataArray(
                data=lon,  # enter data here
                dims=["station"],
                coords={"station": stat},
                attrs={"_FillValue": -9999, "units": "Degree"},
            ),
            "depth": xr.DataArray(
                data=depth,  # enter data here
                dims=["station"],
                coords={"station": stat},
                attrs={"_FillValue": -9999, "units": "m"},
            ),
            "DFMid": xr.DataArray(
                data=dfm_id,  # enter data here
                dims=["station"],
                coords={"station": stat},
                attrs={"_FillValue": -9999, "units": "ID"},
            ),
        },
        attrs={
            "DataSource": rf"Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries\LUT_output_{county}_CMIP6",
            "ProducedBy": "Anita Englestad and Kai Parker",
            "General": "Tm,Tp and Dm are NaN for Hs < 0.05m. Output is for lat_10mIso/lon_10mIso locations. lat_ncPoint and lon_ncPoint are the original nc points to which the closest gridpoints on the 10m isobath were found",
        },
    )

    return ds


def emp_cdf(data):
    import scipy

    data = data[~np.isnan(data)]

    # Calculate the ecdf
    res = scipy.stats.ecdf(data)

    data_out = pd.DataFrame(
        data={"values": res.cdf.quantiles, "cdf": res.cdf.probabilities}
    )

    return data_out


# ===============================================================================
# %% Load the data
# ===============================================================================

diff = np.full((100000, len(Mod_list)), np.nan)
for cnt, Mod in enumerate(Mod_list):
    ds_his = LoadWaveLUTmats(
        os.path.join(
            dir_in,
            f"LUT_output_{county}_CMIP6_historical",
            f"LUT_output_{county}_{Mod}_his.mat",
        )
    )

    ds_fut = LoadWaveLUTmats(
        os.path.join(
            dir_in,
            f"LUT_output_{county}_CMIP6_future",
            f"{Mod}",
            f"LUT_output_{county}_{Mod}_SLR{SLR}.mat",
        )
    )

    # Find index for stations.
    Lat = ds_his["Lat"].values
    Lon = ds_his["Lon"].values

    dist = distance_ll(
        np.column_stack((Lat, Lon)), np.column_stack((Stat_Lat, Stat_Lon))
    )
    ind_pull = np.argmin(dist)

    ds_his = ds_his.isel(station=ind_pull, drop=True)

    Lat = ds_fut["Lat"].values
    Lon = ds_fut["Lon"].values

    dist = distance_ll(
        np.column_stack((Lat, Lon)), np.column_stack((Stat_Lat, Stat_Lon))
    )
    ind_pull = np.argmin(dist)

    ds_fut = ds_fut.isel(station=ind_pull, drop=True)

    # Calculate the cdf
    cdf_H = emp_cdf(ds_his["Hs"].values)
    cdf_F = emp_cdf(ds_fut["Hs"].values)

    # Calculate the difference
    diff[:, cnt], quants = calc_diff(cdf_H, cdf_F, version="abs")


# ===============================================================================
# %% Plots
# ===============================================================================

fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 6)

cmap = matplotlib.colormaps["cividis"].resampled(12).colors

month_plot = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for cnt, month in enumerate(month_plot):
    data_month = ds.sel(time=ds.time.dt.month.isin(month))

    l1 = ax.scatter(
        data_month["Hs"], data_month["hs_quants"], s=10, marker=".", color=cmap[cnt]
    )

ax.grid()
ax.set_title("Wave Height CDF")
ax.set_xlabel("Hs (m)")
ax.set_ylabel("CDF")
ax.set_ylim([0, 1])
ax.legend(
    ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
)
fig.savefig(os.path.join(dir_out, "HS_Monthly_CDFcompare.tiff"), dpi=600)


fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 6)

cmap = matplotlib.colormaps["cividis"].resampled(12).colors
for cnt, month in enumerate(month_plot):
    data_month = ds.sel(time=ds.time.dt.month.isin(month))

    l1 = ax.scatter(
        data_month["hs_quants"],
        data_month["cmip_diff"].mean(dim="cmip6"),
        s=10,
        marker=".",
        color=cmap[cnt],
    )

ax.grid()
ax.set_title("Wave Height CDF Diff")
ax.set_ylabel("Hs Diff (m)")
ax.set_xlabel("CDF")
ax.set_xlim([0, 1])
ax.legend(
    ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
)
fig.savefig(os.path.join(dir_out, "HS_Monthly_CDFdiffCompare.tiff"), dpi=600)

fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 6)

l1 = ax.plot(ds["time"], ds["Hs"], color="k", label="WaterLevel")

for ii in range(ds["cmip6"].size):
    ax.plot(ds["time"], ds["Hs"] + ds["cmip_diff"].isel(cmip6=ii))


ax.set_xlim(pd.Timestamp("1942-11-01"), pd.Timestamp("1942-11-10"))
# ax.set_xlim(100,200)
ax.grid()
ax.set_title("Wave Height")
ax.set_ylabel("Hs (m)")
ax.set_xlabel("Date")

ax.legend(
    ["ERA5", "CMCC", "CNRM", "EcEarth", "GFDL", "HadGemHH", "HadGemHM", "HadGemHMsst"]
)
fig.savefig(os.path.join(dir_out, "HS_Ts_AllMods.tiff"), dpi=600)


#######################################################################

fig, ax = matplotlib.pyplot.subplots(1, 1)
fig.set_size_inches(8, 6)

l1 = ax.plot(ds["time"], ds["Hs"], color="k", label="WaveHeight")

l1 = ax.fill_between(
    ds["time"],
    ds["Hs"] + ds["cmip_diff"].max(dim="cmip6"),
    ds["Hs"] + ds["cmip_diff"].min(dim="cmip6"),
    alpha=0.5,
)

l2 = ax.plot(ds["time"], ds["Hs"] + ds["cmip_diff"].mean(dim="cmip6"), color="b")

l3 = ax.plot(ds["time"], ds["Hs"], color="k", label="WaveHeight")

# Make the shaded region
# ax.fill(x,y,color='b',alpha = 0.5)

ax.set_xlim(pd.Timestamp("1942-11-01"), pd.Timestamp("1942-11-10"))
# ax.set_xlim(100,200)
ax.grid()
ax.set_title("Wave Height ")
ax.set_ylabel("Hs (m)")
ax.set_xlabel("Date")
ax.legend({"Cmip6 Model Range", "CMIP6 Main", "ERA5"})
fig.savefig(os.path.join(dir_out, "HS_Ts_Range.tiff"), dpi=600)

################################################################################


fig = matplotlib.pyplot.subplots(3, 4)
fig[0].set_size_inches(8, 6)
ax = fig[0].get_axes()


diff_save = np.full([1000, 12, ds["cmip6"].shape[0]], np.nan)

Mod_list = ds["cmip6"].values

for Mod_cnt, Mod in enumerate(Mod_list):
    print(f"Processing Model {Mod}")

    for month in np.arange(1, 13, 1, dtype=int):
        # Load the CMIP6 historic data
        with open(
            os.path.join(
                dir_in,
                "LUT_output_KingPierce_CMIP6_historical",
                Mod,
                f"CDFmonthly_{month:02d}_{Mod}.pkl",
            ),
            "rb",
        ) as f:
            cdf_cmipH = pickle.load(f)

        # Load the CMIP6 future data
        with open(
            os.path.join(
                dir_in,
                "LUT_output_KingPierce_CMIP6_future",
                Mod,
                f"CDFmonthly_{month:02d}_{Mod}_SLR{SLR}.pkl",
            ),
            "rb",
        ) as f:
            cdf_cmipF = pickle.load(f)

        # Subset to station of interest
        cdf_cmipH = cdf_cmipH.loc[cdf_cmipH["stat"] == Stat]
        cdf_cmipF = cdf_cmipF.loc[cdf_cmipF["stat"] == Stat]

        diff, quants = calc_diff(cdf_cmipH, cdf_cmipF, "per")

        diff_save[:, month - 1, Mod_cnt] = diff

        l1 = ax[month - 1].plot(quants, diff)
        l2 = ax[month - 1].plot([0, 1], [0, 0], "k--")

        ax[month - 1].set_xlim([0, 1])
        ax[month - 1].grid()

        ax[month - 1].set_title(Month_list[month - 1])

        if month <= 8:
            ax[month - 1].set_xticklabels([])

        if month == 5:
            ax[month - 1].set_ylabel("Perc. Change in Q by Quantile")

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

diff_mean = np.mean(diff_save, axis=2)
for month in np.arange(1, 13, 1, dtype=int):
    l3 = ax[month - 1].plot(quants, diff_mean[:, month - 1], "k")

fig[0].savefig(os.path.join(dir_out, "Wave%_Monthly_Models.tiff"), dpi=600)


################################################################################

fig = matplotlib.pyplot.subplots(3, 4)
fig[0].set_size_inches(8, 6)
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
    ax[month - 1].grid()

    ax[month - 1].set_title(Month_list[month - 1])

    if month <= 8:
        ax[month - 1].set_xticklabels([])

    if month == 5:
        ax[month - 1].set_ylabel("Perc. Change in Q by Quantile")

fig[0].savefig(os.path.join(dir_out, "Wave%_Monthly_Range.tiff"), dpi=600)
