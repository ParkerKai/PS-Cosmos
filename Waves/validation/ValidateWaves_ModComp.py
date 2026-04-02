# -*- coding: utf-8 -*-
"""
Created on 12/12/2025

This script compares wave outputs vs observational data

Originally copied from

@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"

# ===============================================================================
# %% Import Modules
# ===============================================================================
import sys
import os
import scipy
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib
from glob import glob
from scipy.interpolate import interp1d
from scipy import io as sio
import re
import h5py
import matplotlib.patheffects as pe

sys.path.append(r"C:\Users\kaparker\GitHub\Python\Kai_Python\General_Functions")
from Kai_MatlabTools import matlab2datetime
from Kai_GeoTools import distance_ll
from Kai_ModelValidation import TaylorDiagram, mean_absolute_difference_corrected
from Kai_ModelValidation import ModelStat_Var, intersect_Var
from Kai_ModelValidation import bias, SpiderPlot


# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the DFM data resides
# dir_in = r'D:\DFM'
dir_in_model = (
    r"Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries\LUT_output_observations_ERA5"
)
dir_in_data = r"Y:\PS_Cosmos\02_models\Wave_LUT\01_data\WaveTimeSeries_observations"
dir_out = r"Y:\PS_Cosmos\Figures\Waves\validation"


# Variable to process (calculate stats and make plots)
Var_process = "Tm"  # 'Hs', 'Tm',


# ===============================================================================
# %% Define some functions
# ===============================================================================


def gappy_interp(xint, x0, y0, *, maxgap=None, **kwargs):
    """
    Interpolate as scipy.interpolate.CubicSpline,
    but fill np.NaN is gaps of x0 that are greater than *maxgap*.

    xint : 1-D sequence of np.datetime64[ns]
        The x-coordinates at which to evaluate the interpolated values.
    x0 : 1-D sequence of np.datetime64[ns]
        The x-coordinates of the data points, must be increasing if argument
        period is not specified. Otherwise, xp is internally sorted after
        normalizing the periodic boundaries with x0 = x0 % period.
    y0 : 1-D sequence of float or complex
        The y-coordinates of the data points, same length as x0.
        If nans are present they will be removed
    maxgap : np.timedelta64   e.g. np.timedelta64(1, 'D')
        maximum gap size in xint to interpolate over.  Data between gaps is
        filled with NaN.

    **kwargs :
        Passed to `scipy.interpolate.CubicSpline`.

    """

    # See if there are nans
    if np.sum(np.isnan(y0)) > 0:
        print(f"{np.sum(np.isnan(y0))} Nans Found. Removing for Interpolation")
        x0 = x0[~np.isnan(y0)]
        y0 = y0[~np.isnan(y0)]

    f = scipy.interpolate.PchipInterpolator(x0, y0, **kwargs)
    yint = f(xint)
    # yint = np.interp(xint, x0, y0, **kwargs)   # original version with linear np interolate

    # figure out which x0 each xint belongs to:
    x_index = np.searchsorted(x0, xint, side="right")
    x_index = np.clip(x_index, 0, len(x0) - 1)

    # figure out the space between sample pairs
    dx = np.concatenate(([0], np.diff(x0)))
    # get the gap size for each xint data point:
    # get the indices of xint that are too large:
    index = dx[x_index] > maxgap

    # this is fine, except the degenerate case when a xint point falls
    # directly on a x0 value.  In that case we want to keep the data at
    # that point.  So we just choose the other inequality for the index:

    # as above, but use side='right':
    x_index = np.searchsorted(x0, xint, side="right")
    x_index = np.clip(x_index, 0, len(x0) - 1)
    dx = np.concatenate(([0], np.diff(x0)))
    index = np.logical_and(index, (dx[x_index] > maxgap))

    # set interpolated values where xint is inside a big gap to NaN:
    yint[index] = np.nan

    return yint


def LoadWaveLUTmats(file_in):
    # Load the .mat file using h5py
    with h5py.File(file_in, "r") as f:
        group = f["LUTout"]
        Hs = group["Hs"][()]
        Tm = group["Tm"][()]

        t = group["t"][:, 0][()]

        # Convert from matlab to pandas datetimeIndex.
        # timeseries is in hours so round to hours to clean up conversion error.
        t_dt = matlab2datetime(t, "h")

        num_stat = Hs.shape[1]

        # Remove low Hs values
        ind = (Hs < 0.03) | (Tm < 1.0)
        Hs[ind] = np.nan
        Tm[ind] = np.nan

        # Wrestle into an xarray dataset
        data_vars = {
            "Hs": (["time", "station"], Hs),
            "Tm": (["time", "station"], Tm),
            "lat": (["station"], group["lat"][()].flatten()),
            "lon": (["station"], group["lon"][()].flatten()),
            "depth": (["station"], group["depth"][()].flatten()),
        }

        # define coordinates
        coords = {
            "time": (["time"], t_dt, {"standard_name": "time"}),
            "station": (["station"], np.arange(0, num_stat, 1)),
        }

        # create dataset
        ds = xr.Dataset(data_vars=data_vars, coords=coords)

    return ds


def decode_u16_matrix_to_strs(
    a: np.ndarray, encoding: str = "utf-16le", orientation: str = "auto"
) -> list[str]:
    """
    Decode a 2D uint16 matrix of UTF-16 code units into a list of Python strings.

    Parameters
    ----------
    a : np.ndarray
        2D array of dtype uint16 (e.g., h5py dataset[...] for a "names" field).
    encoding : str
        UTF-16 endianness ('utf-16le' or 'utf-16be'). Use 'utf-16le' for most files.
    orientation : {'auto', 'rows', 'cols'}
        - 'rows': each row is a string
        - 'cols': each column is a string (typical for MATLAB char arrays)
        - 'auto': try both and choose the one that looks more string-like

    Returns
    -------
    list[str]
        Decoded strings.
    """
    a = np.array(a, dtype=np.uint16)
    if a.ndim != 2:
        raise ValueError(f"Expected a 2D array; got shape {a.shape}")

    def decode_rows(A):
        out = []
        for row in A:
            # Trim at first null (0) if present
            if (row == 0).any():
                end = int(np.argmax(row == 0))
                row = row[:end]
            s = row.astype("<u2").tobytes().decode(encoding)
            # If fixed-length padding is used, strip trailing nulls just in case
            s = s.rstrip("\x00")
            out.append(s)
        return out

    def score(strings):
        # Heuristic: prefer orientation that yields fewer empty strings
        # and higher average non-empty length.
        lengths = [len(s) for s in strings]
        nonempty = [l for l in lengths if l > 0]
        empty_frac = (len(lengths) - len(nonempty)) / max(len(lengths), 1)
        avg_len = np.mean(nonempty) if nonempty else 0.0
        return empty_frac, avg_len

    if orientation == "rows":
        return decode_rows(a)
    elif orientation == "cols":
        return decode_rows(a.T)
    elif orientation == "auto":
        r = decode_rows(a)
        c = decode_rows(a.T)
        empty_r, avg_r = score(r)
        empty_c, avg_c = score(c)
        # Choose the orientation with fewer empties; tie-breaker: larger avg length
        if (empty_c < empty_r) or (empty_c == empty_r and avg_c >= avg_r):
            return c
        else:
            return r
    else:
        raise ValueError("orientation must be one of {'auto', 'rows', 'cols'}")


# ===============================================================================
# %% Read in the model data
# ===============================================================================

ds_model = LoadWaveLUTmats(
    os.path.join(dir_in_model, "LUT_output_observations_ERA5_spec_48.5_-125.mat")
)


# ===============================================================================
# %% Read in the Crosby model data
# ===============================================================================


# Load the .mat file using h5py
# with h5py.File(os.path.join(dir_in_model, "LUT_output_observations_Crosby.mat"), "r") as f:

f = h5py.File(os.path.join(dir_in_model, "LUT_output_observations_Crosby.mat"), "r")

group = f["SL"]
Hs = group["hs"][()].transpose()
Tm = group["tm"][()].transpose()
Md = group["md"][()].transpose()
Stat = group["names"][()]

t = group["time"][()].squeeze()

# Convert from matlab to pandas datetimeIndex.
# timeseries is in hours so round to hours to clean up conversion error.
t_dt = matlab2datetime(t, "h")


Stat = group["names"]
names = decode_u16_matrix_to_strs(Stat, orientation="cols")


# Wrestle into an xarray dataset
data_vars = {
    "Hs": (["time", "station"], Hs),
    "Tm": (["time", "station"], Tm),
    "Md": (["time", "station"], Md),
}

# define coordinates
coords = {
    "time": (["time"], t_dt, {"standard_name": "time"}),
    "station": (["station"], names),
}

# create dataset
ds_model_SC = xr.Dataset(data_vars=data_vars, coords=coords)

# ===============================================================================
# %% Read in the observed data
# ===============================================================================

files = glob(os.path.join(dir_in_data, "*.mat"))
files = files[:-1]


files = [item for item in files if not re.search(r".*AngelesPoint.mat", item)]
files = [item for item in files if not re.search(r".*HeinBank.mat", item)]
files = [item for item in files if not re.search(r".*Spotter_dontUse.mat", item)]


ds_gauge = []
temp = []
for cnt, file in enumerate(files):
    mat_contents = sio.loadmat(file)
    mat_contents = mat_contents["obsv"]

    t = matlab2datetime(mat_contents["t"][0][0].flatten(), "h")
    lon = mat_contents["lon"][0][0].flatten()
    lat = mat_contents["lat"][0][0].flatten()
    if "name" in mat_contents.dtype.names:
        stat = mat_contents["name"][0][0].flatten()
    elif "nme" in mat_contents.dtype.names:
        stat = mat_contents["nme"][0][0].flatten()

    else:
        match = re.search(r"obs_(.*?)\d*\.mat$", file)
        if match:
            site = match.group(1)
            print(site)  # AngelesPoint
        else:
            print("Pattern not found")

    if "Tm" in mat_contents.dtype.names:
        Tm = mat_contents["Tm"][0][0]
    else:
        Tm = np.full((len(t), 1), np.nan)  # No data for this variable at this station

    Hs = mat_contents["Hs"][0][0]

    # Filter the data to only unique time values
    t, ind = np.unique(t, return_index=True)
    Tm = Tm[ind]
    Hs = Hs[ind]

    # Wrestle into an xarray dataset
    data_vars = {
        "Tm": (["time", "station"], Tm),
        "Hs": (["time", "station"], Hs),
        "lat": (["station"], lat),
        "lon": (["station"], lon),
    }

    # define coordinates
    coords = {
        "time": (["time"], t, {"standard_name": "time"}),
        "station": (["station"], stat),
    }

    # create dataset
    ds_gauge.append(xr.Dataset(data_vars=data_vars, coords=coords))

    # Create the pandas dataframe of station locations
    mod_lon = ds_model["lon"].values
    mod_lat = ds_model["lat"].values

    # Distance
    dist = distance_ll(
        np.vstack((mod_lat, mod_lon)).T, np.vstack((lat.squeeze(), lon.squeeze())).T
    )

    ind_mod = np.argmin(dist, axis=0)

    if dist[ind_mod] > 10:
        print(
            f"Warning: Large distance between obs and model at station {stat}. Distance = {dist[ind_mod]} km"
        )
        ind_mod = np.array([np.nan])

    data_row = {
        "Station": stat.squeeze(),
        "Obs_Lat": lat.squeeze(),
        "Obs_Lon": lon.squeeze(),
        "Obs_ind": cnt,
        "Mod_ind": ind_mod.squeeze(),
    }

    if np.isnan(data_vars[Var_process][1]).all():
        data_row["Obs_ind"] = pd.NA

    temp.append(pd.DataFrame(data_row, index=[f"Row{cnt}"]))

# Connect the model data to observed data.
ds_index = pd.concat(temp)



## MAnually add index for Sean model locations (names are all messed up so doing manually)
ds_index["Mod2_ind"] = np.array(
    [3, 5, 6, 7, 1, 0, 8, 9, 10, 4, 11, 12, 13, pd.NA, pd.NA, pd.NA]
)


# Drop any stations that are dont have a model-observation match
ds_index.dropna(subset=["Mod_ind", "Obs_ind", "Mod2_ind"], inplace=True)


# Calculate the time intersection for all stations

t_intersect_save = []
ds_index_keep = np.full((len(ds_index)), True)
cnt = 0
for index, row in ds_index.iterrows():
    pull_gauge = ds_gauge[row["Obs_ind"]].squeeze("station")
    pull_model1 = ds_model.isel(station=int(row["Mod_ind"]))
    pull_model2 = ds_model_SC.isel(station=int(row["Mod2_ind"]))

    # Temporal intersection (and save to save time for the future)
    t_intersect = intersect_Var([pull_model1, pull_model2], pull_gauge, Var_process)

    t_intersect = np.unique(t_intersect)

    if len(t_intersect) == 0:
        print(f"No overlapping data found for station {row['Station']}")
        ds_index_keep[cnt] = False
    else:
        t_intersect_save.append(t_intersect)
    cnt += 1


ds_index = ds_index[ds_index_keep].reset_index(drop=True)


# ===============================================================================
# %% Model validation Statistics
# ===============================================================================


metrics = [root_mean_squared_error, r2_score, mean_absolute_difference_corrected, bias]


Y_pred1 = []
Y_pred2 = []
slope1 = np.full((len(ds_index)), np.nan)
slope2 = np.full((len(ds_index)), np.nan)
model_perf1 = np.full((len(ds_index), len(metrics)), np.nan)
model_perf2 = np.full((len(ds_index), len(metrics)), np.nan)
cnt1 = 0
for index, row in ds_index.iterrows():
    print(f"Processing Gauge {row['Station']}")

    pull_gauge = ds_gauge[row["Obs_ind"]].squeeze("station")
    pull_model1 = ds_model.isel(station=int(row["Mod_ind"]))
    pull_model2 = ds_model_SC.isel(station=int(row["Mod2_ind"]))

    # Get intersection (saved above to save time)
    t_intersect = t_intersect_save[cnt1]

    pull_gauge = pull_gauge.sel(time=t_intersect)
    pull_model1 = pull_model1.sel(time=t_intersect)
    pull_model2 = pull_model2.sel(time=t_intersect)

    if (len(pull_model1[Var_process].dims) != 1) or (
        len(pull_gauge[Var_process].dims) != 1
        or (len(pull_model1[Var_process].dims) != 1)
    ):
        raise Exception("Model and Observed variables must have 1 dimention (time)")

    # Calculate performance metric for models
    for cnt2, metric in enumerate(metrics):
        model_perf1[cnt1, cnt2] = ModelStat_Var(
            pull_model1, pull_gauge, Var_process, None, metric, intersect=False
        ).squeeze()

    for cnt2, metric in enumerate(metrics):
        model_perf2[cnt1, cnt2] = ModelStat_Var(
            pull_model2, pull_gauge, Var_process, None, metric, intersect=False
        ).squeeze()

    # Fit regression line
    model = LinearRegression()
    X = pull_gauge[Var_process].values
    Y = pull_model1[Var_process].values
    model.fit(X.reshape(-1, 1), Y)
    slope1[cnt1] = model.coef_.squeeze()
    Y_pred1.append(model.predict(X.reshape(-1, 1)))

    model = LinearRegression()
    X = pull_gauge[Var_process].values
    Y = pull_model2[Var_process].values
    model.fit(X.reshape(-1, 1), Y)
    slope2[cnt1] = model.coef_.squeeze()
    Y_pred2.append(model.predict(X.reshape(-1, 1)))

    cnt1 += 1

ModelMetrics = pd.DataFrame(
    {
        "Station": ds_index["Station"].values,
        "RMSE1": model_perf1[:, 0],
        "RMSE2": model_perf2[:, 0],
        "R2_1": model_perf1[:, 1],
        "R2_2": model_perf2[:, 1],
        "MADc1": model_perf1[:, 2],
        "MADc2": model_perf2[:, 2],
        "AbsBias1": np.absolute(model_perf1[:, 3]),
        "AbsBias2": np.absolute(model_perf2[:, 3]),
        "Slope1": slope1,
        "Slope2": slope2,
    }
)

# ===============================================================================
# %% Plots
# ===============================================================================

fig, ax = matplotlib.pyplot.subplots(5, 3, figsize=[6, 8])
ax = ax.flatten()
cnt = 0
for index, row in ds_index.iterrows():
    pull_gauge = ds_gauge[row["Obs_ind"]].squeeze("station")
    pull_model1 = ds_model.isel(station=int(row["Mod_ind"]))
    pull_model2 = ds_model_SC.isel(station=int(row["Mod2_ind"]))

    # Get intersection (saved above to save time)
    t_intersect = t_intersect_save[cnt]

    pull_gauge = pull_gauge[Var_process].sel(time=t_intersect)
    pull_model1 = pull_model1[Var_process].sel(time=t_intersect)
    pull_model2 = pull_model2[Var_process].sel(time=t_intersect)

    ax[cnt].plot(pull_gauge, pull_model1, "k.", label="Model ERA5", alpha=0.5)
    ax[cnt].plot(pull_gauge, pull_model2, "r.", label="Model Crosby", alpha=0.5)

    ax[cnt].plot([-2.5, 30], [-2.5, 30], "k:")
    ax[cnt].set_xlim(
        [
            0,
            np.max(
                [
                    pull_model2.max().values,
                    pull_gauge.max().values,
                    pull_model1.max().values,
                ]
            ),
        ]
    )
    ax[cnt].set_ylim(
        [
            0,
            np.max(
                [
                    pull_model2.max().values,
                    pull_gauge.max().values,
                    pull_model1.max().values,
                ]
            ),
        ]
    )

    ax[cnt].text(
        0.3,
        0.9,
        f"RMSE = {ModelMetrics['RMSE1'][cnt].round(2)}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[cnt].transAxes,
        fontsize=8,
        bbox=dict(facecolor="grey", alpha=0.5),
    )

    ax[cnt].text(
        0.3,
        0.78,
        f"RMSE = {ModelMetrics['RMSE2'][cnt].round(2)}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[cnt].transAxes,
        fontsize=8,
        bbox=dict(facecolor="pink", alpha=0.5),
    )

    ax[cnt].plot(
        pull_gauge,
        Y_pred1[cnt],
        "k",
        path_effects=[pe.Stroke(linewidth=3, foreground="w"), pe.Normal()],
    )
    ax[cnt].plot(
        pull_gauge,
        Y_pred2[cnt],
        "r",
        path_effects=[pe.Stroke(linewidth=3, foreground="w"), pe.Normal()],
    )

    ax[cnt].grid()
    ax[cnt].set_title(row["Station"])

    if cnt == 3:
        ax[cnt].set_ylabel(f"Model {Var_process} (m)")

    if cnt == 7:
        ax[cnt].set_xlabel(f"Obs {Var_process} (m)")

    if cnt != 6:
        ax[cnt].set_yticklabels([])

    if cnt != 6:
        ax[cnt].set_xticklabels([])

    if cnt == 2:
        ax[cnt].legend(loc="lower right")

    cnt += 1

fig.savefig(
    os.path.join(dir_out, f"Wave_Validation_ModComp_Scatter_{Var_process}.tif"),
    dpi=600,
    bbox_inches="tight",
)


##############################################################################

# xlims = [pd.Timestamp('2021-01-01-'),pd.Timestamp('2021-01-14')]


fig, ax = matplotlib.pyplot.subplots(len(ds_index), 1, figsize=[6, 8])

cnt = 0
for index, row in ds_index.iterrows():
    pull_gauge = ds_gauge[row["Obs_ind"]].squeeze("station")
    pull_model1 = ds_model.isel(station=int(row["Mod_ind"]))
    pull_model2 = ds_model_SC.isel(station=int(row["Mod2_ind"]))

    # Get intersection (saved above to save time)
    t_intersect = t_intersect_save[cnt]

    pull_gauge = pull_gauge[Var_process].sel(time=t_intersect)
    pull_model1 = pull_model1[Var_process].sel(time=t_intersect)
    pull_model2 = pull_model2[Var_process].sel(time=t_intersect)

    ax[cnt].plot(pull_gauge["time"], pull_gauge, "k", label="Gauge")

    ax[cnt].plot(pull_model1["time"], pull_model1, "r", label="Model ERA5")

    ax[cnt].plot(pull_model2["time"], pull_model2, "b", label="Model Crosby")

    ax[cnt].grid()

    ax[cnt].set_title(row["Station"])
    ax[cnt].set_xlim(t_intersect[0], t_intersect[0] + pd.Timedelta("7D"))

    ax[cnt].set_xticklabels([])

    cnt += 1

ax[4].set_ylabel("Mean Wave Period (sec)")
ax[4].legend()

fig.savefig(
    os.path.join(dir_out, f"Wave_Validation_ModComp_TS_{Var_process}.tif"),
    dpi=600,
    bbox_inches="tight",
)

##############################################################################


fig, [ax1, ax2] = matplotlib.pyplot.subplots(2, 1, sharex="col", figsize=[8, 5])

cnt = 0
row = ds_index.iloc[cnt]

pull_gauge = ds_gauge[row["Obs_ind"]].squeeze("station")
pull_model1 = ds_model.isel(station=int(row["Mod_ind"]))
pull_model2 = ds_model_SC.isel(station=int(row["Mod2_ind"]))

# Get intersection (saved above to save time)
t_intersect = t_intersect_save[cnt]


ax1.scatter(
    pull_gauge["time"].sel(time=t_intersect),
    pull_gauge["Hs"].sel(time=t_intersect),
    10,
    "k",
    label="Gauge",
)

ax1.scatter(
    pull_model1["time"].sel(time=t_intersect),
    pull_model1["Hs"].sel(time=t_intersect),
    10,
    "r",
)
ax1.plot(
    pull_model1["time"].sel(time=t_intersect),
    pull_model1["Hs"].sel(time=t_intersect),
    "r",
    label="Model ERA5",
)

ax1.scatter(
    pull_model2["time"].sel(time=t_intersect),
    pull_model2["Hs"].sel(time=t_intersect),
    10,
    "b",
)
ax1.plot(
    pull_model2["time"].sel(time=t_intersect),
    pull_model2["Hs"].sel(time=t_intersect),
    "b",
    label="Model Crosby",
)


ax1.grid()

ax1.set_title(row["Station"])
ax1.set_xlim([pd.Timestamp("2020-05-01"), pd.Timestamp("2020-05-14")])
ax1.set_ylabel("Wave Height (m)")
ax1.legend()


ax2.scatter(
    pull_gauge["time"].sel(time=t_intersect),
    pull_gauge["Tm"].sel(time=t_intersect),
    10,
    "k",
    label="Gauge",
)


ax2.scatter(
    pull_model1["time"].sel(time=t_intersect),
    pull_model1["Tm"].sel(time=t_intersect),
    10,
    "r",
    label="Model ERA5",
)
ax2.plot(
    pull_model1["time"].sel(time=t_intersect),
    pull_model1["Tm"].sel(time=t_intersect),
    "r",
    label="Model ERA5",
)


ax2.scatter(
    pull_model2["time"].sel(time=t_intersect),
    pull_model2["Tm"].sel(time=t_intersect),
    10,
    "b",
    label="Model Crosby",
)
ax2.plot(
    pull_model2["time"].sel(time=t_intersect),
    pull_model2["Tm"].sel(time=t_intersect),
    "b",
    label="Model Crosby",
)


ax2.grid()
ax2.set_xlim([pd.Timestamp("2020-05-01"), pd.Timestamp("2020-05-14")])
ax2.set_ylabel("Mean Wave Period (sec)")


# fig.savefig(
#     os.path.join(r'Y:\temp', f"W1_{Var_process}.tif"),
#     dpi=300,
#     bbox_inches="tight",
# )


##############################################################################

fig, ax = matplotlib.pyplot.subplots(5, 3, figsize=[6, 10])
ax = ax.flatten()
cnt = 0
for index, row in ds_index.iterrows():
    pull_gauge = ds_gauge[row["Obs_ind"]].squeeze("station")
    pull_model1 = ds_model.isel(station=int(row["Mod_ind"]))
    pull_model2 = ds_model_SC.isel(station=int(row["Mod2_ind"]))

    # Get intersection (saved above to save time)
    t_intersect = t_intersect_save[cnt]

    pull_gauge = pull_gauge
    pull_model1 = pull_model1
    pull_model2 = pull_model2
    ax[cnt].scatter(
        pull_gauge["Hs"].sel(time=t_intersect),
        pull_gauge["Tm"].sel(time=t_intersect),
        10,
        "k",
        label="Gauge",
        alpha=0.5,
    )

    ax[cnt].scatter(
        pull_model1["Hs"].sel(time=t_intersect),
        pull_model1["Tm"].sel(time=t_intersect),
        10,
        "r",
        label="Model ERA5",
        alpha=0.5,
    )

    ax[cnt].scatter(
        pull_model2["Hs"].sel(time=t_intersect),
        pull_model2["Tm"].sel(time=t_intersect),
        10,
        "b",
        label="Model Crosby",
        alpha=0.5,
    )

    ax[cnt].grid()

    ax[cnt].text(
        0.5,
        0.9,
        f"{row['Station']}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[cnt].transAxes,
        fontsize=8,
        bbox=dict(facecolor="grey", alpha=0.5),
    )

    if cnt == 6:
        ax[cnt].set_xlabel("Model Hs (m)")
        ax[cnt].set_ylabel("Model Tm (sec)")
        ax[cnt].legend(loc="upper right")

    cnt += 1

fig.savefig(
    os.path.join(dir_out, "Wave_HsTp_ModComp_Scatter.tif"), dpi=600, bbox_inches="tight"
)


############################### Taylor Plot#####################################

# Var_list = ['Hs','Tm']

# for Var in Var_list:

#     subplot_ind = [141,142,143,144]
#     fig, ax = matplotlib.pyplot.subplots(1, 4)
#     ax = ax.flatten()

#     cnt = 0
#     for index, row in ds_index.iterrows():
#         pull_gauge = ds_gauge[row["Obs_ind"]].squeeze("station")
#         pull_model1 = ds_model.isel(station=int(row["Mod_ind"]))
#         pull_model2 = ds_model_SC.isel(station=int(row["Mod2_ind"]))

#         # Get intersection (saved above to save time)
#         t_intersect = t_intersect_save[cnt]

#         pull_gauge = pull_gauge
#         pull_model1 = pull_model1
#         pull_model2 = pull_model2


#         data = pull_gauge[Var].sel(time=t_intersect)
#         refstd = np.nanstd(data,ddof=1)           # Reference standard deviation


#         # Generate models
#         m1 = pull_model1[Var].sel(time=t_intersect)    # Model 1
#         m2 = pull_model2[Var].sel(time=t_intersect) # Model 2

#         # Compute stddev and correlation coefficient of models
#         samples = np.array([ [np.nanstd(m,ddof=1), np.corrcoef(data, m)[0, 1]]
#                              for m in (m1,m2)])


#         # Taylor diagram
#         dia = TaylorDiagram(refstd, fig=fig,rect=subplot_ind[cnt], label="Reference",
#                             srange=(0.5, 1.5))

#         colors =  matplotlib.pyplot.matplotlib.cm.jet(np.linspace(0, 1, len(samples)))


#         # Add the models to Taylor diagram
#         for i, (stddev, corrcoef) in enumerate(samples):
#             dia.add_sample(stddev, corrcoef,
#                            marker='o', ms=10, ls='',
#                            mfc=colors[i], mec=colors[i],
#                            label=f"Model {i+1}")

#         # Add grid
#         dia.add_grid()

#         # Add RMS contours, and label them
#         contours = dia.add_contours(colors='0.5')
#         matplotlib.pyplot.clabel(contours, inline=1, fontsize=10, fmt='%.2f')

#         if cnt == 1:
#             # Add a figure legend
#             fig.legend(dia.samplePoints,
#                        [ p.get_label() for p in dia.samplePoints ],
#                        numpoints=1, prop=dict(size='small'), loc='upper right')


#         ax[cnt].set_title(row['Station'])
#         ax[cnt].set_xticklabels([])
#         ax[cnt].set_yticklabels([])

#     fig.set_size_inches(12, 4)
#     fig.savefig(os.path.join(dir_out,f'Taylor_ModelComp_{Var}.tif'),  dpi=800,
#             bbox_inches='tight')


############################## Radar Plot #######################################

num_models = 2

fig, ax = matplotlib.pyplot.subplots(
    5,
    3,
    subplot_kw=dict(polar=True),
    figsize=(8, 12),
)

ax = ax.flatten()

cnt = 0
for index, row in ds_index.iterrows():
    pull = ModelMetrics.iloc[
        np.array([ModelMetrics["Station"] == row["Station"]]).flatten()
    ]

    df = pd.DataFrame(
        {
            "Model": ["Model 1 (Downscaled)", "Model 2 (ERA5)"],
            "RMSE": pull[["RMSE1", "RMSE2"]].to_numpy().flatten(),
            "R2": pull[["R2_1", "R2_2"]].to_numpy().flatten(),
            "MADc": pull[["MADc1", "MADc2"]].to_numpy().flatten(),
            "AbsBias": pull[["AbsBias1", "AbsBias2"]].to_numpy().flatten(),
            "Slope": pull[["Slope1", "Slope2"]].to_numpy().flatten(),
        }
    )

    SpiderPlot(
        ax[cnt], df, id_column="Model", title="Model Validation Metrics", padding=1.01
    )

    cnt += 1


fig.savefig(
    os.path.join(dir_out, f"Wave_Validation_ModComp_Radar_{Var_process}.tif"),
    dpi=600,
    bbox_inches="tight",
)


######### Rounded Model Metrics ######

ModelMetrics_rounded = ModelMetrics.copy()
ModelMetrics_rounded = ModelMetrics.round(2)


print(
    f"Average RMSE for {Var_process}, Sean Runs: {ModelMetrics_rounded['RMSE2'].mean().round(2)}"
)
print(
    f"Average RMSE for {Var_process}, ERA5 Runs: {ModelMetrics_rounded['RMSE1'].mean().round(2)}"
)

# %%
