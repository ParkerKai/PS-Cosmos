# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:56:45 2025

This script compares DFM outputs vs Tidegauge data

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
import geopandas as gpd
import pandas as pd
from scipy.interpolate import interp1d

# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the DFM data resides
# dir_in = r'D:\DFM'
dir_in_model = r"Y:\PS_Cosmos\02_models\DFM_Regional"
dir_in_TG = r"Y:\PS_Cosmos\01_data\WaterLevels\TideGauge"
dir_out = r"Y:\PS_Cosmos\Figures\DFM\validation"

Gauges = [
    "9443090",  # Neah Bay
    "9444090",  # Port Anageles
    "9444900",  # Port Townsend
    "9447130",  # SEattle
    "9446484",  # Tacoma
    "9449880",  # Friday Harbor
    "9449424",
]  # Cherry POint

# sites=['cdip162']

Gauge_name = [
    "Neah Bay",
    "Port Angeles",
    "Port Townsend",
    "Seattle",
    "Tacoma",
    "Friday Harbor",
    "Cherry Point",
]


# Variable to process (calculate stats and make plots)
Var_process = "wl"  # 'wl', 'tide', 'ntr'

# ===============================================================================
# %% Define some functions
# ===============================================================================
sys.path.append(r"C:\Users\kaparker\GitHub\Python\Kai_Python\General_Functions")
from Kai_MatlabTools import matlab2datetime

from Kai_ModelValidation import TaylorDiagram, mean_absolute_difference_corrected
from Kai_ModelValidation import ModelStat_Var, intersect_Var
from Kai_ModelValidation import bias, SpiderPlot


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


def Read_TG_Mat(dir_in_TG, gauge_ID):
    file_in = os.path.join(dir_in_TG, f"NOAA_TG_{gauge_ID}.mat")
    dat = scipy.io.loadmat(file_in)

    t_obs = matlab2datetime(dat["date_obs"].flatten(), "min")
    t_pre = matlab2datetime(dat["date_pred"].flatten(), "min")

    # interpolate tides onto the observed timeseries. Do this rather than intersection
    # To preserve the higher temporal resolution of the observations

    tide = gappy_interp(
        t_obs.to_numpy(),
        t_pre.to_numpy(),
        dat["wl_pre"].flatten(),
        maxgap=np.timedelta64(1, "D"),  # 1 day in nanoseconds
        extrapolate=False,
    )

    # f = scipy.interpolate.CubicSpline(toTimestamp(t_pre.to_numpy()),
    #                                   dat['wl_pre'].flatten(),
    #                                   extrapolate=False)

    # tide = f(toTimestamp(t_obs.to_numpy()))

    data_vars = {
        "wl": (
            ["time"],
            dat["wl_obs"].flatten(),
            {"units": "m", "long_name": "Water Level Observed"},
        ),
        "tide": (
            ["time"],
            tide,
            {"units": "m", "long_name": "Tide (Water Level Predicted)"},
        ),
        "ntr": (
            ["time"],
            dat["wl_obs"].flatten() - tide,
            {
                "units": "m",
                "long_name": "Non-Tidal Residual (Observed vs. predicted Water Levels)",
            },
        ),
        # 'quality':(['t_obs'],  dat['quality'].flatten(),
        #              {'units': 'm',
        #               'standard_name':'Sigma'}),
        # 'sigma':(['t_obs'],  dat['sigma'].flatten(),
        #              {'units': 'None',
        #               'standard_name':'Quality Code'}),
    }

    # define coordinates
    coords = {"time": (["time"], t_obs, {"standard_name": "time observed"})}
    # create dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "station": dat["station_info"][0][0][0][0][0][0],
            "Datum": dat["station_info"][0][0][0][0][0][2],
            "Units": dat["station_info"][0][0][0][0][0][3],
            "TimeZone": dat["station_info"][0][0][0][0][0][4],
        },
    )

    return ds


# ===============================================================================
# %% Read in the Tide Gauge data
# ===============================================================================
ds_gauge = []
for cnt, gauge_ID in enumerate(Gauges):
    ds = Read_TG_Mat(dir_in_TG, gauge_ID)

    # Conversion to NAVD88 (from VDATUM). Download of NAVD unavailable for gauge.
    # Port Townsend
    if gauge_ID == "9444900":
        ds["wl"] = ds["wl"] + 1.190
        ds["tide"] = ds["tide"] + 1.190
        ds.attrs["Datum"] = "NAVD88"

    # Friday Harbor
    if gauge_ID == "9449880":
        ds["wl"] = ds["wl"] + 1.271
        ds["tide"] = ds["tide"] + 1.271
        ds.attrs["Datum"] = "NAVD88"

    # Cherry POint
    if gauge_ID == "9449424":
        ds["wl"] = ds["wl"] + 1.317
        ds["tide"] = ds["tide"] + 1.317
        ds.attrs["Datum"] = "NAVD88"

    # Can't use resample as it doesn't preserve gaps
    # ds_sel = ds.resample(time="10min").interpolate("linear")

    # Resample to 10min
    rounded_dt = pd.to_datetime(ds["time"][0].values).round("D")
    t_start = np.datetime64(rounded_dt)

    rounded_dt = pd.to_datetime(ds["time"][-1].values).round("D")
    t_end = np.datetime64(rounded_dt)

    t_new = np.arange(t_start, t_end, np.timedelta64(10, "m"), dtype="datetime64[ns]")

    wl = gappy_interp(
        t_new,
        ds["time"].values,
        ds["wl"].values,
        maxgap=np.timedelta64(1, "D"),  # in np.timedelta64
        extrapolate=False,
    )

    tide = gappy_interp(
        t_new,
        ds["time"].values,
        ds["tide"].values,
        maxgap=np.timedelta64(1, "D"),  # in np.timedelta64
        extrapolate=False,
    )

    ntr = gappy_interp(
        t_new,
        ds["time"].values,
        ds["ntr"].values,
        maxgap=np.timedelta64(1, "D"),  # in np.timedelta64
        extrapolate=False,
    )

    # Add NTR as the subtraction of full model minus tide only
    data_vars = {
        "wl": (["time"], wl, {"units": "m", "long_name": "waterlevel"}),
        "tide": (["time"], tide, {"units": "m", "long_name": "tide"}),
        "ntr": (["time"], ntr, {"units": "m", "long_name": "non-tidal residual"}),
    }

    coords = {"time": (["time"], t_new, {"standard_name": "time"})}

    ds_gauge.append(xr.Dataset(data_vars=data_vars, coords=coords))


# ===============================================================================
# %% Read in the Model Data 2
# ===============================================================================

files = glob(
    os.path.join(
        dir_in_model, "ERA5", "ERA5", "ERA5_000", "Results_Combined", "DFM_wl*"
    )
)
ds_full = xr.open_mfdataset(
    files, engine="h5netcdf", parallel=True, chunks={"time": -1, "station": 1}
)

files = glob(
    os.path.join(
        dir_in_model, "ERA5", "ERA5_tidal_Results", "Results_Combined", "000", "DFM_wl*"
    )
)
ds_tidal = xr.open_mfdataset(
    files, engine="h5netcdf", parallel=True, chunks={"time": -1, "station": 1}
)

station_id = ds_full["station"].values

ds_model2 = []
for cnt, gauge_ID in enumerate(Gauges):
    # FInd index for station_id
    stat_id = next(
        (cnt for cnt, s in enumerate(station_id) if gauge_ID in str(s)), None
    )

    # Subset to station and from 1995 on (since thats when gauge records start)
    ds_sel = ds_full.isel(station=stat_id)
    ds_sel = ds_sel.sel(time=slice("1995-01-01", "2025-01-01"))
    ds_sel["waterlevel"] = ds_sel["waterlevel"] / 10000

    # Add tide from tide only runs
    ds_sel_tide = ds_tidal.isel(station=stat_id)
    ds_sel_tide = ds_sel_tide.sel(time=slice("1995-01-01", "2025-01-01"))
    ds_sel_tide["waterlevel"] = ds_sel_tide["waterlevel"] / 10000
    ds_sel = ds_sel.assign(tide=ds_sel_tide["waterlevel"])

    # Add NTR as the subtraction of full model minus tide only
    data_vars = {
        "ntr": (
            ["time"],
            ds_sel["waterlevel"].data - ds_sel["tide"].data,
            {"units": "m", "long_name": "Non-Tidal Residual"},
        )
    }

    coords = {"time": (["time"], ds_sel["time"].data, {"standard_name": "time"})}

    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    # create dataset
    ds_sel = ds_sel.assign(ntr=ds["ntr"])

    # Change variable names
    ds_sel = ds_sel.rename({"waterlevel": "wl"})

    ds_model2.append(ds_sel)


# ===============================================================================
# %% Read in the Model 1  Data
# ===============================================================================

# Load the data
files = os.path.join(dir_in_model, "BabakRelease", "Stations_1985_2015_WL_TS.nc")
ds_full = xr.open_mfdataset(
    files, engine="h5netcdf", parallel=True, chunks={"time": -1, "station": 1}
)

ds_full["time"] = matlab2datetime(ds_full["time"].values, "min")


ds_model1 = []
for cnt, gauge_ID in enumerate(Gauges):
    # Find index for station_id
    station_id = ds_full["sta_name"].values
    stat_id = next(
        (cnt for cnt, s in enumerate(station_id) if gauge_ID in str(s)), None
    )

    # Subset to station and from 1995 on (since thats when gauge records start)
    ds_sel = ds_full.isel(sta_name=stat_id)
    ds_sel = ds_sel.sel(time=slice("1995-01-01", "2025-01-01"))
    ds_sel["WL_1985_2015"] = ds_sel["WL_1985_2015"] / 100

    # Add tide from tide only runs
    station_id = ds_tidal["station"].values
    stat_id = next(
        (cnt for cnt, s in enumerate(station_id) if gauge_ID in str(s)), None
    )

    ds_sel_tide = ds_tidal.isel(station=stat_id)
    ds_sel_tide = ds_sel_tide.sel(time=slice("1995-01-01", "2025-01-01"))
    ds_sel_tide["waterlevel"] = ds_sel_tide["waterlevel"] / 10000
    ds_sel = ds_sel.assign(tide=ds_sel_tide["waterlevel"])

    # Add NTR as the subtraction of full model minus tide only
    data_vars = {
        "ntr": (
            ["time"],
            ds_sel["WL_1985_2015"].data - ds_sel["tide"].data,
            {"units": "m", "long_name": "Non-Tidal Residual"},
        )
    }

    coords = {"time": (["time"], ds_sel["time"].data, {"standard_name": "time"})}

    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    # create dataset
    ds_sel = ds_sel.assign(ntr=ds["ntr"])

    # Change variable names
    ds_sel = ds_sel.rename({"WL_1985_2015": "wl"})

    ds_model1.append(ds_sel)

# ===============================================================================
# %% Model validation Statistics
# ===============================================================================


metrics = [root_mean_squared_error, r2_score, mean_absolute_difference_corrected, bias]

t_intersect_save = []
Y_pred1 = []
Y_pred2 = []
slope1 = np.full((len(Gauges)), np.nan)
slope2 = np.full((len(Gauges)), np.nan)
model_perf1 = np.full((len(Gauges), len(metrics)), np.nan)
model_perf2 = np.full((len(Gauges), len(metrics)), np.nan)

for cnt1, gauge_ID in enumerate(Gauges):
    print(f"Processing Gauge {gauge_ID}")

    pull_gauge = ds_gauge[cnt1]
    pull_model1 = ds_model1[cnt1]
    pull_model2 = ds_model2[cnt1]

    # Temporal intersection (and save to save time for the future)
    t_intersect = intersect_Var([pull_model1, pull_model2], pull_gauge, Var_process)

    t_intersect_save.append(t_intersect)

    pull_gauge = pull_gauge.sel(time=t_intersect)
    pull_model1 = pull_model1.sel(time=t_intersect)
    pull_model2 = pull_model2.sel(time=t_intersect)

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


ModelMetrics = pd.DataFrame(
    {
        "Gauge": Gauges,
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

fig, ax = matplotlib.pyplot.subplots(2, 4, figsize=[8, 6])
ax = ax.flatten()
for cnt, site in enumerate(Gauges):
    pull_gauge = ds_gauge[cnt]
    pull_model1 = ds_model1[cnt]
    pull_model2 = ds_model2[cnt]

    # Get intersection (saved above to save time)
    t_intersect = t_intersect_save[cnt]

    pull_gauge = pull_gauge[Var_process].sel(time=t_intersect)
    pull_model1 = pull_model1[Var_process].sel(time=t_intersect)
    pull_model2 = pull_model2[Var_process].sel(time=t_intersect)

    ax[cnt].plot(pull_gauge, pull_model1, "b.", label="Model V1", alpha=0.5)
    ax[cnt].plot(pull_gauge, pull_model2, "r.", label="Model V2", alpha=0.5)

    ax[cnt].plot([-2.5, 4.5], [-2.5, 4.5], "k:")
    ax[cnt].set_xlim([-3, 4.5])
    ax[cnt].set_ylim([-3, 4.5])

    ax[cnt].plot(pull_gauge, Y_pred1[cnt], "c", label="Model Fit 1")
    ax[cnt].plot(pull_gauge, Y_pred2[cnt], "m", label="Model Fit 2")

    ax[cnt].grid()
    ax[cnt].set_title(site)

    ax[cnt].text(
        -2,
        4,
        f"RMSE M1= {ModelMetrics['RMSE1'][cnt].round(2)}",
        fontsize=8,
        bbox=dict(facecolor="grey", alpha=0.5),
    )
    ax[cnt].text(
        -2,
        3.5,
        f"RMSE M2= {ModelMetrics['RMSE2'][cnt].round(2)}",
        fontsize=8,
        bbox=dict(facecolor="grey", alpha=0.5),
    )
    if cnt == 4:
        ax[cnt].set_ylabel(f"Model {Var_process} (m)")
        ax[cnt].set_xlabel(f"Obs. {Var_process} (m)")

    if (cnt != 0) or (cnt != 4):
        ax[cnt].set_yticklabels([])

    if (cnt != 4) or (cnt != 5) or (cnt != 6) or (cnt != 7):
        ax[cnt].set_xticklabels([])

    if cnt == 6:
        ax[cnt].legend(loc="lower right")

fig.savefig(
    os.path.join(dir_out, f"DFM_Validation_Scatter_MultiMod_{Var_process}.tif"),
    dpi=800,
    bbox_inches="tight",
)

##############################################################################

xlims = [pd.Timestamp("1998-11-22"), pd.Timestamp("1998-12-29")]


fig, ax = matplotlib.pyplot.subplots(len(Gauges), 1, sharex="col", figsize=[8, 6])

for cnt, site in enumerate(Gauges):
    pull_gauge = ds_gauge[cnt]
    pull_model1 = ds_model1[cnt]
    pull_model2 = ds_model2[cnt]

    # Get intersection (saved above to save time)
    t_intersect = t_intersect_save[cnt]

    ax[cnt].plot(
        pull_gauge["time"].sel(time=t_intersect),
        pull_gauge[Var_process].sel(time=t_intersect),
        "k",
        label="Gauge",
    )

    ax[cnt].plot(
        pull_model1["time"].sel(time=t_intersect),
        pull_model1[Var_process].sel(time=t_intersect),
        "r",
        label="Model1",
    )

    ax[cnt].plot(
        pull_model2["time"].sel(time=t_intersect),
        pull_model2[Var_process].sel(time=t_intersect),
        "b",
        label="Model2",
    )

    ax[cnt].grid()

    ax[cnt].set_title(site)
    ax[cnt].set_xlim(xlims)

    if cnt != 6:
        ax[cnt].set_xticklabels([])


ax[4].set_ylabel(f"{Var_process} (m)")
ax[4].legend()

fig.savefig(
    os.path.join(dir_out, f"DFM_Validation_TS_MultiMod_{Var_process}.tif"),
    dpi=800,
    bbox_inches="tight",
)

############################### Taylor Plot#####################################

# Var_list = ['Hsig','Tp','Dir']

# for Var in Var_list:

#     subplot_ind = [141,142,143,144]
#     fig, ax = matplotlib.pyplot.subplots(1, 4)
#     ax = ax.flatten()

#     for cnt,site in enumerate(sites):
#         # Reference dataset
#         t_intersect = intersect_Var([modelv1.sel(station = site),
#                                      modelv2.sel(station = site)],
#                                      buoy.sel(station=site),'Hsig')

#         data = buoy[Var].sel(time=t_intersect,station=site)
#         refstd = np.nanstd(data,ddof=1)           # Reference standard deviation


#         # Generate models
#         m1 = modelv1[Var].sel(time=t_intersect,station=site)    # Model 1
#         m2 =  modelv2[Var].sel(time=t_intersect,station=site) # Model 2

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


#         ax[cnt].set_title(site)
#         ax[cnt].set_xticklabels([])
#         ax[cnt].set_yticklabels([])

#     fig.set_size_inches(12, 4)
#     fig.savefig(os.path.join(dir_out,f'Taylor_{Var}.tif'),  dpi=800,
#             bbox_inches='tight')


############################## Radar Plot #######################################


# num_models = 2

# fig, ax = matplotlib.pyplot.subplots(4, 2, subplot_kw=dict(polar=True),figsize=(8, 12),)

# ax = ax.flatten()

# for cnt,site in enumerate(Gauges):

#     row = ModelMetrics.iloc[np.array([ModelMetrics['Gauge'] == site]).flatten()]

#     df = pd.DataFrame({'Model':['Model 1 (Downscaled)','Model 2 (ERA5)'],
#                             'RMSE': row[['RMSE1','RMSE2']].to_numpy().flatten(),
#                             'R2': row[['R2_1','R2_2']].to_numpy().flatten(),
#                             'MADc': row[['MADc1','MADc2']].to_numpy().flatten(),
#                             'AbsBias':row[['AbsBias1','AbsBias2']].to_numpy().flatten(),
#                             'Slope':row[['Slope1','Slope2']].to_numpy().flatten()})

#     SpiderPlot(ax[cnt],df,
#         id_column='Model',
#         title='Model Validation Metrics',
#         padding=1.01)


# fig.savefig(os.path.join(dir_out,f'DFM_Validation_Radar_MultiMod_{Var_process}.tif'),  dpi=800,
#         bbox_inches='tight')


num_models = 2

for cnt, site in enumerate(Gauges):
    fig, ax = matplotlib.pyplot.subplots(
        1,
        1,
        subplot_kw=dict(polar=True),
        figsize=(3, 3),
    )

    row = ModelMetrics.iloc[np.array([ModelMetrics["Gauge"] == site]).flatten()]

    df = pd.DataFrame(
        {
            "Model": ["Model 1 (Downscaled)", "Model 2 (ERA5)"],
            "RMSE": row[["RMSE1", "RMSE2"]].to_numpy().flatten(),
            "R2": row[["R2_1", "R2_2"]].to_numpy().flatten(),
            "MADc": row[["MADc1", "MADc2"]].to_numpy().flatten(),
            "AbsBias": row[["AbsBias1", "AbsBias2"]].to_numpy().flatten(),
            "Slope": row[["Slope1", "Slope2"]].to_numpy().flatten(),
        }
    )

    SpiderPlot(ax, df, id_column="Model", title=f"{site}", padding=1.01)


    fig.savefig(
        os.path.join(
            dir_out,
            "RadarByGauge",
            f"DFM_Validation_Radar_MultiMod_{Var_process}_{site}.tif",
        ),
        dpi=600,
        bbox_inches="tight",
    )
