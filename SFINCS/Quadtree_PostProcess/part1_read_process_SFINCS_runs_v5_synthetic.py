"""
Created on Tue Feb 13 11:37:51 2024
Python script to read and process SFINCS netcdf outputs

v0.1  Nederhoff   2023-03-07
v0.3  Nederhoff   2023-11-24
v0.4  Parker      2026-03-30   move to quadtree

@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"

# ===============================================================================
# Import Modules
# ===============================================================================

# Modules needed
from ast import List
from importlib.resources import path
import os
import netCDF4 as nc
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import xarray as xr
from functools import partial
from typing import Optional, Sequence, Tuple, Literal

from POT_Extremes import pot_threshold_set_num_xr, get_extremes_pot_xr, rp_axis
from Xarray_NCtools import (
    batch_check_nc_files,
    ensure_unique_sorted_time,
    _detect_time_coord,
)
from SFINCS_QuadtreeTools import load_SfincsQuadtree

# ===============================================================================
# User inputs
# ===============================================================================

destin = r"D:\Kai\SFINCS"
destout = r"C:\Users\kai\Documents\KaiRuns\PostProcess"

return_period_wanted = ["1", "2", "5", "10", "20", "50", "100"]
SLRs_formatted = ["000"]
counties = ["03_Kitsap"]
sub_categories = ["_median"]  # ,'_low','_high'


# Choose the data variables you want to keep
# Define at module scope (important for Dask pickling when parallel=True)
vars_to_keep: Sequence[str] | None = ["zsmax"]
coords_to_keep: Sequence[str] | None = ["timemax"]


# Some processing of the inputs
return_period_wanted = [int(RP) for RP in return_period_wanted]
sub_categories = [sub.replace("_median", "") for sub in sub_categories]


# User settings
# return_period_wanted    = [1, 2, 5, 10, 20, 50, 100]     # number of years requested
hh_criteria = 0.010001  # just above the treshold from SFINCS

# Directory
include_qmax = True
include_tmax = False


# ===============================================================================
# Functions
# ===============================================================================


def preprocess_(
    ds: xr.Dataset,
    year: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    time_name: Optional[str] = None,
    mode_year_if_unspecified: bool = True,
    tie_break: Literal["earliest", "latest"] = "earliest",
) -> xr.Dataset:
    """
      1) Keep only selected data variables and coordinates.
      2) Trim by time:
         - If `year` is provided: keep only that year.
         - Else if `start`/`end` provided: keep inclusive date window.
         - Else if `mode_year_if_unspecified` True: keep the most common year
           (drops small spillover timepoints at the front/back).
         - Else: no trimming.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset loaded from a single file.
    year : int, optional
        If provided, keep only samples whose .dt.year == `year`.
    start, end : str (ISO date), optional
        If provided, keep data where time is within [start, end] inclusive.
        Only applied if `year` is None. Examples: "1999-01-01", "1999-12-31".
    time_name : str, optional
        Explicit time coordinate name. If None, will auto-detect.
    mode_year_if_unspecified : bool, default True
        If no explicit `year` or `[start, end]` is given, trim to the most common year.
    tie_break : {'earliest','latest'}, default 'earliest'
        How to resolve ties if multiple years have the same count.

    Notes
    -----
    - Assumes `vars_to_keep` and `coords_to_keep` exist in the outer scope
    - Works with both numpy datetime64 and cftime calendars via `.dt.year`.
    """
    # --- 1) Subset variables and coords
    requested = [*(vars_to_keep or []), *(coords_to_keep or [])]
    names = [n for n in requested if n in ds]  # ds keys include data_vars + coords

    if names:
        ds = ds[names]  # single selection of both data vars and coords
    else:
        # If nothing matched, keep only coords by dropping all data vars (optional)
        ds = ds.drop_vars(list(ds.data_vars))

    # --- 2) Trim by time if requested ---
    # Detect time coordinate (assumes you have `_detect_time_coord` defined)
    tname = _detect_time_coord(ds, preferred=time_name)  # noqa: F821
    if tname is None:
        return ds  # No time coordinate present; nothing to trim.

    # Ensure time is a coordinate
    if tname not in ds.coords:
        try:
            ds = ds.set_coords(tname)
        except Exception:
            return ds  # Can't promote; skip trimming

    time_da = ds[tname]

    # Case A: explicit year
    if year is not None:
        mask = time_da.dt.year == year
        ds = ds.where(mask, drop=True)
        return ds

    # Case B: explicit [start, end] inclusive window
    if start is not None or end is not None:
        mask = xr.ones_like(time_da, dtype=bool)
        if start is not None:
            mask = mask & (time_da >= start)
        if end is not None:
            mask = mask & (time_da <= end)
        ds = ds.where(mask, drop=True)
        return ds

    # Case C: trim to most common year if unspecified
    if mode_year_if_unspecified:
        # xarray-lazy approach: group by year and count
        years = time_da.dt.year  # works for both numpy datetime64 and cftime
        try:
            counts = years.groupby(years).count()  # dimension name becomes 'year'
            max_count = counts.max()
            # candidates are the year labels where count == max_count
            candidates = counts["year"].where(counts == max_count, drop=True)
            if candidates.size == 0:
                return ds  # Safety: shouldn't happen, but avoid errors
            if tie_break == "latest":
                mode_year_val = int(candidates.max().item())
            else:
                mode_year_val = int(candidates.min().item())
        except Exception:
            # Fallback using NumPy
            yvals = years.values
            if yvals.size == 0:
                return ds
            uniq, occ = np.unique(yvals, return_counts=True)
            # Tie-break
            max_occ = occ.max()
            candidates = uniq[occ == max_occ]
            mode_year_val = int(
                candidates.max() if tie_break == "latest" else candidates.min()
            )

        # Apply trim
        ds = ds.where(years == mode_year_val, drop=True)
        return ds

    # No trimming requested
    return ds


# ===============================================================================
# Some pre-processing
# ===============================================================================
if include_qmax:
    vars_to_keep.append("qmax")

if include_tmax:
    vars_to_keep.append("tmax")


# ===============================================================================
# Load the data
# ===============================================================================

# Go to folder and loop over domains
for county in counties:
    # Start with this county first
    print(f"Started with {county}", flush=True)
    destin_TMP = os.path.join(destin, county)

    # Go over SLRs
    for index, slr in enumerate(SLRs_formatted):
        # Go over sub_categories (standard, low and high)
        for index_cat, category in enumerate(sub_categories):
            # Make print statement
            print(f" => SLR value: {slr} - {sub_categories[index_cat]}", flush=True)
            destin_TMP = os.path.join(destin, county)
            TMP_string = slr + category
            destin_TMP = os.path.join(destin_TMP, TMP_string)

            # ===============================================================================
            # Check all the files are good to go
            # ===============================================================================

            # Get list of files for the run
            dirs = [entry.name for entry in os.scandir(destin_TMP) if entry.is_dir()]
            files = [
                os.path.join(destin_TMP, dir_pull, "sfincs_map.nc") for dir_pull in dirs
            ]

            # Check the files before we try to combine them.

            good_files, report, schema = batch_check_nc_files(
                files,
                required_vars=vars_to_keep,
                required_coords=coords_to_keep,
                check_time=True,
                sample_data=False,  # set True if you want to trigger decode/scale on tiny slices
            )

            # Throw a Runtime error if any of the netCDF files have issues, with a readable message of what the issues are.
            bad = [r for r in report if r["issues"]]
            if bad:
                lines = ["=== Issues detected in NetCDF files ==="]
                for r in bad:
                    lines.append(os.path.basename(r["path"]))
                    for iss in r["issues"]:
                        lines.append(f"  - {iss}")
                # Raise exception with a readable multi-line message
                raise RuntimeError("\n".join(lines))

            # Read netcdf in WY folders
            count_years = 0
            total_years = len(files)
            estimate_runtimes = []

            # ===============================================================================
            # Load all the Data
            # ===============================================================================

            # Automatically trim to the most common year (mode) if you don't pass year/start/end.
            # If multiple years tie (same count), choose the earliest; set tie_break='latest' to prefer the latest.
            preprocess_trim = partial(
                preprocess_,  # your updated function
                mode_year_if_unspecified=True,
                tie_break="earliest",  # or 'latest'
                # Optional: force the time coordinate name if you know it
                # time_name="time",
            )

            ds = xr.open_mfdataset(
                files,
                combine="nested",
                concat_dim="timemax",  # use the name you detected (e.g., 'timemax')
                compat="no_conflicts",
                coords="minimal",
                data_vars="all",  # or 'minimal' if variables identical
                parallel=True,
                decode_cf=True,
                preprocess=preprocess_trim,  # you can keep trimming/sanitizing
                engine="netcdf4",
            )

            ds = ds.chunk({"timemax": -1, "nmesh2d_face": 1})

            ds = ensure_unique_sorted_time(ds, time_name="timemax", keep="first")

            # Load the bedlevel
            xc, yc, zb = load_SfincsQuadtree(os.path.join(destin_TMP, "sfincs.nc"))

            # ===============================================================================
            # Calculate the maximums
            # ===============================================================================

            r = "48h"  # Deculstering time window for POT (e.g., 24h, 48h, etc.)
            num_exce = total_years
            num_grid = ds["nmesh2d_face"].size

            # Initialize output arrays
            tzsmax_out = np.empty((num_grid, num_exce), dtype="datetime64[ns]")

            zsmax_out = np.empty((num_grid, num_exce), dtype=np.float32)

            # create a mask to identify stations with all NaN or all zero values across time
            mask_all_nan = (
                ds["zsmax"].isnull().all(dim="timemax")
            )  # True where every time step is NaN for each face
            mask_all_zero = (ds["zsmax"].fillna(0) == 0).all(dim="timemax")
            mask_all_neg = (
                (ds["zsmax"] < -100)
                .where(ds["zsmax"].notnull(), True)
                .all(dim="timemax")
            )

            # Combined mask of good data
            mask = ~(mask_all_nan | mask_all_zero | mask_all_neg).compute()

            print(
                f"{np.sum(~mask).values} stations out of {mask.shape} have all NaN, zero, or negative values and will be skipped."
            )

            # Run through the grid faces and use POT to find maximums
            faces = ds["nmesh2d_face"].values
            nfaces = faces.size

            # Preallocate outputs
            exceedance = np.arange(num_exce)

            tzsmax_out = xr.DataArray(
                np.full((nfaces, num_exce), np.datetime64("NaT")),
                dims=("nmesh2d_face", "exceedance"),
                coords={"nmesh2d_face": faces, "exceedance": exceedance},
                name="tzsmax",
            )

            zsmax_out = xr.DataArray(
                np.full((nfaces, num_exce), np.nan, dtype=float),
                dims=("nmesh2d_face", "exceedance"),
                coords={"nmesh2d_face": faces, "exceedance": exceedance},
                name="zsmax",
            )

            threshold_out = xr.DataArray(
                np.full(nfaces, np.nan, dtype=float),
                dims=("nmesh2d_face",),
                coords={"nmesh2d_face": faces},
                name="zs_threshold",
            )

            # Run through the grid faces and use POT to find maximums
            for station in ds["nmesh2d_face"]:
                if mask.sel(nmesh2d_face=station):
                    pull = ds["zsmax"].sel(nmesh2d_face=station)

                    th = pot_threshold_set_num_xr(
                        pull,
                        r=r,
                        num_exce=num_exce,
                        time_dim="timemax",
                        strategy="closest",
                    )

                    threshold_out.loc[dict(nmesh2d_face=station)] = (
                        float(th.values) if hasattr(th, "values") else float(th)
                    )

                    extremes = get_extremes_pot_xr(
                        pull, th, r=r, time_dim="timemax", num_exce=num_exce
                    )

                    # k = number of extremes found (could be < num_exce)
                    k = min(num_exce, extremes.sizes.get("timemax", num_exce))

                    # Assign time + value into 2D arrays
                    tzsmax_out.loc[
                        dict(nmesh2d_face=station, exceedance=slice(0, k))
                    ] = extremes["timemax"].values[:k]
                    zsmax_out.loc[
                        dict(nmesh2d_face=station, exceedance=slice(0, k))
                    ] = extremes.values[:k]

            # --- Build Dataset ---
            ds_maxes = xr.Dataset(
                {
                    "tzsmax": tzsmax_out,  # datetime64, shape (nmesh2d_face, exceedance)
                    "zsmax": zsmax_out,  # float,       shape (nmesh2d_face, exceedance)
                    "zs_threshold": threshold_out,  # float,       shape (nmesh2d_face,)
                },
                coords={
                    "nmesh2d_face": faces,
                    "exceedance": exceedance,
                },
            )

            # --- dd other variables at the same exceedance times ---
            # We want values of other variables at the POT times for each face.
            if len(vars_to_keep) > 1:
                for var in vars_to_keep[1:]:
                    # Output array for this var aligned to (face, exceedance)
                    var_out = xr.DataArray(
                        np.full((nfaces, num_exce), np.nan, dtype=float),
                        dims=("nmesh2d_face", "exceedance"),
                        coords={"nmesh2d_face": faces, "exceedance": exceedance},
                        name=var,
                    )
                    # Fill per face using tzsmax_out times
                    for station in faces:
                        tvals = tzsmax_out.sel(
                            nmesh2d_face=station
                        ).values  # (num_exce,)
                        valid = ~np.isnat(tvals)
                        if not np.any(valid):
                            continue

                        # Select variable values at the extreme times for *this* face.
                        v_face = ds[var].sel(nmesh2d_face=station)
                        vsel = v_face.sel(
                            timemax=xr.DataArray(tvals[valid], dims=["exceedance"])
                        )

                        var_out.loc[
                            dict(nmesh2d_face=station, exceedance=np.where(valid)[0])
                        ] = vsel.values

                    ds_maxes[var] = var_out

            # --- Attributes---
            ds_maxes["tzsmax"].attrs.update(
                {
                    "long_name": "Times of top POT exceedances for zsmax",
                    "standard_name": "time",
                }
            )
            ds_maxes["zsmax"].attrs.update(
                {
                    "long_name": "Values of top POT exceedances for zsmax",
                    "units": "m",
                }
            )
            ds_maxes["zs_threshold"].attrs.update(
                {
                    "long_name": "POT threshold applied to zsmax per face",
                    "units": "m",
                }
            )
            ds_maxes.attrs.update(
                {
                    "title": "Peaks Over Threshold (POT) extremes per mesh face",
                    "source": "Computed from ds['zsmax'] using pot_threshold_set_num_xr & get_extremes_pot_xr",
                    "num_exceedances": num_exce,
                    "declustering_r": r,
                }
            )

            # Export in case we want this for later.
            ds_maxes.to_netcdf(
                os.path.join(destout, county, TMP_string, "POT_Maxes.nc")
            )

            # ===============================================================================
            # Get the specific return periods we want
            # ===============================================================================

            r_axis = rp_axis(num_exce, "weibull", "ascending")

            # Initialize  output
            zsmax_out = np.full((num_grid, len(return_period_wanted)), np.nan)

            tzsmax_out = np.full((num_grid, len(return_period_wanted)), np.nan)

            if include_qmax:
                qmax_out = np.full((num_grid, len(return_period_wanted)), np.nan)

            if include_tmax:
                tmax_out = np.full((num_grid, len(return_period_wanted)), np.nan)

            # Run through the grid faces and find RP values
            for station in range(num_grid):
                if mask.sel(nmesh2d_face=station):
                    ############ ZSMAX ############
                    # Pull values for this station (and fix nans)
                    zsmax = ds_maxes["zsmax"].sel(nmesh2d_face=station).values
                    zsmax[np.isnan(zsmax)] = zb[station]

                    # Sort zsmax and find RPs we want
                    ascending_indices = np.argsort(zsmax)
                    zsmax_wanted = zsmax[ascending_indices]

                    # Find the indices for the return periods we want
                    nearest_indices = np.full(
                        len(return_period_wanted), -999, dtype="int64"
                    )
                    for cnt, rp in enumerate(return_period_wanted):
                        idx = np.searchsorted(r_axis, rp, side="left")
                        if idx == 0:
                            nearest_idx = 0
                        elif idx == len(r_axis):
                            nearest_idx = len(r_axis) - 1
                        else:
                            prev_val = r_axis[idx - 1]
                            next_val = r_axis[idx]
                            if abs(rp - prev_val) <= abs(rp - next_val):
                                nearest_idx = idx - 1
                            else:
                                nearest_idx = idx
                        nearest_indices[cnt] = nearest_idx

                    r_axis_found = r_axis[nearest_indices]

                    # And now also interpolate with log
                    log_r_axis = np.log10(r_axis)  # Take the logarithm of x-axis values
                    log_r_axis[0] = 0.0  # Trick to get to yearly
                    log_return_period_wanted = np.log10(
                        return_period_wanted
                    )  # Take the logarithm of the desired return period

                    # Interpolate using log10 values
                    interpolated_value = np.interp(
                        log_return_period_wanted, log_r_axis, zsmax_wanted
                    )
                    zsmax_out[station, :] = interpolated_value

                    ############ TZSMAX ############
                    tzsmax = ds_maxes["tzsmax"].sel(nmesh2d_face=station).values
                    tzsmax[np.isnan(tzsmax)] = np.datetime64("NaT")
                    tzsmax_wanted = tzsmax[ascending_indices]
                    tzsmax_out[station, :] = tzsmax_wanted[nearest_indices]

                    ############ QMAX ############
                    if include_qmax:
                        qmax = ds_maxes["qmax"].sel(nmesh2d_face=station).values
                        qmax[np.isnan(qmax)] = 0.0
                        qmax_wanted = qmax[ascending_indices]
                        qmax_out[station, :] = qmax_wanted[nearest_indices]

                    ############ TMAX ############
                    if include_tmax:
                        tmax = ds_maxes["tmax"].sel(nmesh2d_face=station).values
                        tmax[np.isnan(tmax)] = 0.0

                        # Option 1 => sorting the indices
                        tmax_wanted = tmax[ascending_indices]

                        # Option 2 => cumulative sum
                        if np.min(tmax) > 0:
                            ascending_indices = np.argsort(tmax)
                            tmax_wanted = tmax[ascending_indices]

                        # Option 1 => find the nearest value
                        tmax_out[station, :] = tmax_wanted[nearest_indices]

                        # Option 2 => average duration in hours for the return period
                        # This means = the maximum that occurs on average once every 'rp' years
                        num_samples = 1000  # Number of Monte Carlo samples
                        for idx, rp in enumerate(return_period_wanted):
                            # Check if this is not zero
                            if np.max(tmax_wanted) > 0:
                                # Generate all random samples at once for better performance
                                sampled_indices = np.random.choice(
                                    tmax_wanted.shape[0], (num_samples, rp)
                                )

                                # Gather the sampled tmax values
                                sampled_tmax = tmax_wanted[
                                    sampled_indices
                                ]  # Shape: (num_samples, rp)

                                # Compute the maximum for each sample (axis=1)
                                sampled_max = np.nanmax(
                                    sampled_tmax, axis=1
                                )  # Shape: (num_samples,)

                                # Compute the mean of the sampled maxima
                                final_average = np.nanmean(sampled_max)

                                # Store the result in hours
                                tmax_out[station, idx] = (
                                    final_average / 3600
                                )  # Convert from seconds to hourseconds to hours

                            else:
                                tmax_out[station, :] = 0.0

                    # Compute a hmax critera => nice to look at intermediate results
                    hh_out = np.squeeze(zsmax_out[station, :]) - zb[station]
                    idfind = np.where(hh_out < hh_criteria)
                    zsmax_out[station, idfind] = np.nan

                    if include_qmax == 1:
                        qmax_out[station, idfind] = np.nan
                    if include_tmax == 1:
                        tmax_out[station, idfind] = np.nan

            # Done with this iteration (county and SLR): let's plot and save
            destout_TMP = os.path.join(destout, county, TMP_string)
            if not os.path.exists(destout_TMP):
                os.makedirs(destout_TMP)

            # Loop over return periods and make plots
            for t in range(zsmax_out.shape[1]):
                # Make figure
                fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

                # Water level
                p1 = axs[0, 0].scatter(
                    xc / 1000,
                    yc / 1000,
                    20,
                    np.squeeze(zsmax_out[:, t]),
                    cmap="viridis",
                )
                axs[0, 0].set_title("Water level")
                plt.colorbar(p1, ax=axs[0, 0])

                # Flow velocity
                if include_qmax == 1:
                    p2 = axs[0, 1].scatter(
                        xc / 1000, yc / 1000, np.squeeze(qmax_out[:, t]), cmap="Reds"
                    )
                    axs[0, 1].set_title("Velocity")
                    plt.colorbar(p2, ax=axs[0, 1])

                    # Duration
                    if include_tmax == 1:
                        p3 = axs[1, 0].scatter(
                            xc / 1000,
                            yc / 1000,
                            np.squeeze(tmax_out[:, t]),
                            cmap="GnBu",
                        )
                        axs[1, 0].set_title("Duration")
                        plt.colorbar(p3, ax=axs[1, 0])

                    # Bed level
                    p4 = axs[1, 1].scatter(
                        xc / 1000, yc / 1000, zb, cmap="terrain", vmin=-20, vmax=20
                    )
                    axs[1, 1].set_title("Bed level")
                    plt.colorbar(p4, ax=axs[1, 1])

                    # Set limits
                    p1.set_clim(vmin=0, vmax=10)
                    if include_qmax == 1:
                        p2.set_clim(vmin=0, vmax=5)
                    if include_tmax == 1:
                        p3.set_clim(vmin=0, vmax=86400)
                    p4.set_clim(vmin=-20, vmax=20)

                    # Print this
                    fname = "overview_" + str(return_period_wanted[t]) + "yr.png"
                    fname = os.path.join(destout_TMP, fname)
                    plt.savefig(fname, dpi="figure", format=None)
                    plt.close()

                    # Make one large figure for zsmax only
                    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
                    p1 = axs.pcolor(
                        xc / 1000,
                        yc / 1000,
                        np.squeeze(zsmax_out[:, t]),
                        cmap="viridis",
                    )
                    p1.set_clim(vmin=0, vmax=10)  # Set the color limits from 0 to 10
                    axs.set_title("Water level")
                    plt.colorbar(p1, ax=axs)
                    fname = "zsmax_only_" + str(return_period_wanted[t]) + "yr.png"
                    fname = os.path.join(destout_TMP, fname)
                    plt.savefig(fname, dpi="figure", format=None)
                    plt.close()

                # Create an xarray Dataset per return period
                for t in range(zsmax_out.shape[0]):
                    # Get coordinates ready
                    zsmax_out_now = np.squeeze(zsmax_out[:, t])

                    # Make base
                    ds_out = xr.Dataset()
                    coords = "nmesh2d_face"
                    ds_out.coords["nmesh2d_face"] = ds["zsmax"].coords["nmesh2d_face"]

                    ############ ZSMAX ############
                    ds_out["zsmax"] = (coords, np.float32(zsmax_out_now))

                    # Add more description
                    ds_out["zsmax"].attrs["units"] = "m"
                    ds_out["zsmax"].attrs["standard_name"] = (
                        "maximum of sea_surface_height_above_mean_sea_level"
                    )
                    ds_out["zsmax"].attrs["long_name"] = "maximum_water_level"
                    ds_out["zsmax"].attrs["coordinates"] = "nmesh2d_face"

                    ds_out["tmax_zs"] = (coords, np.squeeze(tmax_zs_out[:, t]))

                    # Optionally add qmax and tmax if include_qmax and include_tmax are set to 1
                    if include_qmax:
                        data_out_now = np.squeeze(qmax_out[t, :, :])
                        ds_out["qmax"] = (coords, np.float32(data_out_now))

                    if include_tmax:
                        data_out_now = np.squeeze(tmax_out[t, :, :])
                        ds_out["tmax"] = (coords, np.float32(data_out_now))

                    # Also add bed level and coordinates
                    ds_out["zb"] = (coords, np.float32(zb))
                    ds_out["x"] = (coords, xc)
                    ds_out["y"] = (coords, yc)

                    ds_out["zb"].attrs["units"] = "m"
                    ds_out["zb"].attrs["standard_name"] = "Bed_level"
                    ds_out["zb"].attrs["datum"] = "NAVD88"

                    ds_out["x"].attrs["units"] = "m"
                    ds_out["x"].attrs["standard_name"] = "projection_x_coordinate"
                    ds_out["x"].attrs["long_name"] = "x coordinate of projection"
                    ds_out["x"].attrs["Projection"] = "UTM zone 10N"

                    ds_out["y"].attrs["units"] = "m"
                    ds_out["y"].attrs["standard_name"] = "projection_y_coordinate"
                    ds_out["y"].attrs["long_name"] = "y coordinate of projection"
                    ds_out["y"].attrs["Projection"] = "UTM zone 10N"

                    ############ TZSMAX ############
                    ds_out["tmax_zs"] = (coords, np.squeeze(tmax_zs_out[:, t]))
                    ds_out["tmax_zs"].attrs["units"] = "date"
                    ds_out["tmax_zs"].attrs["standard_name"] = (
                        "time_of_maximum_sea_surface_height_above_mean_sea_level"
                    )
                    ds_out["tmax_zs"].attrs["Description"] = (
                        "Spatial time of occurence (to the day) of Return interval maximum water level"
                    )

                    # Add global attributes
                    ds_out.attrs["description"] = (
                        "NetCDF file with process SFINCS outputs"
                    )
                    ds_out.attrs["history"] = "Created " + datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    # Save the Dataset to a NetCDF file
                    filename = (
                        "processed_SFINCS_output_RP"
                        + "{:03}".format(int(return_period_wanted[t]))
                        + ".nc"
                    )
                    filename = os.path.join(destout_TMP, filename)
                    ds_out.to_netcdf(filename)

                    # Additionally write simple wet-dry interface
                    zsmax_out_now = np.squeeze(zsmax_out[:, t])
                    wet_dry = np.where(np.isnan(zsmax_out_now), 0, 1)
                    ds_out = xr.Dataset()
                    coords = "nmesh2d_face"
                    ds_out.coords["nmesh2d_face"] = ds["zsmax"].coords["nmesh2d_face"]

                    ds_out["wetdry"] = (coords, np.float32(wet_dry))
                    filename = (
                        "wetdry_SFINCS_output_RP"
                        + "{:03}".format(int(return_period_wanted[t]))
                        + ".nc"
                    )
                    filename = os.path.join(destout_TMP, filename)
                    ds_out.to_netcdf(filename)

            # handle the exception if the file cannot be read
            print(f"done with this iteration - {destout_TMP}", flush=True)

    # Do analysis on runtimes (how much done and stuff)
    a = 1

    # Done with this particular counties


# Done with the script
print("done!")
