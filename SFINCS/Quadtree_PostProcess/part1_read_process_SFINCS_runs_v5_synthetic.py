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

from POT_Extremes import  pot_threshold_set_num_xr, get_extremes_pot_xr
from Xarray_NCtools import batch_check_nc_files, ensure_unique_sorted_time,_detect_time_coord

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
coords_to_keep: Sequence[str] | None = ["timemax", "nmesh2d_face"]


# Some processing of the inputs
return_period_wanted = [int(RP) for RP in return_period_wanted]
sub_categories = [sub.replace("_median", "") for sub in sub_categories]



# User settings
# return_period_wanted    = [1, 2, 5, 10, 20, 50, 100]     # number of years requested
hh_criteria = 0.010001  # just above the treshold from SFINCS

# Directory
include_qmax = 1
include_tmax = 1
include_tmax_zs = 1


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
            required_vars = ["zsmax"]  # e.g., ["eta", "depth", "waterlevel"]
            required_coords = ["timemax"]  # e.g., ["time", "lat", "lon"]

            good_files, report, schema = batch_check_nc_files(
                files,
                required_vars=required_vars,
                required_coords=required_coords,
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

            # ===============================================================================
            # Calculate the maximums
            # ===============================================================================
                        

            r = "48h"   # Deculstering time window for POT (e.g., 24h, 48h, etc.)
            num_target = total_years
            
            tmax_out =np.empty((ds["nmesh2d_face"].size, num_target),dtype='datetime64[ns]')
            zmax_out = np.empty((ds["nmesh2d_face"].size, num_target),dtype= np.float32)
            
            for station in ds['nmesh2d_face']:
                pull = ds['zsmax'].sel(nmesh2d_face=station)

                if pull.isnull().all():
                    print(f"Station {station.values} has all NaN values; skipping POT analysis.")
                else:
                    th = pot_threshold_set_num_xr(pull, r=r, num_exce=total_years, time_dim="timemax", strategy="geq")
                    extremes = get_extremes_pot_xr(pull,th, r=r, time_dim="timemax")
                tmax_out[station,:] = extremes['timemax'].values
                zmax_out[station,:] = extremes.values

            
            adsf


            # Reading done
            # Make empty matrix
            if count_years == 0:
                zsmax_matrix = np.full(
                    (total_years, np.size(zsmax, 0), np.size(zsmax, 1)), np.nan
                )
                if include_qmax == 1:
                    qmax_matrix = np.full(
                        (total_years, np.size(zsmax, 0), np.size(zsmax, 1)), np.nan
                    )
                if include_tmax == 1:
                    tmax_matrix = np.full(
                        (total_years, np.size(zsmax, 0), np.size(zsmax, 1)), np.nan
                    )
                if include_tmax_zs == 1:
                    tmax_zs_matrix = np.full(
                        (total_years, np.size(zsmax, 0), np.size(zsmax, 1)), np.nan
                    )

                Ns, nx, ny = np.shape(zsmax_matrix)

                # Place files in the dictionary
                zsmax_matrix[count_years, :, :] = zsmax
                if include_qmax == 1:
                    qmax_matrix[count_years, :, :] = qmax
                if include_tmax == 1:
                    tmax_matrix[count_years, :, :] = tmax
                if include_tmax_zs == 1:
                    tmax_zs_matrix[count_years, :, :] = tmax_zs
                count_years = count_years + 1

            # NaN out all points that there
            id_delete = zsmax_matrix < -100
            zsmax_matrix[id_delete] = np.nan
            if include_qmax == 1:
                qmax_matrix[id_delete] = np.nan
            if include_tmax == 1:
                tmax_matrix[id_delete] = np.nan
            if include_tmax_zs == 1:
                tmax_zs_matrix[id_delete] = np.nan

            # TMP: determine maximum depth for different years
            hhmax_matrix = zsmax_matrix - zb
            hhmax = np.nanmax(hhmax_matrix, axis=1)
            hhmax = np.nanmax(hhmax, axis=1)
            non_nan_mask = np.isnan(hhmax)
            indices = np.where(non_nan_mask)[0]

            # Specify return period vector
            Ns, nx, ny = np.shape(zsmax_matrix)

            lambda_value = 1.0  # since we simulate individual years
            r_axis = np.zeros(Ns)
            for ii in range(1, Ns + 1):
                r_axis[ii - 1] = (Ns + 1) / ((Ns + 1 - ii) * lambda_value)

            # Prep output
            zsmax_out = np.full(
                (
                    len(return_period_wanted),
                    np.size(zsmax_matrix, 1),
                    np.size(zsmax_matrix, 2),
                ),
                np.nan,
            )
            if include_qmax == 1:
                qmax_out = np.full(
                    (
                        len(return_period_wanted),
                        np.size(zsmax_matrix, 1),
                        np.size(zsmax_matrix, 2),
                    ),
                    np.nan,
                )
            if include_tmax == 1:
                tmax_out = np.full(
                    (
                        len(return_period_wanted),
                        np.size(zsmax_matrix, 1),
                        np.size(zsmax_matrix, 2),
                    ),
                    np.nan,
                )
            if include_tmax_zs == 1:
                tmax_zs_out = np.full(
                    (
                        len(return_period_wanted),
                        np.size(zsmax_matrix, 1),
                        np.size(zsmax_matrix, 2),
                    ),
                    np.nan,
                )
            years_out = np.full(
                (
                    len(return_period_wanted),
                    np.size(zsmax_matrix, 1),
                    np.size(zsmax_matrix, 2),
                ),
                -999,
                dtype="int64",
            )

            # loop over the second and third dimensions
            for i in range(zsmax_matrix.shape[1]):
                for j in range(zsmax_matrix.shape[2]):
                    # access the value at (i, j)
                    zsmax = zsmax_matrix[:, i, j]
                    if include_qmax == 1:
                        qmax = qmax_matrix[:, i, j]
                    if include_tmax == 1:
                        tmax = tmax_matrix[:, i, j]
                    if include_tmax_zs == 1:
                        tmax_zs = tmax_zs_matrix[:, i, j]

                    # Count number of zeros
                    num_non_nans = np.count_nonzero(~np.isnan(zsmax))

                    # If we have more than 1 valid numbers we do this
                    if num_non_nans > 0:
                        # Is this a coastal point?
                        if zb[i, j] < 0:
                            a = 1

                        # Replace NaNs with 0 and zb
                        zsmax[np.isnan(zsmax)] = zb[i, j]
                        if include_qmax == 1:
                            qmax[np.isnan(qmax)] = 0.0
                        if include_tmax == 1:
                            tmax[np.isnan(tmax)] = 0.0
                        if include_tmax_zs == 1:
                            tmax_zs[np.isnan(tmax_zs)] = 0.0

                        # Sort zsmax and find RPs we want
                        ascending_indices = np.argsort(zsmax)
                        zsmax_wanted = zsmax[ascending_indices]
                        if include_qmax == 1:
                            qmax_wanted = qmax[ascending_indices]
                        if include_tmax == 1:
                            # Option 1 => sorting the indices
                            tmax_wanted = tmax[ascending_indices]

                            # Option 2 => cumulative sum
                            if np.min(tmax) > 0:
                                ascending_indices = np.argsort(tmax)
                                tmax_wanted = tmax[ascending_indices]

                        if include_tmax_zs == 1:
                            tmax_zs_wanted = tmax_zs[ascending_indices]

                        # Find the indices for the return periods we want
                        nearest_indices_old = np.searchsorted(
                            r_axis, return_period_wanted
                        )
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

                        r_axis_found1 = r_axis[nearest_indices_old]
                        r_axis_found2 = r_axis[nearest_indices]

                        # Save Output
                        zsmax_out[:, i, j] = zsmax_wanted[nearest_indices]
                        if include_qmax == 1:
                            qmax_out[:, i, j] = qmax_wanted[nearest_indices]
                        if include_tmax == 1:
                            # Option 1 => find the nearest value
                            tmax_out[:, i, j] = tmax_wanted[nearest_indices]

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
                                    tmax_out[idx, i, j] = (
                                        final_average / 3600
                                    )  # Convert from seconds to hourseconds to hours

                                else:
                                    tmax_out[:, i, j] = 0.0

                        if include_tmax_zs == 1:
                            tmax_zs_out[:, i, j] = tmax_zs_wanted[nearest_indices]

                        # Pull out the years corresopnding to the pulled maximums
                        sim_list = np.array(
                            [int(file.replace("SY", "")) for file in files]
                        )

                        # Sort like all the other output parameters
                        sim_list = sim_list[ascending_indices]

                        # Need to back out the correct indices before sorting.
                        years_out[:, i, j] = sim_list[nearest_indices]

                        # And now also interpolate with log
                        log_r_axis = np.log10(
                            r_axis
                        )  # Take the logarithm of x-axis values
                        log_r_axis[0] = 0.0  # Trick to get to yearly
                        log_return_period_wanted = np.log10(
                            return_period_wanted
                        )  # Take the logarithm of the desired return period

                        # Interpolate using log10 values
                        interpolated_value = np.interp(
                            log_return_period_wanted, log_r_axis, zsmax_wanted
                        )
                        zsmax_out[:, i, j] = interpolated_value

                        # Compute a hmax critera => nice to look at intermediate results
                        hh_out = np.squeeze(zsmax_out[:, i, j]) - zb[i, j]
                        idfind = np.where(hh_out < hh_criteria)
                        zsmax_out[idfind, i, j] = np.nan
                        if include_qmax == 1:
                            qmax_out[idfind, i, j] = np.nan
                        if include_tmax == 1:
                            tmax_out[idfind, i, j] = np.nan
                        if include_tmax_zs == 1:
                            tmax_zs_out[idfind, i, j] = np.nan

                        years_out[idfind, i, j] = -999

            # Done with this iteration (county and SLR): let's plot and save
            destout_TMP = os.path.join(destout, county, TMP_string)
            if not os.path.exists(destout_TMP):
                os.makedirs(destout_TMP)

            # Provide some feedback to on runtime
            estimate_runtimes_in_hours = [
                runtime / 3600 for runtime in estimate_runtimes
            ]
            plt.plot(estimate_runtimes_in_hours)
            plt.xlabel("simulations")
            plt.ylabel("run time [hr]")
            fname = "estimate_runtimes"
            fname = os.path.join(destout_TMP, fname)
            plt.savefig(fname, dpi="figure", format=None)
            plt.close()

            # Store run times in general for all SLRs
            matrix_runtimes[:, index] = estimate_runtimes_in_hours

            # Loop over return periods and make plots
            for t in range(zsmax_out.shape[0]):
                # Make figure
                fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

                # Water level
                p1 = axs[0, 0].pcolor(
                    x / 1000, y / 1000, np.squeeze(zsmax_out[t, :, :]), cmap="viridis"
                )
                axs[0, 0].set_title("Water level")
                plt.colorbar(p1, ax=axs[0, 0])

                # Flow velocity
                if include_qmax == 1:
                    p2 = axs[0, 1].pcolor(
                        x / 1000, y / 1000, np.squeeze(qmax_out[t, :, :]), cmap="Reds"
                    )
                    axs[0, 1].set_title("Velocity")
                    plt.colorbar(p2, ax=axs[0, 1])

                    # Duration
                    if include_tmax == 1:
                        p3 = axs[1, 0].pcolor(
                            x / 1000,
                            y / 1000,
                            np.squeeze(tmax_out[t, :, :]),
                            cmap="GnBu",
                        )
                        axs[1, 0].set_title("Duration")
                        plt.colorbar(p3, ax=axs[1, 0])

                    # Bed level
                    p4 = axs[1, 1].pcolor(
                        x / 1000, y / 1000, zb, cmap="terrain", vmin=-20, vmax=20
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
                        x / 1000,
                        y / 1000,
                        np.squeeze(zsmax_out[t, :, :]),
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
                    x_coord = x[1, :]
                    y_coord = y[:, 1]
                    zsmax_out_now = np.squeeze(zsmax_out[t, :, :])

                    # Make base
                    ds = xr.Dataset()
                    coords = ("y", "x")
                    ds["zsmax"] = (coords, np.float32(zsmax_out_now))
                    if read_binary == 1:
                        ds.coords["x"] = x_coord.squeeze()
                        ds.coords["y"] = y_coord.squeeze()
                    else:
                        ds.coords["x"] = x_coord.compressed()
                        ds.coords["y"] = y_coord.compressed()

                    # Get more description
                    ds["zsmax"].attrs["units"] = "m"
                    ds["zsmax"].attrs["standard_name"] = (
                        "maximum of sea_surface_height_above_mean_sea_level"
                    )
                    ds["zsmax"].attrs["long_name"] = "maximum_water_level"
                    ds["zsmax"].attrs["coordinates"] = "y x"

                    # Optionally add qmax and tmax if include_qmax and include_tmax are set to 1
                    if include_qmax == 1:
                        data_out_now = np.squeeze(qmax_out[t, :, :])
                        ds["qmax"] = (coords, np.float32(data_out_now))
                    if include_tmax == 1:
                        data_out_now = np.squeeze(tmax_out[t, :, :])
                        ds["tmax"] = (coords, np.float32(data_out_now))
                    if include_tmax_zs == 1:
                        data_out_now = np.squeeze(tmax_zs_out[t, :, :])
                        ds["tmax_zs"] = (coords, np.float32(data_out_now))

                    data_out_now = np.squeeze(years_out[t, :, :])
                    id_bad = np.isnan(zsmax_out_now)
                    data_out_now[id_bad] = -999

                    ds["year_max"] = (coords, data_out_now)
                    ds["year_max"].attrs["units"] = "Year (integer)"
                    ds["year_max"].attrs["_FillValue"] = "-999"
                    ds["year_max"].attrs["standard_name"] = (
                        "Simulation Year RP occurs (With file header removed)"
                    )
                    ds["year_max"].attrs["coordinates"] = "y x"

                    # Also add bed level
                    ds["zb"] = (coords, np.float32(zb))

                    # Add global attributes
                    ds.attrs["description"] = "NetCDF file with process SFINCS outputs"
                    ds.attrs["history"] = "Created " + datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    # Save the Dataset to a NetCDF file
                    filename = (
                        "processed_SFINCS_output_RP"
                        + "{:03}".format(int(return_period_wanted[t]))
                        + ".nc"
                    )
                    filename = os.path.join(destout_TMP, filename)
                    ds.to_netcdf(filename)

                    # Additionally write simple wet-dry interface
                    zsmax_out_now = np.squeeze(zsmax_out[t, :, :])
                    wet_dry = np.where(np.isnan(zsmax_out_now), 0, 1)
                    ds = xr.Dataset()
                    coords = ("y", "x")
                    if read_binary == 1:
                        ds.coords["x"] = x_coord.squeeze()
                        ds.coords["y"] = y_coord.squeeze()
                    else:
                        ds.coords["x"] = x_coord.compressed()
                        ds.coords["y"] = y_coord.compressed()
                    ds["wetdry"] = (coords, np.float32(wet_dry))
                    filename = (
                        "wetdry_SFINCS_output_RP"
                        + "{:03}".format(int(return_period_wanted[t]))
                        + ".nc"
                    )
                    filename = os.path.join(destout_TMP, filename)
                    ds.to_netcdf(filename)

            # handle the exception if the file cannot be read
            print(f"done with this iteration - {destout_TMP}", flush=True)

    # Do analysis on runtimes (how much done and stuff)
    a = 1

    # Done with this particular counties


# Done with the script
print("done!")
