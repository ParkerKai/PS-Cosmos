# -*- coding: utf-8 -*-
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

#===============================================================================
# Import Modules
#===============================================================================

# Modules needed
from ast import List
from importlib.resources import path
import os
import hydromt_sfincs
import netCDF4 as nc
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import xarray as xr
from hydromt_sfincs import SfincsModel, utils
import traceback 
from functools import partial
from typing import Optional, Sequence, Tuple, Literal

#===============================================================================
# User inputs 
#===============================================================================

destin                  = r'D:\Kai\SFINCS'
destout                 = r'C:\Users\kai\Documents\KaiRuns\PostProcess'

return_period_wanted    = ['1', '2', '5', '10', '20', '50', '100']
SLRs_formatted          = ['000']
counties                = ["03_Kitsap"]
sub_categories          = ['_median']   # ,'_low','_high'



# Choose the data variables you want to keep
# Define at module scope (important for Dask pickling when parallel=True)
#vars_to_keep: Sequence[str] = ["zsmax"]  
#coords_to_keep: Sequence[str] = ["timemax", "nmesh2d_face"]   
vars_to_keep = None
coords_to_keep = None

#Some processing of the inputs
return_period_wanted = [int(RP) for RP in return_period_wanted]
sub_categories = [sub.replace('_median', '') for sub in sub_categories]


# User settings
#return_period_wanted    = [1, 2, 5, 10, 20, 50, 100]     # number of years requested
hh_criteria             = 0.010001                       # just above the treshold from SFINCS

# Directory
include_qmax            = 1
include_tmax            = 1
include_tmax_zs         = 1


#===============================================================================
# Functions
#===============================================================================

def check_nc_file(
    path,
    required_vars=None,
    required_coords=None,
    check_time=True,
    sample_data=False,
    ):

    """
    Returns (ok, issues, summary) for a single NetCDF file.
    - ok: bool
    - issues: list of strings describing problems
    - summary: dict with lightweight metadata (dims, vars, coords, dtypes)
    """
    issues = []
    summary = {"path": path}

    try:
        # Avoid full load; just open the dataset lazily.
        # netcdf4 engine is robust for classic/NETCDF4 files; can switch to 'scipy' if needed.
        ds = xr.open_dataset(path, decode_cf=True, mask_and_scale=True, engine="netcdf4")
    except Exception as e:
        issues.append(f"OPEN_ERROR: {type(e).__name__}: {e}")
        return False, issues, summary

    # Collect summary info
    summary["dims"] = dict(ds.sizes)
    summary["coords"] = list(ds.coords)
    summary["vars"] = [v for v in ds.data_vars]
    summary["global_attrs"] = dict(ds.attrs)

    # Dtypes snapshot (no compute)
    dtypes = {}
    for v in ds.variables:
        try:
            dtypes[v] = str(ds[v].dtype)
        except Exception:
            dtypes[v] = "UNKNOWN"
    summary["dtypes"] = dtypes

    # 1) Dimensions non-empty
    for d, n in ds.sizes.items():
        if n <= 0:
            issues.append(f"DIM_EMPTY: dimension {d} has size {n}")

    # 2) Required variables/coords
    if required_vars:
        for v in required_vars:
            if v not in ds.variables:
                issues.append(f"MISSING_VAR: {v}")
    if required_coords:
        for c in required_coords:
            if c not in ds.coords:
                issues.append(f"MISSING_COORD: {c}")

    # 3) Time checks
    if check_time:
        time_name = None
        # Common time names
        for nm in ["time", "t", "Time"]:
            if nm in ds.coords:
                time_name = nm
                break
        if time_name:
            try:
                t = ds[time_name].values  # NumPy array (may be datetime64)
                if np.isnan(t).any():
                    issues.append(f"TIME_NAN: {time_name} contains NaNs")
                # Monotonic increasing check (allow equal for multi-file splits)
                if np.any(np.diff(t) < np.timedelta64(0, 'ns')) if np.issubdtype(t.dtype, np.datetime64) else np.any(np.diff(t) < 0):
                    issues.append(f"TIME_NON_MONOTONIC: {time_name} not strictly increasing")
            except Exception as e:
                issues.append(f"TIME_READ_ERROR: {time_name} {type(e).__name__}: {e}")

    # 4) Sample small data slices (optional, can be costly)
    if sample_data:
        try:
            for v in list(ds.data_vars)[:5]:  # sample up to 5 variables
                # Take a tiny slice to trigger decode/scale without loading full
                slc = {d: 0 for d in ds[v].dims}
                _ = ds[v].isel(**slc).values
        except Exception as e:
            issues.append(f"DATA_SAMPLE_ERROR: {type(e).__name__}: {e}")

    # 5) Minimal attribute & encoding sanity
    for v in ds.variables:
        enc = ds[v].encoding
        # Known source of conflicts in open_mfdataset: variable attrs differ across files
        # We just record presence; comparison across files happens later.
        if "_FillValue" in enc and enc["_FillValue"] is None:
            issues.append(f"ENCODING_WARN: {v} has null _FillValue")

    # Clean up file handle
    ds.close()

    ok = len(issues) == 0
    return ok, issues, summary


def batch_check_nc_files(
    files,
    required_vars=None,
    required_coords=None,
    check_time=True,
    sample_data=False,
    compare_coords=("time", "lat", "lon"),  # coordinates to compare shape/attrs across files
):
    """
    Checks files and returns:
    - good_files: list of valid file paths
    - report: list of per-file records (path, ok, issues)
    - schema_ref: snapshot from first good file (dims, dtypes, coord shapes)
    """
    report = []
    good_files = []
    schema_ref = {"dims": None, "dtypes": None, "coord_shapes": {}}

    # First pass: individual checks
    per_file_summaries = {}
    for p in files:
        ok, issues, summary = check_nc_file(
            p,
            required_vars=required_vars,
            required_coords=required_coords,
            check_time=check_time,
            sample_data=sample_data,
        )
        report.append({"path": p, "ok": ok, "issues": issues})
        per_file_summaries[p] = summary
        if ok:
            good_files.append(p)

    # Establish schema reference from first good file
    if good_files:
        ref = per_file_summaries[good_files[0]]
        schema_ref["dims"] = ref["dims"]
        schema_ref["dtypes"] = ref["dtypes"]
        # Record coordinate shapes
        coord_shapes = {}
        for c in compare_coords:
            if c in ref["dims"]:
                coord_shapes[c] = ref["dims"][c]
            elif c in ref["coords"]:
                # Some coords are 1D not in dims; try len
                coord_shapes[c] = None
        schema_ref["coord_shapes"] = coord_shapes

    # Second pass: cross-file schema/coord consistency
    for p in good_files:
        ref = schema_ref
        summ = per_file_summaries[p]

        # Compare key coordinate dimension sizes (lightweight)
        for c, sz in ref["coord_shapes"].items():
            if c in summ["dims"] and sz is not None:
                if summ["dims"][c] != sz:
                    # Not necessarily fatal (concat may differ), but flag it
                    for r in report:
                        if r["path"] == p:
                            r["issues"].append(f"COORD_SIZE_DIFF: {c}={summ['dims'][c]} (ref {sz})")
                            break

        # Compare dtypes for variables present in both
        ref_dtypes = ref["dtypes"]
        for v, dt in summ["dtypes"].items():
            if v in ref_dtypes and dt != ref_dtypes[v]:
                for r in report:
                    if r["path"] == p:
                        r["issues"].append(f"DTYPE_DIFF: var {v} dtype {dt} (ref {ref_dtypes[v]})")
                        break

    # Recompute good_files to exclude anything with issues after cross-check
    final_good = [r["path"] for r in report if r["ok"] and len(r["issues"]) == 0]

    return final_good, report, schema_ref



def _detect_time_coord(ds: xr.Dataset,
                       preferred: Optional[str] = None) -> Optional[str]:
    """Return the name of the time coordinate if found, else None."""
    
    # Candidate names for the time coordinate; we'll auto-detect the first that exists.
    TIME_CANDIDATES: List[str] = ["time", "Time", "t", "datetime", "date"]

 
    if preferred and preferred in ds.coords:
        return preferred
    for name in TIME_CANDIDATES:
        if name in ds.coords:
            return name
    return None

def preprocess_(
    ds: xr.Dataset,
    year: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    time_name: Optional[str] = None,
    mode_year_if_unspecified: bool = True,
    tie_break: Literal['earliest', 'latest'] = 'earliest',
) -> xr.Dataset:
    """
    Defensive preprocessing:
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
    - Assumes `vars_to_keep` and `coords_to_keep` exist in the outer scope (as in your original).
    - Works with both numpy datetime64 and cftime calendars via `.dt.year`.
    """
    # --- 1) Subset variables and coords defensively (unchanged from your style) ---
    if vars_to_keep is not None:
        keep_vars = [v for v in vars_to_keep if v in ds.data_vars]  # noqa: F821 (assumes global list)
    if coords_to_keep is not None:
        keep_coords = [c for c in coords_to_keep if c in ds.coords]  # noqa: F821 (assumes global list)

    if keep_vars is not None:
        if keep_coords is not None:
            ds = ds[keep_coords]
        else:
            ds = ds.drop_vars(list(ds.data_vars))
    else:
        ds = ds[keep_vars + keep_coords]

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
        mask = (time_da.dt.year == year)
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
            candidates = counts['year'].where(counts == max_count, drop=True)
            if candidates.size == 0:
                return ds  # Safety: shouldn't happen, but avoid errors
            if tie_break == 'latest':
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
            mode_year_val = int(candidates.max() if tie_break == 'latest' else candidates.min())

        # Apply trim
        ds = ds.where(years == mode_year_val, drop=True)
        return ds

    # No trimming requested
    return ds

#===============================================================================
# Load the data 
#===============================================================================

# Go to folder and loop over domains
for county in counties:

    # Start with this county first
    print(f'Started with {county}',flush=True)
    destin_TMP          = os.path.join(destin, county)

    # Go over SLRs
    for index, slr in enumerate(SLRs_formatted):

        # Go over sub_categories (standard, low and high)
        for index_cat, category in enumerate(sub_categories):
            
            # Make print statement
            print(f" => SLR value: {slr} - {sub_categories[index_cat]}",flush=True)
            destin_TMP          = os.path.join(destin, county)
            TMP_string          = slr + category
            destin_TMP          = os.path.join(destin_TMP, TMP_string)

            # Get list of files for the run 
            dirs = [entry.name for entry in os.scandir(destin_TMP) if entry.is_dir()]
            files = [os.path.join(destin_TMP, dir_pull, 'sfincs_map.nc') for dir_pull in dirs]

            
            # Check the files before we try to combine them.
            required_vars = ['zsmax']  # e.g., ["eta", "depth", "waterlevel"]
            required_coords = ['timemax']  # e.g., ["time", "lat", "lon"]

            good_files, report, schema = batch_check_nc_files(
                files,
                required_vars=required_vars,
                required_coords=required_coords,
                check_time=True,
                sample_data=False,          # set True if you want to trigger decode/scale on tiny slices
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
            count_years         = 0
            total_years         = len(files)
            estimate_runtimes   = []

            # Automatically trim to the most common year (mode) if you don't pass year/start/end.
            # If multiple years tie (same count), choose the earliest; set tie_break='latest' to prefer the latest.
            preprocess_trim = partial(
                preprocess_,                  # your updated function
                mode_year_if_unspecified=True,
                tie_break='earliest',         # or 'latest'
                # Optional: force the time coordinate name if you know it
                # time_name="time",
            )

            ds = xr.open_mfdataset(
                files,
                combine="nested",
                concat_dim="timemax",          # use the name you detected (e.g., 'timemax')
                compat="no_conflicts",
                coords="minimal",
                data_vars="all",               # or 'minimal' if variables identical
                parallel=True,
                decode_cf=True,
                preprocess=preprocess_trim,    # you can keep trimming/sanitizing
                engine="netcdf4",
                chunks = {'time': 512, 'nmesh2d_face': 8192},
            )



            print(directory_path)  # Output: /home/user/documents

            # Define matrix for runtimes
            if index == 0: matrix_runtimes = np.full((total_years, len(SLRs_formatted)), np.nan)
            
            for file in files:
                
                # List SFINCS map
                sfincs_map = os.path.join(destin_TMP, file, 'sfincs_map.nc')

                # Read netcdf 
                try:

                    # Open the netcdf file
                    #print(file)
                    ds = xr.open_dataset(sfincs_map)

                    # read the variables of interest and process the
                    zsmax   = ds["zsmax"][:]
                    zsmax   = np.squeeze(zsmax,0)
                    if include_qmax == 1:
                        qmax    = ds["qmax"][:]
                        qmax    = np.squeeze(qmax,0)
                    if include_tmax == 1:
                        tmax    = ds["tmax"][:]
                        tmax    = np.squeeze(tmax,0)
                    if include_tmax_zs == 1:
                        tmax_zs    = ds["tmax_zs"][:]
                        tmax_zs    = np.squeeze(tmax_zs,0)

                    # Also read the grid and bed levels
                    x       = dataset.variables["x"][:]
                    y       = dataset.variables["y"][:]
                    zb      = dataset.variables["zb"][:]

                    # Read runtime
                    total_runtime = dataset.variables["total_runtime"][:].data[0]
                    if (total_runtime > 86400*7):
                        print(" => simulation not finished - " + file)
                        total_runtime = np.nan
                    estimate_runtimes.append(total_runtime)

                    # Close dataset
                    dataset.close()

                except:

                    # handle the exception if the file cannot be read
                    total_runtime = np.nan
                    estimate_runtimes.append(total_runtime)
                    print(" => cannot read this netcdf: " + sfincs_map)
                
                # Reading done
                # Make empty matrix
                if count_years == 0:
                    zsmax_matrix        = np.full((total_years, np.size(zsmax, 0), np.size(zsmax, 1)), np.nan)
                    if include_qmax == 1:
                        qmax_matrix     = np.full((total_years, np.size(zsmax, 0), np.size(zsmax, 1)), np.nan)
                    if include_tmax == 1:
                        tmax_matrix     = np.full((total_years, np.size(zsmax, 0), np.size(zsmax, 1)), np.nan)
                    if include_tmax_zs == 1:
                        tmax_zs_matrix  = np.full((total_years, np.size(zsmax, 0), np.size(zsmax, 1)), np.nan)

                Ns, nx,ny       = np.shape(zsmax_matrix)
                
                # Place files in the dictionary 
                zsmax_matrix[count_years,:,:]       = zsmax
                if include_qmax == 1:
                    qmax_matrix[count_years,:,:]    = qmax
                if include_tmax == 1:
                    tmax_matrix[count_years,:,:]    = tmax
                if include_tmax_zs == 1:
                    tmax_zs_matrix[count_years,:,:] = tmax_zs
                count_years = count_years+1


            # NaN out all points that there 
            id_delete                       = zsmax_matrix < -100 
            zsmax_matrix[id_delete]         = np.nan
            if include_qmax == 1:
                qmax_matrix[id_delete]      = np.nan
            if include_tmax == 1:
                tmax_matrix[id_delete]      = np.nan
            if include_tmax_zs == 1:
                tmax_zs_matrix[id_delete]   = np.nan


            # TMP: determine maximum depth for different years
            hhmax_matrix = zsmax_matrix - zb
            hhmax        = np.nanmax(hhmax_matrix, axis=1)
            hhmax        = np.nanmax(hhmax, axis=1)
            non_nan_mask = np.isnan(hhmax)
            indices      = np.where(non_nan_mask)[0]

            # Specify return period vector
            Ns, nx,ny       = np.shape(zsmax_matrix)
           
            lambda_value    = 1.0           # since we simulate individual years
            r_axis          = np.zeros(Ns)
            for ii in range(1, Ns+1):
                r_axis[ii-1] = (Ns+1) / ((Ns+1-ii) * lambda_value)

            # Prep output
            zsmax_out       = np.full((len(return_period_wanted), np.size(zsmax_matrix, 1), np.size(zsmax_matrix, 2)), np.nan)
            if include_qmax == 1:
                qmax_out        = np.full((len(return_period_wanted), np.size(zsmax_matrix, 1), np.size(zsmax_matrix, 2)), np.nan)
            if include_tmax == 1:
                tmax_out        = np.full((len(return_period_wanted), np.size(zsmax_matrix, 1), np.size(zsmax_matrix, 2)), np.nan)
            if include_tmax_zs == 1:
                tmax_zs_out        = np.full((len(return_period_wanted), np.size(zsmax_matrix, 1), np.size(zsmax_matrix, 2)), np.nan)
            years_out = np.full((len(return_period_wanted), np.size(zsmax_matrix, 1), np.size(zsmax_matrix, 2)),-999, dtype='int64')

            # loop over the second and third dimensions
            for i in range(zsmax_matrix.shape[1]):
                for j in range(zsmax_matrix.shape[2]):

                    # access the value at (i, j)
                    zsmax           = zsmax_matrix[:, i, j]
                    if include_qmax == 1:
                        qmax            = qmax_matrix[:, i, j]
                    if include_tmax == 1:
                        tmax            = tmax_matrix[:, i, j]
                    if include_tmax_zs == 1:
                        tmax_zs            = tmax_zs_matrix[:, i, j]

                    # Count number of zeros
                    num_non_nans    = np.count_nonzero(~np.isnan(zsmax))

                    # If we have more than 1 valid numbers we do this
                    if num_non_nans > 0:

                        # Is this a coastal point?
                        if zb[i,j] < 0:
                            a=1

                        # Replace NaNs with 0 and zb
                        zsmax[np.isnan(zsmax)] = zb[i,j]
                        if include_qmax == 1:
                            qmax[np.isnan(qmax)]   = 0.0
                        if include_tmax == 1:
                            tmax[np.isnan(tmax)]   = 0.0
                        if include_tmax_zs == 1:
                            tmax_zs[np.isnan(tmax_zs)]   = 0.0

                        # Sort zsmax and find RPs we want
                        ascending_indices   = np.argsort(zsmax)
                        zsmax_wanted        = zsmax[ascending_indices]
                        if include_qmax == 1:
                            qmax_wanted         = qmax[ascending_indices]
                        if include_tmax == 1:

                            # Option 1 => sorting the indices
                            tmax_wanted         = tmax[ascending_indices]

                            # Option 2 => cumulative sum
                            if np.min(tmax) > 0:
                                ascending_indices   = np.argsort(tmax)
                                tmax_wanted         = tmax[ascending_indices]
                            
                        if include_tmax_zs == 1:
                            tmax_zs_wanted         = tmax_zs[ascending_indices]
                        
                        # Find the indices for the return periods we want
                        nearest_indices_old    = np.searchsorted(r_axis, return_period_wanted)
                        nearest_indices         = np.full(len(return_period_wanted),-999,dtype='int64')
                        for cnt,rp in enumerate(return_period_wanted):
                            idx = np.searchsorted(r_axis, rp, side='left')
                            if idx == 0:
                                nearest_idx = 0
                            elif idx == len(r_axis):
                                nearest_idx =len(r_axis) - 1
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
                        zsmax_out[:,i,j]    = zsmax_wanted[nearest_indices]
                        if include_qmax == 1:
                            qmax_out[:,i,j]     = qmax_wanted[nearest_indices]
                        if include_tmax == 1:

                            # Option 1 => find the nearest value
                            tmax_out[:,i,j]     = tmax_wanted[nearest_indices]

                            # Option 2 => average duration in hours for the return period
                            # This means = the maximum that occurs on average once every 'rp' years
                            num_samples = 1000  # Number of Monte Carlo samples
                            for idx, rp in enumerate(return_period_wanted):

                                # Check if this is not zero
                                if np.max(tmax_wanted) > 0:

                                    # Generate all random samples at once for better performance
                                    sampled_indices = np.random.choice(tmax_wanted.shape[0], (num_samples, rp))

                                    # Gather the sampled tmax values
                                    sampled_tmax    = tmax_wanted[sampled_indices]  # Shape: (num_samples, rp)

                                    # Compute the maximum for each sample (axis=1)
                                    sampled_max     = np.nanmax(sampled_tmax, axis=1)  # Shape: (num_samples,)

                                    # Compute the mean of the sampled maxima
                                    final_average   = np.nanmean(sampled_max)

                                    # Store the result in hours
                                    tmax_out[idx, i, j] = final_average / 3600  # Convert from seconds to hourseconds to hours

                                else:
                                    tmax_out[:, i, j]   = 0.0
                                    
                        if include_tmax_zs == 1:
                            tmax_zs_out[:,i,j]     = tmax_zs_wanted[nearest_indices]
                        
                        # Pull out the years corresopnding to the pulled maximums
                        sim_list = np.array([int(file.replace('SY','')) for file in files])
                        
                        # Sort like all the other output parameters 
                        sim_list = sim_list[ascending_indices]
                        
                        # Need to back out the correct indices before sorting.
                        years_out[:,i,j] = sim_list[nearest_indices]
                        
                        # And now also interpolate with log
                        log_r_axis                  = np.log10(r_axis)                  # Take the logarithm of x-axis values
                        log_r_axis[0]               = 0.0                               # Trick to get to yearly
                        log_return_period_wanted    = np.log10(return_period_wanted)    # Take the logarithm of the desired return period

                        # Interpolate using log10 values
                        interpolated_value          = np.interp(log_return_period_wanted, log_r_axis, zsmax_wanted)
                        zsmax_out[:,i,j]            = interpolated_value

                        # Compute a hmax critera => nice to look at intermediate results
                        hh_out                      = np.squeeze(zsmax_out[:,i,j]) - zb[i,j]
                        idfind                      = np.where(hh_out< hh_criteria)
                        zsmax_out[idfind,i,j]       = np.nan
                        if include_qmax == 1:
                            qmax_out[idfind,i,j]        = np.nan
                        if include_tmax == 1:
                            tmax_out[idfind,i,j]        = np.nan
                        if include_tmax_zs == 1:
                            tmax_zs_out[idfind,i,j]     = np.nan
                            
                        years_out[idfind,i,j]           = -999
                        
                        
            # Done with this iteration (county and SLR): let's plot and save
            destout_TMP          = os.path.join(destout, county, TMP_string)
            if not os.path.exists(destout_TMP):
                os.makedirs(destout_TMP)

            # Provide some feedback to on runtime
            estimate_runtimes_in_hours = [runtime / 3600 for runtime in estimate_runtimes]
            plt.plot(estimate_runtimes_in_hours)
            plt.xlabel('simulations')
            plt.ylabel('run time [hr]')
            fname = 'estimate_runtimes'
            fname = os.path.join(destout_TMP, fname)
            plt.savefig(fname, dpi='figure', format=None)
            plt.close()

            # Store run times in general for all SLRs
            matrix_runtimes[:,index] = estimate_runtimes_in_hours

            # Loop over return periods and make plots
            for t in range(zsmax_out.shape[0]):

                # Make figure
                fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

                # Water level
                p1 = axs[0, 0].pcolor(x/1000, y/1000, np.squeeze(zsmax_out[t,:,:]) , cmap='viridis')
                axs[0, 0].set_title("Water level")
                plt.colorbar(p1, ax=axs[0, 0])

                # Flow velocity
                if include_qmax == 1:
                    p2 = axs[0, 1].pcolor(x/1000, y/1000, np.squeeze(qmax_out[t,:,:]) , cmap='Reds')
                    axs[0, 1].set_title("Velocity")
                    plt.colorbar(p2, ax=axs[0, 1])

                    # Duration
                    if include_tmax == 1:
                        p3 = axs[1, 0].pcolor(x/1000, y/1000, np.squeeze(tmax_out[t,:,:]) , cmap='GnBu')
                        axs[1, 0].set_title("Duration")
                        plt.colorbar(p3, ax=axs[1, 0])

                    # Bed level
                    p4 = axs[1, 1].pcolor(x/1000, y/1000, zb , cmap='terrain', vmin=-20, vmax=20)
                    axs[1, 1].set_title("Bed level")
                    plt.colorbar(p4, ax=axs[1, 1])

                    # Set limits
                    p1.set_clim(vmin=0, vmax=10)  
                    if include_qmax == 1:p2.set_clim(vmin=0, vmax=5)  
                    if include_tmax == 1: p3.set_clim(vmin=0, vmax=86400)  
                    p4.set_clim(vmin=-20, vmax=20)  

                    # Print this
                    fname = 'overview_' + str(return_period_wanted[t]) + 'yr.png'
                    fname = os.path.join(destout_TMP, fname)
                    plt.savefig(fname, dpi='figure', format=None)
                    plt.close()

                    # Make one large figure for zsmax only
                    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
                    p1 = axs.pcolor(x/1000, y/1000, np.squeeze(zsmax_out[t,:,:]) , cmap='viridis')
                    p1.set_clim(vmin=0, vmax=10)  # Set the color limits from 0 to 10
                    axs.set_title("Water level")
                    plt.colorbar(p1, ax=axs)
                    fname = 'zsmax_only_' + str(return_period_wanted[t]) + 'yr.png'
                    fname = os.path.join(destout_TMP, fname)
                    plt.savefig(fname, dpi='figure', format=None)
                    plt.close()
                

                # Create an xarray Dataset per return period
                for t in range(zsmax_out.shape[0]):

                    # Get coordinates ready
                    x_coord         = x[1, :]
                    y_coord         = y[:, 1]
                    zsmax_out_now   = np.squeeze(zsmax_out[t, :, :])    

                    # Make base
                    ds              = xr.Dataset()
                    coords          = ('y', 'x')
                    ds['zsmax']     = (coords, np.float32(zsmax_out_now))
                    if read_binary == 1:
                        ds.coords['x']  = x_coord.squeeze()
                        ds.coords['y']  = y_coord.squeeze()
                    else:
                        ds.coords['x']  = x_coord.compressed()
                        ds.coords['y']  = y_coord.compressed()

                    # Get more description
                    ds['zsmax'].attrs['units']          = 'm'
                    ds['zsmax'].attrs['standard_name']  = 'maximum of sea_surface_height_above_mean_sea_level'
                    ds['zsmax'].attrs['long_name']      = 'maximum_water_level'
                    ds['zsmax'].attrs['coordinates']    = 'y x'

                    # Optionally add qmax and tmax if include_qmax and include_tmax are set to 1
                    if include_qmax == 1: 
                        data_out_now   = np.squeeze(qmax_out[t, :, :])    
                        ds['qmax']      = (coords, np.float32(data_out_now))
                    if include_tmax == 1: 
                        data_out_now   = np.squeeze(tmax_out[t, :, :])    
                        ds['tmax']     = (coords, np.float32(data_out_now))
                    if include_tmax_zs == 1:
                        data_out_now   = np.squeeze(tmax_zs_out[t, :, :])
                        ds['tmax_zs']      = (coords, np.float32(data_out_now))
                    
                    data_out_now   = np.squeeze(years_out[t, :, :])
                    id_bad                      = np.isnan(zsmax_out_now)
                    data_out_now[id_bad]       = -999
                    
                    ds['year_max']     = (coords, data_out_now)
                    ds['year_max'].attrs['units']          = 'Year (integer)'
                    ds['year_max'].attrs['_FillValue']     = '-999'
                    ds['year_max'].attrs['standard_name']  = 'Simulation Year RP occurs (With file header removed)'
                    ds['year_max'].attrs['coordinates']    = 'y x'  

                    # Also add bed level
                    ds['zb']     = (coords, np.float32(zb))

                    # Add global attributes
                    ds.attrs["description"] = "NetCDF file with process SFINCS outputs"
                    ds.attrs["history"] = "Created " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Save the Dataset to a NetCDF file
                    filename = 'processed_SFINCS_output_RP' +  "{:03}".format(int(return_period_wanted[t]))  + '.nc'
                    filename = os.path.join(destout_TMP, filename)
                    ds.to_netcdf(filename)

                    # Additionally write simple wet-dry interface
                    zsmax_out_now   = np.squeeze(zsmax_out[t, :, :])    
                    wet_dry         = np.where(np.isnan(zsmax_out_now), 0, 1)
                    ds              = xr.Dataset()
                    coords          = ('y', 'x')
                    if read_binary == 1:
                        ds.coords['x']  = x_coord.squeeze()
                        ds.coords['y']  = y_coord.squeeze()
                    else:
                        ds.coords['x']  = x_coord.compressed()
                        ds.coords['y']  = y_coord.compressed()
                    ds['wetdry']    = (coords, np.float32(wet_dry))
                    filename = 'wetdry_SFINCS_output_RP' +  "{:03}".format(int(return_period_wanted[t]))  + '.nc'
                    filename = os.path.join(destout_TMP, filename)
                    ds.to_netcdf(filename)

            # handle the exception if the file cannot be read
            print(f"done with this iteration - {destout_TMP}",flush=True)
    
    # Do analysis on runtimes (how much done and stuff)
    a=1

    # Done with this particular counties


# Done with the script
print('done!')
