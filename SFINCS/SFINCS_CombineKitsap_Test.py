# -*- coding: utf-8 -*-
"""
Created on February 16th, 2026

This script combines SFINCS run years and calculates extremes

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
import xarray as xr
import re
import numpy as np
import cftime
from hydromt_sfincs import SfincsModel
import sys
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
import rasterio
from rasterio.transform import from_origin
from scipy.stats import binned_statistic_2d

# ===============================================================================
# %% User Defined inputs
# ===============================================================================
SLR = "000"

dir_in = r"D:\Kai\SFINCS\2026-02-06-Kitsap"
file_DEM = r"D:\Kai\SFINCS\Kitsap_DEM\PS_Coned_Kitsap.tif"
dir_out = r"D:\Kai\SFINCS\2026-02-06-Kitsap_Out"

time_var = "timemax"  # time coordinate name


# ===============================================================================
# %% Functions
# ===============================================================================


sys.path.append(r"C:\Users\kai\Documents\Python Scripts\SFINCS")
from SFINCS_QuadtreeTools import load_SfincsQuadtree


def parse_year_from_path(path):
    """
    Extract the main year from the **parent directory** name of the file.
    Expected layout: .../<YEAR>/sfincs_map.nc
    Falls back to scanning the whole path if parent isn't numeric.
    """
    parent = os.path.basename(os.path.dirname(path))
    # Prefer parent folder if it's exactly a 4-digit year
    if re.fullmatch(r"\d{4}", parent):
        return int(parent)

    # Fallback: find the last 4-digit year in the path (1900–2099)
    years = re.findall(r"(?:^|[^0-9])((?:19|20)\d{2})(?=[^0-9]|$)", path)
    if years:
        return int(years[-1])

    raise ValueError(f"Could not infer year from path: {path}")


def start_end_for_calendar(year, cal):
    """
    Return (start, end) cftime datetimes for the full year, inclusive.
    Handles common CF calendars used in ocean/climate models.
    """
    cal = (cal or "standard").lower()
    if cal in ("standard", "gregorian"):
        dt = cftime.DatetimeGregorian
    elif cal in ("proleptic_gregorian",):
        dt = cftime.DatetimeProlepticGregorian
    elif cal in ("julian",):
        dt = cftime.DatetimeJulian
    elif cal in ("noleap", "365_day"):
        dt = cftime.DatetimeNoLeap
    elif cal in ("all_leap", "366_day"):
        dt = cftime.DatetimeAllLeap
    elif cal in ("360_day",):
        dt = cftime.Datetime360Day
    else:
        # Fallback to Gregorian if calendar is unknown/missing
        dt = cftime.DatetimeGregorian

    # For daily data, this inclusive range generally works well.
    # If your daily timestamps represent end-of-day (e.g., 23:59), this still matches.
    start = dt(year, 1, 1, 0, 0, 0)
    end = dt(year, 12, 31, 23, 59, 59)
    return start, end


def make_preprocess(time_var="timemax", var_select="zsmax"):
    """
    Build a preprocess function that trims each dataset to its year,
    inferred from the **file path** attached in ds.encoding['source'].
    """

    def _pre(ds):
        src = ds.encoding.get("source", "")
        yr = parse_year_from_path(src)

        # Detect calendar from the time coordinate (if present)
        cal = ds[time_var].attrs.get("calendar", "standard")

        start, end = start_end_for_calendar(yr, cal)

        # Trim spinup (from previous year) and any spillover beyond this year
        # Also subeset to variable of interest
        ds = ds[var_select].sel({time_var: slice(start, end)})

        # Ensure monotonic ascending time for clean concatenation
        ds = ds.sortby(time_var)

        return ds

    return _pre


# ---------- MAIN CONCAT ----------
def concat_sfincs_years(
    paths,
    time_var="timemax",
    var_select="zsmax",
    chunks=None,  # e.g., {'timemax': 90, 'y': 400, 'x': 400}
    combine="by_coords",
    parallel=True,
):
    """
    Open, trim each file to its calendar year, and combine along time.
    """
    ds = xr.open_mfdataset(
        paths,
        combine=combine,
        preprocess=make_preprocess(time_var=time_var, var_select=var_select),
        decode_times=True,
        use_cftime=True,  # important for non-standard calendars
        chunks=chunks,
        parallel=parallel,
    )
    # Safety: sort time and drop exact duplicates if they somehow remain
    ds = ds.sortby(time_var)

    # Optional: drop exact duplicate timestamps (should be rare with proper trimming)
    times = ds[time_var].values
    unique, counts = np.unique(times, return_counts=True)
    if (counts > 1).any():
        import pandas as pd

        idx = pd.Index(times)  # object dtype works for cftime
        keep = ~idx.duplicated()
        ds = ds.isel({time_var: keep})

    return ds


def scattered_to_geotiff_interpolate(
    x,
    y,
    z,
    epsg=32610,  # UTM Zone 10N by default (Santa Cruz area)
    resolution=50,  # meters per pixel
    method="linear",
    nodata=np.nan,
    out_path="zsmax_interpolated.tif",
    pad=0,
):
    """
    Interpolate scattered points to a regular grid and write as GeoTIFF.

    Parameters
    ----------
    x, y, z : 1D arrays
        UTM easting, northing (meters) and values (zsmax).
    epsg : int
        EPSG code for CRS (e.g., 32610 for UTM Zone 10N).
    resolution : float
        Pixel size in meters.
    method : {'nearest','linear','cubic'}
        Interpolation method for griddata.
    nodata : float
        Nodata value written to raster (QGIS will treat as transparent if set to NaN).
    out_path : str
        Output GeoTIFF path.
    pad : float
        Optional padding (meters) around data extent.
    """

    # Clean/finite mask
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]

    # Bounds
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    # Optional padding
    xmin -= pad
    ymin -= pad
    xmax += pad
    ymax += pad

    # Grid size
    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))

    # Build grid (note: y goes from max to min for north-up)
    grid_x = np.linspace(
        xmin + resolution / 2, xmin + resolution / 2 + (width - 1) * resolution, width
    )
    grid_y = np.linspace(
        ymax - resolution / 2, ymax - resolution / 2 - (height - 1) * resolution, height
    )
    GX, GY = np.meshgrid(grid_x, grid_y)

    # Interpolate
    points = np.column_stack([x, y])
    Z = griddata(points, z, (GX, GY), method=method)

    # Write GeoTIFF
    transform = from_origin(xmin, ymax, resolution, resolution)  # origin at upper-left
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "height": height,
        "width": width,
        "count": 1,
        "crs": f"EPSG:{epsg}",
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        # Replace NaNs with nodata value if nodata is not NaN
        if not np.isnan(nodata):
            Z_to_write = np.where(np.isfinite(Z), Z.astype("float32"), nodata)
        else:
            Z_to_write = Z.astype("float32")
        dst.write(Z_to_write, 1)

    print(f"Wrote {out_path} ({width} x {height}, {resolution} m pixels, EPSG:{epsg})")


def scattered_to_geotiff_binned(
    x,
    y,
    z,
    epsg=32610,
    resolution=50,
    stat="mean",  # 'mean', 'median', 'count', 'min', 'max'
    nodata=np.nan,
    out_path="zsmax_binned.tif",
    pad=0,
):
    """
    Aggregate scattered points into grid cells using binned statistics and write GeoTIFF.
    """

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad

    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))

    # Bin edges (x increases to the right, y increases upward)
    x_edges = np.linspace(xmin, xmin + width * resolution, width + 1)
    y_edges = np.linspace(ymin, ymin + height * resolution, height + 1)

    # binned_statistic_2d returns (y,x) indexed array with y increasing
    stats, _, _, _ = binned_statistic_2d(
        y, x, z, statistic=stat, bins=[y_edges, x_edges]
    )

    # Flip vertically so row 0 is top (north) for raster
    Z = np.flipud(stats.astype("float32"))

    transform = from_origin(xmin, ymin + height * resolution, resolution, resolution)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "height": height,
        "width": width,
        "count": 1,
        "crs": f"EPSG:{epsg}",
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        if not np.isnan(nodata):
            Z_to_write = np.where(np.isfinite(Z), Z, nodata)
        else:
            Z_to_write = Z
        dst.write(Z_to_write, 1)

    print(
        f"Wrote {out_path} ({width} x {height}, {resolution} m pixels, EPSG:{epsg}, stat={stat})"
    )


# ===============================================================================
# %% Load the data
# ===============================================================================
# Build the folder list
files = glob(os.path.join(dir_in, "**", "sfincs_map.nc"), recursive=True)
files = sorted(files)  # sort for deterministic order


file = files[0]
ds = xr.open_dataset(file)


if not files:
    raise FileNotFoundError(f"No sfincs_map.nc files under {dir_in}")

ds = concat_sfincs_years(
    files, time_var=time_var, var_select="zsmax", chunks={"timemax": 90}
)
print(ds)


ds


# ===============================================================================
# %% Open the Model configuration files
# ===============================================================================
# Load model output
from hydromt_sfincs import utils


xc, yc = load_SfincsQuadtree(os.path.join(dir_in, "sfincs.nc"))

maxwl = ds["zsmax"].max(dim="timemax").compute()


# Load topo-bathy
depfile = xr.open_dataset(file_DEM)
depfile = depfile["band_data"].isel(band=0)
depfile.attrs["crs"] = "EPSG:6339"


temp = utils.downscale_floodmap(maxwl, depfile, hmin=0.02, reproj_method="bilinear")


# plot

# x, y are UTM easting/northing; zsmax is your variable
# x = np.array([...]); y = np.array([...]); zsmax = np.array([...])

fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(xc, yc, c=maxwl, s=20, cmap="viridis", edgecolor="none")

cb = plt.colorbar(sc, ax=ax, label="zsmax")
ax.set_xlabel("UTM Easting (m)")
ax.set_ylabel("UTM Northing (m)")
ax.set_aspect("equal")  # preserves meters on both axes
ax.set_title("Scattered zsmax in UTM")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


scattered_to_geotiff_binned(
    xc,
    yc,
    maxwl,
    epsg=32610,
    resolution=50,
    stat="mean",
    out_path=os.path.join(dir_out, "zsmax_mean_50m.tif"),
)
