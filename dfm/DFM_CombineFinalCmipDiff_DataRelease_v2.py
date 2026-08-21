"""
Created on August 20th, 2026

This script combines ERA5 DFM water levels, CMIP6 differences, and tidal-only simulation water levels into a single dataset for each station.
It also adds the monthly quantiles of the water levels and the corresponding CMIP6 difference for each quantile.
The output is a netcdf file for each county with the data from the stations within that county.

Netcdf is processed for USGS data release

@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"

# ===============================================================================
# Import Modules (no Dask)
# ===============================================================================
import os
from glob import glob
import re
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

import shapely
from shapely import make_valid

# ===============================================================================
# Helpers & Functions
# ===============================================================================

def normalize(s):
    """Normalize strings: remove non-alphanumerics, strip, lowercase."""
    if s is None:
        return None
    return re.sub(r"\W+", "", str(s)).strip().lower()


def _preprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess a single dataset:
      - sort by time
      - drop duplicate time stamps (keep first)
      - normalize station dtype to string, strip
      - remove 'time' dim from invariant vars (lon, lat, bedlevel)
    """
    ds = ds.sortby("time")
    # Unique times
    tvals = ds["time"].values
    _, keep = np.unique(tvals, return_index=True)
    ds = ds.isel(time=np.sort(keep))

    # Normalize station dtype
    st = xr.DataArray(
        pd.Index(pd.Series(ds["station"].values).astype(str).str.strip().values),
        dims="station",
        name="station",
    )
    ds = ds.assign_coords(station=st.astype("U64"))

    # Remove time from invariant vars
    for var in ["lon", "lat", "bedlevel"]:
        if var in ds and "time" in ds[var].dims:
            ds[var] = ds[var].isel(time=0).astype("float32")

    return ds


def ensure_unique_sorted_time(ds: xr.Dataset, keep: str = "first") -> xr.Dataset:
    """
    Drop duplicate time stamps and sort by time.
    keep: 'first' or 'last' — which duplicate to keep.
    """
    tvals = ds["time"].values
    pdt = pd.to_datetime(tvals)  # robust conversion
    dup_mask = pd.Series(pdt).duplicated(keep=keep).to_numpy()
    if dup_mask.any():
        ds = ds.isel(time=~dup_mask)
    ds = ds.sortby("time")
    return ds


def assert_regular_time(ds: xr.Dataset, label: str = "dataset") -> None:
    """
    Assert the 'time' coordinate is strictly monotonic, has no duplicates,
    and is regularly sampled (constant cadence) after concatenation.

    Raises ValueError with informative details on failure.
    """
    if "time" not in ds.coords:
        raise ValueError(f"{label}: missing 'time' coordinate")

    t = pd.to_datetime(ds["time"].values)
    if t.size == 0:
        raise ValueError(f"{label}: empty time axis")

    # No NaT
    if pd.isna(t).any():
        bad_idx = np.where(pd.isna(t))[0]
        raise ValueError(f"{label}: NaT found at indices {bad_idx[:10]}")

    # Monotonic non-decreasing
    diffs = np.diff(t.values)
    if (diffs < np.timedelta64(0, "ns")).any():
        bad_idx = np.where(diffs < np.timedelta64(0, "ns"))[0]
        raise ValueError(f"{label}: time not monotonically increasing; examples at indices {bad_idx[:10]}")

    # No duplicates
    nunique = pd.Index(t).nunique()
    if nunique != t.size:
        raise ValueError(f"{label}: duplicate time stamps detected (n={t.size}, unique={nunique})")

    # Regular cadence check (constant delta)
    # Convert diffs to integer nanoseconds for exact comparison
    diffs_ns = np.array([int(np.timedelta64(d, "ns")) for d in diffs])
    if diffs_ns.size == 0:
        return  # single sample, trivially regular

    # Most common step (mode)
    # Use numpy to get mode robustly
    vals, counts = np.unique(diffs_ns, return_counts=True)
    step_ns = vals[np.argmax(counts)]

    irregular_idx = np.where(diffs_ns != step_ns)[0]
    if irregular_idx.size > 0:
        # Show a few examples with actual timestamps
        examples = [
            (str(t[i]), str(t[i + 1]), f"Δ={pd.to_timedelta(diffs_ns[i], unit='ns')}")
            for i in irregular_idx[:10]
        ]
        raise ValueError(
            f"{label}: irregular sampling detected. Expected constant step "
            f"{pd.to_timedelta(step_ns, unit='ns')} but found {irregular_idx.size} deviations. "
            f"Examples: {examples}"
        )


def load_and_concat(
    files: list[str],
    preprocess=_preprocess,
    engine: str = "netcdf4",
    label: str = "dataset",
) -> xr.Dataset:
    """
    Load each file via xarray.open_dataset (no Dask), materialize it in memory,
    preprocess, and concatenate along the 'time' dimension. Ensures unique, sorted times.

    Parameters
    ----------
    files : list[str]
        List of file paths to load and concatenate.
    preprocess : callable
        Function applied to each dataset prior to concatenation.
    engine : str
        NetCDF engine for open_dataset; default 'netcdf4'.
    label : str
        Human-readable dataset label used in error messages.

    Returns
    -------
    ds_cat : xr.Dataset
        Concatenated dataset along time.
    """
    if not files:
        raise FileNotFoundError(f"{label}: no input files found")

    dsets = []
    for fp in sorted(files):
        # Load and preprocess
        ds = xr.open_dataset(fp, engine=engine)
        if preprocess is not None:
            ds = preprocess(ds)
        # Load into memory (detach from file handles)
        ds.load()
        dsets.append(ds)

    # Concatenate along time
    ds_cat = xr.concat(
        dsets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )

    # Ensure uniqueness and sorted times
    ds_cat = ensure_unique_sorted_time(ds_cat)

    # Validate time axis integrity
    assert_regular_time(ds_cat, label=label)

    return ds_cat


def mask_stations_by_polygon(
    ds: xr.Dataset,
    remove_shp_path: str,
    station_dim: str = "station",
    lat_var: str = "lat",
    lon_var: str = "lon",
    predicate: str = "contains",  # 'contains', 'covered_by', or 'intersects'
    target_epsg: int = 4326,
):
    """
    Remove stations whose lat/lon fall inside (or touch) polygons from a removal shapefile.
    Works with scattered xarray datasets where lat/lon are variables along the 'station' dimension.
    """
    # --- Read and normalize shapefile
    if not os.path.exists(remove_shp_path):
        raise FileNotFoundError(f"Removal shapefile not found: {remove_shp_path}")

    remove_gdf = gpd.read_file(remove_shp_path)
    if remove_gdf.crs is None:
        raise ValueError("StationRemove.shp has no CRS defined. Please set its correct CRS before running.")

    # Reproject polygons to WGS84 if needed
    if (remove_gdf.crs.to_epsg() or 0) != target_epsg:
        remove_gdf = remove_gdf.to_crs(target_epsg)

    # Union polygons into a single coverage geometry and fix validity
    removal_geom = make_valid(remove_gdf.geometry.union_all())

    # --- Extract station coordinates (1D arrays)
    if lat_var not in ds or lon_var not in ds:
        raise KeyError(f"Dataset must contain variables '{lat_var}' and '{lon_var}'.")

    lat = ds[lat_var].values
    lon = ds[lon_var].values

    # Expect lat/lon to be 1D over 'station'
    if ds[lat_var].ndim != 1 or ds[lon_var].ndim != 1 or ds[lat_var].dims[0] != station_dim or ds[lon_var].dims[0] != station_dim:
        raise ValueError(f"'{lat_var}' and '{lon_var}' must be 1D variables over the '{station_dim}' dimension.")

    # --- Build Points (vectorized) and test predicate
    pts = shapely.points(lon, lat)  # array of shapely Point objects

    if predicate == "contains":
        inside = shapely.contains(removal_geom, pts)
    elif predicate == "covered_by":
        inside = shapely.covered_by(pts, removal_geom)
    elif predicate == "intersects":
        inside = shapely.intersects(pts, removal_geom)
    else:
        raise ValueError("predicate must be one of: 'contains', 'covered_by', 'intersects'")

    # --- Build a station mask DataArray and apply
    station_flag = xr.DataArray(
        inside,
        dims=(station_dim,),
        coords={station_dim: ds[station_dim]},
    )

    ds_masked = ds.where(~station_flag, drop=True)
    n_removed = int(station_flag.sum().item())
    print(f"Removed {n_removed} stations.")
    return ds_masked, station_flag


# ===============================================================================
# Main
# ===============================================================================

# ===============================================================================
# User Defined inputs
# ===============================================================================
SLR = "000"

dir_ERA5 = os.path.join(r"D:\Kai\DFM\ERA5", f"ERA5_{SLR}","Results_Combined")
dir_diff = os.path.join(r"D:\Kai\DFM\CDF_diff", f"{SLR}")
dir_Tidal = os.path.join(r"D:\Kai\DFM\ERA5_tidal", "ResultsCombined",SLR)
dir_gis = r"D:\Kai\DFM\GIS"
dir_out = os.path.join(r"D:\Kai\DFM", f"Combined_{SLR}")

# Packing information
PACK_SCALE = 1e-4  # meters per integer count (i.e., meters * 1e4)
FILL_INT = -9999

# ===============================================================================
# Load the data (no Dask; manual concatenation)
# ===============================================================================
print("Loading & concatenating data (no Dask)...")

files = sorted(glob(os.path.join(dir_ERA5, "ERA5_cdf*")))
ds_full = load_and_concat(files, preprocess=_preprocess, engine="netcdf4", label="ERA5/full")

files = sorted(glob(os.path.join(dir_Tidal, "*.nc")))
ds_tidal = load_and_concat(files, preprocess=_preprocess, engine="netcdf4", label="Tidal-only")

files = sorted(glob(os.path.join(dir_diff, "*.nc")))
ds_diff = load_and_concat(files, preprocess=_preprocess, engine="netcdf4", label="CMIP6 difference")

# Some of the ds_diff values got limited by the integer conversion — constrain only the data var
if "cmip_diff" in ds_diff:
    ds_diff["cmip_diff"] = ds_diff["cmip_diff"].where(
        (ds_diff["cmip_diff"] >= -2_000_000_000) & (ds_diff["cmip_diff"] <= 2_000_000_000)
    )

# Tide starts earlier: clip to ERA5 time span
ds_tidal = ds_tidal.sel(time=slice(ds_full["time"][0], ds_full["time"][-1]))

# ===============================================================================
# Process and convert to new dataset
# ===============================================================================

# Interpolate to hourly using nearest within a 2h tolerance
print("Interpolating to hourly...")
ds_full = ds_full.resample(time="1h").nearest(tolerance="2h")
ds_tidal = ds_tidal.resample(time="1h").nearest(tolerance="2h")
ds_diff = ds_diff.resample(time="1h").nearest(tolerance="2h")

# Ensure uniqueness post-resample
ds_full = ensure_unique_sorted_time(ds_full)
ds_tidal = ensure_unique_sorted_time(ds_tidal)
ds_diff = ensure_unique_sorted_time(ds_diff)

# Exact intersection along time & station
ds_full, ds_tidal, ds_diff = xr.align(
    ds_full,
    ds_tidal,
    ds_diff,
    join="inner",  # intersection only
    exclude=[],
)

# Confirm uniqueness again
ds_full = ensure_unique_sorted_time(ds_full)
ds_tidal = ensure_unique_sorted_time(ds_tidal)
ds_diff = ensure_unique_sorted_time(ds_diff)

# Create the final dataset and derived variables
print("Creating ds_era5 dataset...")
ds_era5 = ds_full.copy()
ds_era5["ntr"] = ds_full["waterlevel"] - ds_tidal["waterlevel"]

# Drop Bedlevel if present
if "bedlevel" in ds_era5:
    ds_era5 = ds_era5.drop_vars("bedlevel")

# Remove any packing/encoding from lon/lat and ensure pure float32 without 'time'
for var in ["lon", "lat"]:
    if var in ds_era5:
        da = ds_era5[var].copy().astype("float32")
        for key in ("ScaleFactor", "scale_factor", "add_offset", "_FillValue", "dtype"):
            da.attrs.pop(key, None)
        da.encoding.clear()
        ds_era5[var] = da

# Add the CMIP6 difference to the dataset (aligned)
ds_diff = ds_diff.reindex(time=ds_full["time"], station=ds_full["station"], method=None)
ds_era5["wl_CmipDiff"] = ds_diff["cmip_diff"] / 10000.0  # convert packed ints to meters

# Deal with scaling for waterlevel and ntr (original integers)
ds_era5["waterlevel"] = ds_era5["waterlevel"] / 10000.0
ds_era5["ntr"] = ds_era5["ntr"] / 10000.0

# Remove SLR (SLR provided in cm units)
ds_era5["waterlevel"] = ds_era5["waterlevel"] - (int(SLR) / 100.0)

# --- Monthly quantiles (percent ranks per station within calendar month) ---
ds_era5["waterlevel"] = ds_era5["waterlevel"].astype("float32")
wl_quants = ds_era5["waterlevel"].groupby("time.month").map(
    lambda g: g.rank(dim="time", pct=True)
).transpose("time", "station").astype("float32")
ds_era5["wl_quants"] = wl_quants

# Attributes
ds_era5["waterlevel"].attrs = {
    "units": "meters",
    "standard_name": "sea_surface_height_above_reference_datum",
    "long_name": "water level",
    "reference": "NAVD88",
    "desc": "Modeled water levels for the reanalysis period (SLR removed)",
    "note": "Variable scaled with _ScaleFactor in file. Confirm correct decoding by your software.",
    "precision": "Data encoded as integer with 4 significant digits.",
}

ds_era5["ntr"].attrs = {
    "units": "meters",
    "long_name": "non-tidal residual",
    "desc": "Calculated by subtracting modeled water levels with tidal-only forcing from a run with full forcing.",
    "reference": "NAVD88",
    "note": "Variable scaled with _ScaleFactor in file.",
    "precision": "Data encoded as integer with 4 significant digits.",
}

ds_era5["wl_CmipDiff"].attrs = {
    "long_name": "CMIP6 difference in water levels",
    "units": "meters",
    "desc": (
        "Predicted change by each CMIP6 model for each ERA5 water level value. "
        "Delta is calculated by subtracting the future period (2015-2050) from the historic period (1950-2014). "
        "Change is calculated for each quantile for each month."
    ),
    "usage": "Adding wl_CmipDiff to waterlevel produces the pseudo-global-warming time series.",
    "note": "Variable scaled with _ScaleFactor in file.",
    "precision": "Data encoded as integer with 4 significant digits.",
}

ds_era5["wl_quants"].attrs = {
    "units": "None",
    "standard_name": "waterlevel_monthly_percentile",
    "long_name": "Monthly water level percentile (per station, across all years)",
    "desc": (
        "For each timestamp, percentile (0-1) of the water level within its calendar month, "
        "computed across all years for that station."
    ),
    "note": "Encoded as integer in file with scale_factor=1e-4 (see encoding).",
    "precision": "4 decimal places after unpacking.",
}

ds_era5["lon"].attrs = {
    "standard_name": "longitude",
    "long_name": "x-coordinate of station",
    "projection": "WGS 84",
    "epsg": "4326",
    "units": "degree_east",
}
ds_era5["lat"].attrs = {
    "standard_name": "latitude",
    "long_name": "y-coordinate of station",
    "projection": "WGS 84",
    "epsg": "4326",
    "units": "degrees_north",
}

if "cmip6" in ds_era5.coords:
    ds_era5["cmip6"].attrs = {"long_name": "CMIP6 Model (HighResMIP)"}

# Global Attributes
ds_era5.attrs["processing_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
ds_era5.attrs.pop("forcing", None)
ds_era5.attrs.pop("source_dir", None)
ds_era5.attrs["author"] = "Kai Parker (USGS PCMSC)"
ds_era5.attrs["description"] = (
    "This dataset contains modeled water levels and non-tidal residual for the reanalysis period. "
    "Modeled changes to the reanalysis time series (as predicted by CMIP6) are also included. "
    "Output is for stations in the Salish Sea."
)
ds_era5.attrs["DataReleaseCitation"] = "XXXXXX"
ds_era5.attrs["ModelCitation"] = "XXXXX"
ds_era5.attrs["InterpretiveProductCitation"] = "XXXXXX"
ds_era5.attrs.update({
    "Conventions": "CF-1.10, ACDD-1.3",
    "title": "Reanalysis and projected water levels for Salish Sea stations",
    "institution": "USGS PCMSC",
    "source": "DFM ERA5 reanalysis; tidal-only runs; CMIP6 deltas",
    "history": f"Created {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with integer packing (scale_factor=1e-4).",
    "references": "Add citations for DFM configuration, CMIP6 deltas, and interpretive products",
})

# ===============================================================================
# Guardrails (final dataset)
# ===============================================================================

# 1) Time monotonic and no NaT
tvals = ds_era5["time"].values
assert np.all(~pd.isna(tvals)), "Found NaT in time after decode_cf"
assert np.all(np.diff(tvals.astype("datetime64[ns]")) >= np.timedelta64(0, "ns")), "Time is not monotonically increasing"

# 2) Station unique
st = pd.Index(ds_era5["station"].values)
assert st.is_unique, "Duplicate station IDs found"

# 3) Big NaN blocks? Check variables to be written
for v in ["waterlevel", "ntr", "wl_CmipDiff"]:
    if v in ds_era5:
        nan_by_time = ds_era5[v].isnull().mean(dim="station")
        bad = (nan_by_time > 0.95)
        if bool(bad.any()):
            print(f"WARNING: {v} has time slices with >95% NaNs. Example indices:", np.where(bad.values)[0][:10])

# ===============================================================================
# Remove stations using a polygon.
# ===============================================================================
ds_era5, remove_flag = mask_stations_by_polygon(
    ds_era5,
    os.path.join(dir_gis, "StationRemove.shp"),
    lat_var="lat",
    lon_var="lon",
    predicate="contains",  # or "covered_by" to include boundary points
)
print(f"Removed {int(remove_flag.sum().item())} stations.")

# ===============================================================================
# Load the county information for spatial grouping
# ===============================================================================
print("Loading County shapefiles and finding station subsets...")

counties = gpd.read_file(os.path.join(dir_gis, "Washington_Counties_(no_water)___washco_area.shp"))
counties = counties.to_crs(crs="EPSG:4326")

county_list = [
    "Kitsap", "Snohomish", "Island", "Skagit", "Jefferson",
    "King", "Pierce", "Thurston", "Whatcom", "Mason",
    "San Juan", "Clallam",
]

counties["COUNTY_norm"] = counties["COUNTY"].astype(str).map(normalize)
alts = "|".join(re.escape(normalize(c)) for c in county_list)
pattern = rf"(?:^|.*)({alts})(?:.*|$)"
mask = counties["COUNTY_norm"].str.contains(pattern, regex=True, na=False)
counties = counties[mask].copy()

# Station GeoDataFrame from ds_era5
stations = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(ds_era5["lon"].values, ds_era5["lat"].values),
    crs="EPSG:4326",
)

# Spatial join to assign county IDs to station indices
Index_DFM = gpd.sjoin(counties, stations, how="right", predicate="intersects")
Index_DFM = Index_DFM.rename(columns={"index_left": "CountyID"})
Index_DFM["CountyID"] = Index_DFM["CountyID"].fillna(-999).astype("int32")

# ===============================================================================
# Output
# ===============================================================================
print("Outputting county datasets...")

int_encoding = dict(
    dtype="int32",
    zlib=True,
    shuffle=True,
    complevel=5,
    _FillValue=FILL_INT,
    scale_factor=PACK_SCALE,
    add_offset=0.0,
)
coord_float_encoding = dict(
    dtype="float32",
    zlib=True,
    shuffle=True,
    complevel=5,
)

if not os.path.exists(dir_out):
    os.makedirs(dir_out)
    print("Output directory created.")

ind_county = pd.Index(list(counties.index) + [-999])  # keep an OutOfCounty bucket

for county in ind_county:
    county_name = counties.loc[county]["COUNTY_norm"] if county != -999 else "OutOfCounty"
    print(f"Processing county {county_name}...")

    dfm_pnts = Index_DFM[Index_DFM["CountyID"] == county]
    ds_county = ds_era5.isel(station=dfm_pnts.index)
    ds_county.attrs["County"] = county_name

    out_path = os.path.join(dir_out, f"Reanalysis_and_Projected_CoSMoSwaterlevels_{county_name}.nc")
    ds_county.to_netcdf(
        out_path,
        engine="netcdf4",
        encoding={
            "waterlevel": int_encoding,
            "ntr": int_encoding,
            "wl_CmipDiff": int_encoding,
            "wl_quants": int_encoding,
            "lon": coord_float_encoding,
            "lat": coord_float_encoding,
        },
    )

    # Optional quick sanity check
    with xr.open_dataset(out_path, engine="netcdf4") as chk:
        print("Written:", out_path, "| vars:", list(chk.data_vars))

print("All done.")

