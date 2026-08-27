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

# -------------------------
# Config (same as before)
# -------------------------
SLR = "000"

dir_ERA5 = os.path.join(r"D:\Kai\DFM\ERA5", f"ERA5_{SLR}", "Results_Combined")
dir_diff = os.path.join(r"D:\Kai\DFM\CDF_diff", f"{SLR}")
dir_Tidal = os.path.join(r"D:\Kai\DFM\ERA5_tidal\Results_combined", SLR)
dir_gis = r"D:\Kai\DFM\GIS"
dir_out = os.path.join(r"D:\Kai\DFM", f"Combined_{SLR}")

PACK_SCALE = 1e-4  # meters per integer count
FILL_INT = -9999

TARGET_EPSG = 4326
COUNTY_SHP = os.path.join(dir_gis, "Washington_Counties_(no_water)___washco_area.shp")
REMOVE_SHP = os.path.join(dir_gis, "StationRemove.shp")

county_list = [
    "Kitsap", "Snohomish", "Island", "Skagit", "Jefferson",
    "King", "Pierce", "Thurston", "Whatcom", "Mason",
    "San Juan", "Clallam",
]

# ===============================================================================
# Helpers (mostly unchanged, plus new subset-aware loader and county indexer)
# ===============================================================================

def normalize(s):
    if s is None:
        return None
    return re.sub(r"\W+", "", str(s)).strip().lower()

def _preprocess(ds: xr.Dataset) -> xr.Dataset:
    # Sort & dedup time
    ds = ds.sortby("time")
    if "time" in ds:
        tvals = ds["time"].values
        _, keep = np.unique(tvals, return_index=True)
        ds = ds.isel(time=np.sort(keep))

    # --- Ensure 'station' coordinate exists and is string-normalized ---
    if "station" in ds.dims:
        # Prefer coordinate if present; otherwise build from indices
        if "station" in ds.coords:
            st_vals = ds.coords["station"].values
        else:
            # No coordinate variable → use indices as labels
            st_vals = np.arange(ds.dims["station"], dtype=np.int32)

        st = xr.DataArray(
            pd.Index(pd.Series(st_vals).astype(str).str.strip().values),
            dims="station",
            name="station",
        )
        ds = ds.assign_coords(station=st.astype("U64"))
    else:
        # No station dimension at all → fallback (skip later if needed)
        ds = ds.expand_dims({"station": 1})
        ds = ds.assign_coords(station=np.array(["unknown"], dtype="U64"))

    # Remove time from invariant vars
    for var in ["lon", "lat", "bedlevel"]:
        if var in ds and "time" in ds[var].dims:
            ds[var] = ds[var].isel(time=0).astype("float32")

    return ds


def ensure_unique_sorted_time(ds: xr.Dataset, keep: str = "first") -> xr.Dataset:
    pdt = pd.to_datetime(ds["time"].values)
    dup_mask = pd.Series(pdt).duplicated(keep=keep).to_numpy()
    if dup_mask.any():
        ds = ds.isel(time=~dup_mask)
    ds = ds.sortby("time")
    return ds

def assert_regular_time(ds: xr.Dataset, label: str = "dataset") -> None:
    if "time" not in ds.coords:
        raise ValueError(f"{label}: missing 'time' coordinate")
    t = pd.to_datetime(ds["time"].values)
    if t.size == 0:
        raise ValueError(f"{label}: empty time axis")
    if pd.isna(t).any():
        bad_idx = np.where(pd.isna(t))[0]
        raise ValueError(f"{label}: NaT found at indices {bad_idx[:10]}")
    diffs = np.diff(t.values)
    if (diffs < np.timedelta64(0, "ns")).any():
        bad_idx = np.where(diffs < np.timedelta64(0, "ns"))[0]
        raise ValueError(f"{label}: time not monotonically increasing; examples at indices {bad_idx[:10]}")
    nunique = pd.Index(t).nunique()
    if nunique != t.size:
        raise ValueError(f"{label}: duplicate time stamps detected (n={t.size}, unique={nunique})")
    # regular cadence
    diffs_ns = np.array([int(np.timedelta64(d, "ns")) for d in diffs])
    if diffs_ns.size == 0:
        return
    vals, counts = np.unique(diffs_ns, return_counts=True)
    step_ns = vals[np.argmax(counts)]
    irregular_idx = np.where(diffs_ns != step_ns)[0]
    if irregular_idx.size > 0:
        examples = [
            (str(t[i]), str(t[i + 1]), f"Δ={pd.to_timedelta(diffs_ns[i], unit='ns')}")
            for i in irregular_idx[:10]
        ]
        raise ValueError(
            f"{label}: irregular sampling detected. Expected constant step "
            f"{pd.to_timedelta(step_ns, unit='ns')} but found {irregular_idx.size} deviations. "
            f"Examples: {examples}"
        )

def _union_valid(geom_series):
    """Version-safe union and validity fix for polygons."""
    try:
        united = geom_series.union_all()  # GeoPandas ≥0.13
    except AttributeError:
        united = geom_series.unary_union   # older GeoPandas
    return make_valid(united)

def build_station_metadata(example_file: str, engine: str = "netcdf4") -> pd.DataFrame:
    """
    Read station, lon, lat with minimal IO from a representative ERA5 file.
    """
    ds = xr.open_dataset(example_file, engine=engine)
    ds = _preprocess(ds)
    # Extract 1D station info
    st = pd.Index(ds["station"].values)
    lon = ds["lon"].values
    lat = ds["lat"].values
    return pd.DataFrame({"station": st.astype(str), "lon": lon.astype("float32"), "lat": lat.astype("float32")})

def build_county_station_labels(
    station_df: pd.DataFrame,
    county_shp: str,
    county_list: list[str],
    remove_shp: str | None = None,
    target_epsg: int = 4326,
    boundary_predicate: str = "covered_by",  # includes boundary points
):
    """
    Returns:
      county_map: dict {county_name_norm: np.ndarray of station labels}
      out_of_county: np.ndarray of station labels outside all targets
    """
    # County polygons
    counties = gpd.read_file(county_shp)
    counties = counties.to_crs(crs=f"EPSG:{target_epsg}")
    counties["COUNTY_norm"] = counties["COUNTY"].astype(str).map(normalize)
    target_norm = {normalize(c) for c in county_list}
    counties = counties[counties["COUNTY_norm"].isin(target_norm)].copy()
    if counties.empty:
        raise ValueError("No matching counties found after normalization.")

    # Station points
    stations_gdf = gpd.GeoDataFrame(
        station_df,
        geometry=gpd.points_from_xy(station_df["lon"], station_df["lat"]),
        crs=f"EPSG:{target_epsg}",
    )

    # Optional polygon removal
    if remove_shp and os.path.exists(remove_shp):
        removal = gpd.read_file(remove_shp)
        if removal.crs is None:
            raise ValueError("StationRemove.shp has no CRS defined.")
        removal = removal.to_crs(target_epsg)
        removal_geom = _union_valid(removal.geometry)
        pts = shapely.points(station_df["lon"].values, station_df["lat"].values)
        if boundary_predicate == "contains":
            inside = shapely.contains(removal_geom, pts)
        elif boundary_predicate == "covered_by":
            inside = shapely.covered_by(pts, removal_geom)
        elif boundary_predicate == "intersects":
            inside = shapely.intersects(pts, removal_geom)
        else:
            raise ValueError("predicate must be one of: 'contains', 'covered_by', 'intersects'")
        # Keep only stations NOT inside removal polygon
        stations_gdf = stations_gdf.loc[~inside].copy()

    # Spatial join (stations→county)
    j = gpd.sjoin(stations_gdf, counties[["geometry", "COUNTY_norm"]], how="left", predicate="intersects")
    # Build map
    county_map = {}
    for cname in sorted(target_norm):
        labels = j.loc[j["COUNTY_norm"] == cname, "station"].astype(str).values
        county_map[cname] = labels
    out_labels = j.loc[j["COUNTY_norm"].isna(), "station"].astype(str).values
    return county_map, out_labels

def load_and_concat_subset(
    files: list[str],
    station_labels: np.ndarray,
    preprocess=_preprocess,
    engine: str = "netcdf4",
    label: str = "dataset",
) -> xr.Dataset:
    """
    Memory-aware loader: process per file, subset stations BEFORE loading,
    then concatenate along 'time'.
    """
    if not files:
        raise FileNotFoundError(f"{label}: no input files found")

    dsets = []
    # Ensure labels are str array
    station_labels = np.array(pd.Index(station_labels).astype(str))
    for fp in sorted(files):
        ds = xr.open_dataset(fp, engine=engine)
        if preprocess is not None:
            ds = preprocess(ds)
        # Subset by station name/label; if some labels missing, intersect
                
        have = pd.Index(ds["station"].values).astype(str)
        pick = have.intersection(station_labels)
        if pick.size == 0:
            # Nothing to load from this file for this county
            # Optional: print a few sample stations to diagnose mismatches
            print(f"NOTE: {os.path.basename(fp)}: no matching stations for county; sample labels: {list(have[:3])}")
            continue

        ds = ds.sel(station=pick.values)
        # Materialize ONLY the subset
        ds.load()
        dsets.append(ds)

    if not dsets:
        raise ValueError(f"{label}: no data for requested station subset")

    ds_cat = xr.concat(
        dsets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )
    ds_cat = ensure_unique_sorted_time(ds_cat)
    assert_regular_time(ds_cat, label=label)
    return ds_cat

# ===============================================================================
# County-first pipeline
# ===============================================================================

print("Building station metadata and county station labels...")

# Probe one ERA5 file to get station metadata only
era5_files = sorted(glob(os.path.join(dir_ERA5, "ERA5_cdf*")))
tidal_files = sorted(glob(os.path.join(dir_Tidal, "*.nc")))
diff_files = sorted(glob(os.path.join(dir_diff, "*.nc")))

if not era5_files or not tidal_files or not diff_files:
    raise FileNotFoundError("Missing inputs: check ERA5, tidal-only, and CMIP6 diff directories.")

station_meta = build_station_metadata(era5_files[0], engine="netcdf4")
county_map, out_of_county = build_county_station_labels(
    station_meta,
    COUNTY_SHP,
    county_list,
    remove_shp=REMOVE_SHP,
    target_epsg=TARGET_EPSG,
    boundary_predicate="covered_by",
)

# Optional “OutOfCounty” bucket
county_order = list(county_map.keys()) + ["outofcounty"]

o = [4, 9, 0, 1,2,3,5,6,7,8,10,11,12]
county_order = [county_order[i] for i in o]


# Output directory
os.makedirs(dir_out, exist_ok=True)
print("Output directory ready:", dir_out)

# Common encodings
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

# Iterate counties (streaming memory)
for county_name_norm in county_order:
    if county_name_norm == "outofcounty":
        station_labels = out_of_county
        if station_labels.size == 0:
            print("Skipping OutOfCounty: no stations.")
            continue
        out_name = "OutOfCounty"
    else:
        station_labels = county_map[county_name_norm]
        if station_labels.size == 0:
            print(f"Skipping {county_name_norm}: no stations mapped.")
            continue
        # Recover nice county title from original list if desired
        out_name = next((c for c in county_list if normalize(c) == county_name_norm), county_name_norm)

    print(f"\n=== Processing county: {out_name} | {station_labels.size} station(s) ===")

    # 1) Load subsets for this county
    ds_full = load_and_concat_subset(era5_files, station_labels, preprocess=_preprocess, engine="netcdf4", label=f"ERA5/{out_name}")
    ds_tidal = load_and_concat_subset(tidal_files, station_labels, preprocess=_preprocess, engine="netcdf4", label=f"Tidal/{out_name}")
    ds_diff = load_and_concat_subset(diff_files, station_labels, preprocess=_preprocess, engine="netcdf4", label=f"CMIP6diff/{out_name}")

    # 2) Align time range (tide often starts earlier)
    ds_tidal = ds_tidal.sel(time=slice(ds_full["time"][0], ds_full["time"][-1]))

    # 3) Resample to hourly nearest (2h tolerance) and ensure uniqueness
    ds_full = ensure_unique_sorted_time(ds_full.resample(time="1h").nearest(tolerance="2h"))
    ds_tidal = ensure_unique_sorted_time(ds_tidal.resample(time="1h").nearest(tolerance="2h"))
    ds_diff = ensure_unique_sorted_time(ds_diff.resample(time="1h").nearest(tolerance="2h"))

    nan_by_time = ds_diff["cmip_diff"].isnull().mean(dim="station")
    print("ds_diff hourly NaN slices:", int((nan_by_time > 0).sum()))
    print("Example hours with NaN:", pd.to_datetime(ds_diff.time.where(nan_by_time>0, drop=True).values[:10]))


    # 4) Exact intersection along time & station
    ds_full, ds_tidal, ds_diff = xr.align(ds_full, ds_tidal, ds_diff, join="inner")

    # 5) Create ds_era5 for this county and derived variables
    ds_era5 = ds_full.copy()
    ds_era5["ntr"] = ds_full["waterlevel"] - ds_tidal["waterlevel"]

    # Drop Bedlevel if present
    if "bedlevel" in ds_era5:
        ds_era5 = ds_era5.drop_vars("bedlevel")

    # Clean lon/lat encodings and attributes
    for var in ["lon", "lat"]:
        if var in ds_era5:
            da = ds_era5[var].copy().astype("float32")
            for key in ("ScaleFactor", "scale_factor", "add_offset", "_FillValue", "dtype"):
                da.attrs.pop(key, None)
            da.encoding.clear()
            ds_era5[var] = da

    # CMIP6 diff range clamp (if packed ints were clipped)
    if "cmip_diff" in ds_diff:
        ds_diff["cmip_diff"] = ds_diff["cmip_diff"].where(
            (ds_diff["cmip_diff"] >= -2_000_000_000) & (ds_diff["cmip_diff"] <= 2_000_000_000)
        )

    # Add wl_CmipDiff (convert to meters)
    ds_era5["wl_CmipDiff"] = ds_diff["cmip_diff"] / 10000.0

    # Scale to meters, remove SLR (cm)
    ds_era5["waterlevel"] = ds_era5["waterlevel"] / 10000.0
    ds_era5["ntr"] = ds_era5["ntr"] / 10000.0
    ds_era5["waterlevel"] = ds_era5["waterlevel"] - (int(SLR) / 100.0)

    # Monthly quantiles (per station within calendar month)
    ds_era5["waterlevel"] = ds_era5["waterlevel"].astype("float32")
    wl_quants = (
        ds_era5["waterlevel"]
        .groupby("time.month")
        .map(lambda g: g.where(~np.isnan(g)).rank(dim="time", pct=True))
        .astype("float32")
    ).transpose("time", "station")
    ds_era5["wl_quants"] = wl_quants

    # Variable attrs (CF/ACDD tightened)
    for v in ["waterlevel", "ntr"]:
        if v in ds_era5:
            ds_era5[v].attrs.update({
                "units": "meters",
                "reference": "NAVD88",
                "coordinates": "lon lat station time",
                "note": "Variable written with CF scale_factor=1e-4 (integer packing).",
            })
    if "waterlevel" in ds_era5:
        ds_era5["waterlevel"].attrs.update({
            "standard_name": "sea_surface_height_above_reference_datum",
            "long_name": "water level (SLR removed)",
            "precision": "Data encoded as integer with 4 significant digits.",
        })
    if "ntr" in ds_era5:
        ds_era5["ntr"].attrs.update({
            "long_name": "non-tidal residual",
            "precision": "Data encoded as integer with 4 significant digits.",
        })
    if "wl_CmipDiff" in ds_era5:
        ds_era5["wl_CmipDiff"].attrs.update({
            "long_name": "CMIP6 difference in water levels",
            "units": "meters",
            "coordinates": "lon lat station time",
            "usage": "Adding wl_CmipDiff to waterlevel produces the pseudo-global-warming time series.",
            "note": "Variable written with CF scale_factor=1e-4 (integer packing).",
            "precision": "Data encoded as integer with 4 significant digits.",
        })
    ds_era5["wl_quants"].attrs.update({
        "units": "1",
        "standard_name": "waterlevel_monthly_percentile",
        "long_name": "Monthly water level percentile (per station, across all years)",
        "coordinates": "lon lat station time",
        "note": "Encoded as integer in file with scale_factor=1e-4 (see encoding).",
    })
    ds_era5["lon"].attrs.update({
        "standard_name": "longitude", "long_name": "x-coordinate of station",
        "projection": "WGS 84", "epsg": "4326", "units": "degree_east",
    })
    ds_era5["lat"].attrs.update({
        "standard_name": "latitude", "long_name": "y-coordinate of station",
        "projection": "WGS 84", "epsg": "4326", "units": "degrees_north",
    })

    # Global attrs
    ds_era5.attrs["processing_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ds_era5.attrs["author"] = "Kai Parker (USGS PCMSC)"
    ds_era5.attrs["description"] = (
        "Modeled water levels and non-tidal residual for the reanalysis period. "
        "Modeled changes to the reanalysis time series (as predicted by CMIP6) are also included. "
        f"Output subset for county: {out_name}."
    )
    ds_era5.attrs.update({
        "Conventions": "CF-1.10, ACDD-1.3",
        "title": "Reanalysis and projected water levels for Salish Sea stations",
        "institution": "USGS PCMSC",
        "source": "DFM ERA5 reanalysis; tidal-only runs; CMIP6 deltas",
        "history": f"Created {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with integer packing (scale_factor=1e-4).",
        "references": "Add citations for DFM configuration, CMIP6 deltas, and interpretive products",
        "time_coverage_start": pd.to_datetime(ds_era5.time.values[0]).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time_coverage_end": pd.to_datetime(ds_era5.time.values[-1]).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "geospatial_lat_min": float(ds_era5["lat"].min().values),
        "geospatial_lat_max": float(ds_era5["lat"].max().values),
        "geospatial_lon_min": float(ds_era5["lon"].min().values),
        "geospatial_lon_max": float(ds_era5["lon"].max().values),
        "geospatial_lat_units": "degrees_north",
        "geospatial_lon_units": "degree_east",
        "DataReleaseCitation": "XXXXXX",
        "ModelCitation": "XXXXX",
        "InterpretiveProductCitation": "XXXXXX",
    })

    # Guardrails
    tvals = ds_era5["time"].values
    assert np.all(~pd.isna(tvals)), "Found NaT in time after decode_cf"
    assert np.all(np.diff(tvals.astype("datetime64[ns]")) >= np.timedelta64(0, "ns")), "Time is not monotonically increasing"
    st = pd.Index(ds_era5["station"].values)
    assert st.is_unique, "Duplicate station IDs found"

    for v in ["waterlevel", "ntr", "wl_CmipDiff"]:
        if v in ds_era5:
            nan_by_time = ds_era5[v].isnull().mean(dim="station")
            bad = (nan_by_time > 0.95)
            if bool(bad.any()):
                print(f"WARNING [{out_name}]: {v} has time slices with >95% NaNs. Example indices:", np.where(bad.values)[0][:10])

    # Write county file (packed ints)
    out_path = os.path.join(dir_out, f"Reanalysis_and_Projected_CoSMoSwaterlevels_{out_name}.nc")
    ds_era5.to_netcdf(
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

    
    pre_write_nan_frac = float(ds_era5["wl_CmipDiff"].isnull().mean().values)
    print("wl_CmipDiff NaN fraction BEFORE write:", pre_write_nan_frac)

    # Smoke test
    with xr.open_dataset(out_path, engine="netcdf4") as chk:
        print(f"Written: {out_path} | vars:", list(chk.data_vars))
        
        post_write_nan_frac = float(chk["wl_CmipDiff"].isnull().mean().values)
        print("wl_CmipDiff NaN fraction AFTER write:", post_write_nan_frac)


print("\nAll counties processed. Done.")