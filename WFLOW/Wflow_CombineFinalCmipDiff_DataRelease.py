# -*- coding: utf-8 -*-
"""
Created on April 8th, 2026

This script combines ERA5 WFLOW Q, CMIP6 differences, and combines into a single dataset for each station.
It also adds the monthly quantiles of the water levels and the corresponding CMIP6 difference for each quantile.
The output is a netcdf file for each county with the data from the stations within that county.

Netcdf is processed for USGS data release

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
import dask.distributed
from datetime import datetime
import pandas as pd
import re 


# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the DFM data resides
# dir_in = r'D:\DFM'
dir_in = r"C:\Users\kai\Documents\KaiDownloads\WFLOW\11_20_2025_Discharges_SnohomishKitsap"
dir_out = r"Y:\WFLOW\20240419_discharge_wflow_CMIP6_Combined"

# model grid to process (county)
cnty = "kitsap"

# ===============================================================================
# %% Define some functions
# ===============================================================================


# Helper to normalize strings: remove non-alphanumerics, strip, lowercase
def normalize(s):
    if s is None:
        return None
    return re.sub(r"\W+", "", str(s)).strip().lower()


def _preprocess(ds):
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


def ensure_unique_sorted_time(ds, keep="first"):
    """
    Drop duplicate time stamps and sort by time.
    keep: 'first' or 'last' — which duplicate to keep.
    """
    # Convert time coord to pandas datetime for reliable duplicate detection
    tvals = ds["time"].values
    # pd.to_datetime handles datetime64/cftime->numpy conversion reasonably; if cftime persists, decode_cf first.
    pdt = pd.to_datetime(tvals)
    dup_mask = pd.Series(pdt).duplicated(keep=keep).to_numpy()  # True where duplicate
    if dup_mask.any():
        ds = ds.isel(time=~dup_mask)
    ds = ds.sortby("time")
    return ds


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

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with a 'station' dimension and variables lat_var, lon_var of shape (station,).
    remove_shp_path : str
        Path to the StationRemove.shp (polygon coverage).
    station_dim : str
        Name of the station dimension.
    lat_var, lon_var : str
        Variable names holding station lat/lon (1D arrays over station).
    predicate : str
        Spatial predicate to decide removal:
            - 'contains': remove stations strictly inside polygons (excludes boundary).
            - 'covered_by': remove stations inside or on the boundary (inclusive).
            - 'intersects': remove stations that intersect polygons (boundary included).
    target_epsg : int
        Target CRS for polygon operations (default WGS84 = EPSG:4326).

    Returns
    -------
    ds_masked : xr.Dataset
        Dataset with stations removed where predicate is True.
    station_flag : xr.DataArray
        Boolean mask over station: True means removed.
    """

    # --- Read and normalize shapefile
    if not os.path.exists(remove_shp_path):
        raise FileNotFoundError(f"Removal shapefile not found: {remove_shp_path}")

    remove_gdf = gpd.read_file(remove_shp_path)

    if remove_gdf.crs is None:
        raise ValueError(
            "StationRemove.shp has no CRS defined. Please set its correct CRS before running."
        )

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
    if (
        ds[lat_var].ndim != 1
        or ds[lon_var].ndim != 1
        or ds[lat_var].dims[0] != station_dim
        or ds[lon_var].dims[0] != station_dim
    ):
        raise ValueError(
            f"'{lat_var}' and '{lon_var}' must be 1D variables over the '{station_dim}' dimension."
        )

    # --- Build Points (vectorized) and test predicate
    pts = shapely.points(lon, lat)  # array of shapely Point objects

    if predicate == "contains":
        inside = shapely.contains(removal_geom, pts)
    elif predicate == "covered_by":
        inside = shapely.covered_by(pts, removal_geom)
    elif predicate == "intersects":
        inside = shapely.intersects(pts, removal_geom)
    else:
        raise ValueError(
            "predicate must be one of: 'contains', 'covered_by', 'intersects'"
        )

    # --- Build a station mask DataArray and apply
    station_flag = xr.DataArray(
        inside, dims=(station_dim,), coords={station_dim: ds[station_dim]}
    )

    ds_masked = ds.where(~station_flag, drop=True)

    # Optional diagnostics
    n_removed = int(station_flag.sum().item())
    print(f"Removed {n_removed} stations.")
    return ds_masked, station_flag


def main():

    print('THIS HAS not been finished111 still in progress')
    
    # ===============================================================================
    # User Defined inputs
    # ===============================================================================

    dir_ERA5 = os.path.join(r"D:\Kai\DFM", f"ERA5_{SLR}")
    dir_diff = os.path.join(r"D:\Kai\DFM", f"CDF_diff_{SLR}")
    dir_Tidal = os.path.join(r"D:\Kai\DFM", f"ERA5_{SLR}_Tidal")
    dir_gis = r"D:\Kai\DFM\GIS"
    dir_out = os.path.join(r"D:\Kai\DFM", f"Combined_{SLR}")

    # Number of cpus to put into the cluster.
    # n = os.cpu_count() or 1
    n = 10

    # Packing information

    PACK_SCALE = 1e-4  # meters per integer count (i.e., meters * 1e4)
    FILL_INT = -9999

    # ===============================================================================
    # Parallelize with Dask
    # ===============================================================================
    print("starting Dask Cluster")

    # Some of the steps we will take are aware of parallel clustered compute environments
    # using `dask`. We're going to start a cluster now so that future steps can take advantage
    # of this ability.
    #
    # This is an optional step, but speed ups data loading significantly, especially
    # when accessing data from the cloud.
    #
    # We have documentation on how to start a Dask Cluster in different computing environments [here](../environment_set_up/clusters.md).

    # Reduce oversubscription from BLAS libraries
    # os.environ.setdefault("MKL_NUM_THREADS", "1")
    # os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    # os.environ.setdefault("OMP_NUM_THREADS", "1")

    # # Prefer processes for CPU-bound Python work
    # cluster = LocalCluster(
    #     n_workers=n,  # usually #cores
    #     threads_per_worker=1,
    #     processes=True,
    #     silence_logs=False,  # helpful while debugging
    # )
    # client = Client(cluster)

    # print("Dashboard:", cluster.dashboard_link)
    # print("Workers:", client.scheduler_info().get("workers", {}).keys())

    # ===============================================================================
    # Load the data
    # ===============================================================================

    # Load the data
    print("Loading data...")
    # ds_full = xr.open_mfdataset(
    #     files, engine="netcdf4", parallel=True, chunks={"time": 52560, "station": 64}, combine="by_coords")

    # ds_tidal = xr.open_mfdataset(
    #     files,
    #     engine="netcdf4",
    #     parallel=True,
    #     chunks={"time": 52560, "station": 64},
    #     combine="by_coords",
    # )

    # files = glob(os.path.join(dir_full, "*.nc"))

    # ds_diff = xr.open_mfdataset(
    #     files,
    #     engine="netcdf4",
    #     chunks={"station": 64, "time": 52560, "cmip6": 1},
    #     combine="by_coords",
    # )

    # Load the data

    files = glob(os.path.join(dir_in, cnty, "*.nc"))
    ds_diff = xr.open_mfdataset(files, engine="h5netcdf", parallel=True)

    # Some of the ds_diff values got limited by the interger conversion
    ds_diff = ds_diff.where((ds_diff >= -2000000000) & (ds_diff <= 2000000000))

    # ===============================================================================
    # Process and convert to new dataset
    # ===============================================================================

    # Interpolate to hourly
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
        join="inner",  # <-- intersection only
        exclude=[],
    )

    # Confirm uniqueness *again* (should be fine)
    ds_full = ensure_unique_sorted_time(ds_full)
    ds_tidal = ensure_unique_sorted_time(ds_tidal)
    ds_diff = ensure_unique_sorted_time(ds_diff)

    # Create a new dataset to hold the data
    print("Creating dataset...")
    ds_era5 = ds_full.copy()
    ds_era5["ntr"] = ds_full["waterlevel"] - ds_tidal["waterlevel"]

    # Drop time from lat lon and bedlevel
    # Safe: strip the 'time' dimension from lon/lat/bedlevel without altering the Dataset coords
    for var in ["lon", "lat"]:
        da = ds_era5[var].copy()

        # Ensure these are pure float variables with no packing
        da = da.astype("float32")  # Cast to float32

        # Remove any inherited packing from attrs and encoding
        for key in ("ScaleFactor", "scale_factor", "add_offset", "_FillValue", "dtype"):
            da.attrs.pop(key, None)

        # Critically: clear the *encoding* dict; this is what NetCDF writing uses
        da.encoding.clear()

        # Put it back
        ds_era5[var] = da

    # Add the cmip6 difference to the dataset
    ds_diff = ds_diff.reindex(
        time=ds_full["time"], station=ds_full["station"], method=None
    )

    ds_era5["wl_CmipDiff"] = ds_diff["cmip_diff"]

    # Deal with scaling (Not done with _ScaleValue for original dataset)
    ds_era5["wl_CmipDiff"] = ds_era5["wl_CmipDiff"] / 10000
    ds_era5["waterlevel"] = ds_era5["waterlevel"] / 10000
    ds_era5["ntr"] = ds_era5["ntr"] / 10000

    # Remove SLR
    ds_era5["waterlevel"] = ds_era5["waterlevel"] - (int(SLR) / 100)

    ds_era5["waterlevel"].attrs = {
        "units": "meters",
        "standard_name": "sea_surface_height_above_reference_datum",
        "long_name": "water level",
        "reference": "NAVD88",
        "desc": "Modelled Water levels for the reanalysis period",
        "note": "Variable scaled with _ScaleFactor. Check if applied correctly by your software",
        "precision": "Data encoded as integer with 4 significant digits",
    }

    ds_era5["ntr"].attrs = {
        "units": "meters",
        "long_name": "non-tidal residual",
        "desc": "Calculated by subtracting modelled water levels with tidal only forcing from a run with full forcing",
        "reference": "NAVD88",
        "note": "Variable scaled with _ScaleFactor. Check if applied correctly by your software",
        "precision": "Data encoded as integer with 4 significant digits",
    }
    ds_era5["wl_CmipDiff"].attrs = {
        "long_name": "CMIP6 difference in water levels",
        "units": "meters",
        "desc": "Predicted change by each CMIP6 model for each ERA5 water level value. Delta is calculated by subtracting the future period (2015-2050) from the historic period (1950-2014). Change is calculated for each quantile for each month.",
        "usage": "Adding the wl_CmipDiff variable to the waterlevel variable produces the pseudo-global-warming timeseries (experienced climate with climate change delta applied)",
        "note": "Variable scaled with _ScaleFactor. Check if applied correctly by your software",
        "precision": "Data encoded as integer with 4 significant digits",
    }

    ds_era5["wl_quants"].attrs = {
        "units": "None",
        "long_name": "Waterlevel Quantile (Monthly)",
        "desc": "Quantiles (computed for each month) for all data in timeseries within specific month",
        "note": "Variable scaled with _ScaleFactor. Check if applied correctly by your software",
        "precision": "Data encoded as integer with 4 significant digits",
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

    # SEt some attributes to the varialbes
    ds_era5["cmip6"].attrs = {"long_name": "Cmip6 Model (HighResMIP)"}

    # Global Attributes
    ds_era5.attrs["processing_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ds_era5.attrs.pop("forcing", None)
    ds_era5.attrs.pop("source_dir", None)
    ds_era5.attrs["author"] = "Kai Parker (USGS PCMSC)"
    ds_era5.attrs["description"] = (
        "This dataset contains modelled water levels and Non-tidal residual for the reanalysis period. Modelled changes to the reanalysis timeseries (as predicted by CMIP6) are also included. Output is for stations in the Salish Sea"
    )
    ds_era5.attrs["DataReleaseCitation"] = "XXXXXX"
    ds_era5.attrs["ModelCitation"] = "XXXXX"
    ds_era5.attrs["InterpretiveProductCitation"] = "XXXXXX"

    # ===============================================================================
    # Guardrails
    # ================================================================================

    ds_era5 = ds_era5.persist()

    # 1) Time monotonic and no NaT
    assert np.all(~pd.isna(ds_full["time"].values)), "Found NaT in time after decode_cf"
    assert np.all(
        np.diff(ds_full["time"].values.astype("datetime64[ns]"))
        >= np.timedelta64(0, "ns")
    ), "Time is not monotonically increasing"

    # 2) Station unique
    st = pd.Index(ds_full["station"].values)
    assert st.is_unique, "Duplicate station IDs found"

    # 3) Big NaN blocks?
    for v in ["waterlevel", "ntr", "cmip_diff"]:
        if v in ds_full:
            # ratio of NaNs by time slice—shouldn’t be 100% for long blocks
            nan_by_time = ds_full[v].isnull().mean(dim="station")
            bad = (nan_by_time > 0.95).load()  # compute
            if bad.any():
                print(
                    f"WARNING: {v} has time slices with >95% NaNs. Indices:",
                    np.where(bad.values)[0][:10],
                )

    # ===============================================================================
    # Remove stations using a polygon.
    # ================================================================================

    ds_era5, remove_flag = mask_stations_by_polygon(
        ds_era5,
        os.path.join(dir_gis, "StationRemove.shp"),
        lat_var="lat",
        lon_var="lon",
        predicate="contains",
    )

    print(f"Removed {int(remove_flag.sum().item())} grid cells (across all times).")

    # ===============================================================================
    # Load the county information for spatial grouping
    # ===============================================================================
    print("Loading County shapefiles and finding station subsets...")

    # Load the county shapefile
    counties = gpd.read_file(
        os.path.join(dir_gis, "Washington_Counties_(no_water)___washco_area.shp")
    )
    counties = counties.to_crs(crs="EPSG:4326")

    # Subset to only Salish Sea Counties

    county_list = [
        "Kitsap",
        "Snohomish",
        "Island",
        "Skagit",
        "Jefferson",
        "King",
        "Pierce",
        "Thurston",
        "Whatcom",
        "Mason",
        "San Juan",
        "Clallam",
    ]

    # Normalize COUNTY
    counties["COUNTY_norm"] = counties["COUNTY"].astype(str).map(normalize)

    # Build a regex pattern of alternatives (exact normalized tokens)
    alts = "|".join(re.escape(normalize(c)) for c in county_list)
    pattern = rf"(?:^|.*)({alts})(?:.*|$)"  # contains any of the normalized tokens

    # Filter using contains; na=False to ignore NaNs
    mask = counties["COUNTY_norm"].str.contains(pattern, regex=True, na=False)
    counties = counties[mask].copy()

    # Stations from DFM
    stations = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(ds_era5["lon"].values, ds_era5["lat"].values),
        crs="EPSG:4326",
    )

    Index_DFM = gpd.sjoin(counties, stations, how="right", predicate="intersects")
    Index_DFM = Index_DFM.rename(columns={"index_left": "CountyID"})
    Index_DFM["CountyID"] = Index_DFM["CountyID"].fillna(-999).astype("int32")

    # ===============================================================================
    #  Output
    # ===============================================================================
    # Loop through each county and output a netcdf file for each county with the data from the stations within that county.

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

    # Make sure output directory exists

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
        print("Ouptut Directory created.")

    print("Outputting data...")

    ind_county = pd.Index(list(counties.index) + [-999])  # keep an OutOfCounty bucket

    for county in ind_county:
        county_name = (
            counties.loc[county]["COUNTY_norm"] if county != -999 else "OutOfCounty"
        )

        print(f"Processing county {county_name}...")

        # Select the data for the current county
        dfm_pnts = Index_DFM[Index_DFM["CountyID"] == county]
        ds_county = ds_era5.isel(station=dfm_pnts.index)

        ds_county.attrs["County"] = county_name

        # Output the dataset for this county
        ds_county.to_netcdf(
            os.path.join(
                dir_out, f"Reanalysis_and_Projected_CoSMoSwaterlevels_{county_name}.nc"
            ),
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

    # ===============================================================================
    # Cluster shutdown and cleanup
    # ===============================================================================

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
