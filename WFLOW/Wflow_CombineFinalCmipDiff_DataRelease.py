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
import geopandas as gpd
from dask.distributed import Client, LocalCluster


# ===============================================================================
# %% User Defined inputs
# ===============================================================================
# Directory where the DFM data resides
# dir_in = r'D:\DFM'
dir_in = (
    r"C:\Users\kai\Documents\KaiDownloads\WFLOW\11_20_2025_Discharges_SnohomishKitsap"
)

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

    # ===============================================================================
    # User Defined inputs
    # ===============================================================================

    dir_out = os.path.join(dir_in, cnty, "DataRelease")

    # Number of cpus to put into the cluster.
    # n = os.cpu_count() or 1
    n = 10

    # Packing information

    PACK_SCALE = 1e-4  # meters per integer count (i.e., meters * 1e4)
    FILL_INT = -9999

    os.makedirs(dir_out, exist_ok=True)

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
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Prefer processes for CPU-bound Python work
    cluster = LocalCluster(
        n_workers=n,  # usually #cores
        threads_per_worker=1,
        processes=True,
        silence_logs=False,  # helpful while debugging
    )
    client = Client(cluster)

    print("Dashboard:", cluster.dashboard_link)
    print("Workers:", client.scheduler_info().get("workers", {}).keys())

    # ===============================================================================
    # Load the Q data
    # ===============================================================================

    # Load the data
    print("Loading data...")

    # Load the data

    files = glob(os.path.join(dir_in, cnty, "cdf_diff", "*.nc"))
    ds_diff = xr.open_mfdataset(files, engine="netcdf4", parallel=True)

    ds_diff = ds_diff.persist()

    # ===============================================================================
    # Load Station location information
    # ===============================================================================

    # Load the geojson
    df = gpd.read_file(os.path.join(dir_in, cnty, "gauges_contour.geojson"))

    # Add Lat Lon to xarray file
    lat = np.full(ds_diff["station"].size, np.nan)
    lon = np.full(ds_diff["station"].size, np.nan)
    contour = np.full(ds_diff["station"].size, np.nan)
    stat_geom = []
    for cnt, stat in enumerate(ds_diff["station"].values):
        pull = df.query("fid == @stat")

        lat[cnt] = pull["geometry"].y.to_numpy()[0]
        lon[cnt] = pull["geometry"].x.to_numpy()[0]
        contour[cnt] = pull["contour"].to_numpy()[0]

    ds_diff["lat"] = xr.DataArray(
        data=lat,  # enter data here
        dims=["station"],
        coords={"station": ds_diff["station"]},
        attrs= {
            "standard_name": "latitude",
            "long_name": "y-coordinate of station",
            "projection": "WGS 84",
            "epsg": "4326",
            "units": "degrees_north",
            },
    )

    ds_diff["lon"] = xr.DataArray(
        data=lon,  # enter data here
        dims=["station"],
        coords={"station": ds_diff["station"]},
        attrs= {
            "standard_name": "longitude",
            "long_name": "x-coordinate of station",
            "projection": "WGS 84",
            "epsg": "4326",
            "units": "degree_east",
            },
    )

    ds_diff["contour"] = xr.DataArray(
        data=contour,  # enter data here
        dims=["station"],
        coords={"station": ds_diff["station"]},
        attrs={
            "long_name": "Station Contour (approximate)",
            "units": "m",
            "datum": "NAVD88",
            "desc": "Approximate contour elevation of the station as pulled from the WFLOW network. True elevation at location may differ dramatically due to WFLOW resolution and DEM difference",
        },
    )

    # ===============================================================================
    # Process and convert to new dataset
    # ===============================================================================
    ds_era5 = ds_diff.copy()
    ds_era5 = ds_era5.rename({"cmip_diff": "Q_CmipDiff"})

    ds_era5["Q"].attrs = {
        "units": "m3/s",
        "standard_name": "Discharge",
        "long_name": "Discharge (Q) ",
        "desc": "Discharge Q from WFLOW forced with ERA5 reanalysis",
        "note": "Variable scaled with _ScaleFactor. Check if applied correctly by your software",
        "precision": "Data encoded as integer with 4 significant digits",
    }

    
    ds_era5["Q_CmipDiff"].attrs = {
        "long_name": "CMIP6 difference in Discharge (Q)",
        "units": "m3/s",
        "desc": "Predicted change by each CMIP6 model for each ERA5 discharge value. Delta is calculated by subtracting the future period (2015-2050) from the historic period (1950-2014). Change is calculated for each quantile for each month.",
        "usage": "Adding the Q_CmipDiff variable to the Q variable produces the pseudo-global-warming timeseries (experienced climate with climate change delta applied)",
        "note": "Variable scaled with _ScaleFactor. Check if applied correctly by your software",
        "precision": "Data encoded as integer with 4 significant digits",
    }

    ds_era5["Q_quants"].attrs = {
        "units": "None",
        "long_name": "Discharge Quantile (Monthly)",
        "desc": "Quantiles (computed for each month) for all data in timeseries within specific month",
        "note": "Variable scaled with _ScaleFactor. Check if applied correctly by your software",
        "precision": "Data encoded as integer with 4 significant digits",
    }

    # SEt some attributes to the varialbes
    ds_era5["cmip6"].attrs = {"long_name": "Cmip6 Model (HighResMIP)",
        "description": "Source model for projected wave height difference",
    }


    ds_era5["station"].attrs = {"long_name": "station name"}

    # Global Attributes
    ds_era5.attrs["processing_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ds_era5.attrs.pop("DataSource", None)
    ds_era5.attrs["author"] = "Wflow team (Joost Buitink & Brendan Dalmijn, Deltares) and USGS (Kai Parker, PCMSC)"
    ds_era5.attrs["description"] = (
        "This dataset contains modelled Discharge (Q) for the reanalysis period. Modelled changes to the reanalysis timeseries (as predicted by CMIP6) are also included. Output is for stations in the Salish Sea"
    )
    ds_era5.attrs["DataReleaseCitation"] = "XXXXXX"
    ds_era5.attrs["ModelCitation"] = "XXXXX"
    ds_era5.attrs["InterpretiveProductCitation"] = "XXXXXX"

    # ===============================================================================
    # Guardrails
    # ================================================================================

    ds_era5 = ds_era5.persist()

    # 1) Time monotonic and no NaT
    assert np.all(~pd.isna(ds_era5["time"].values)), "Found NaT in time after decode_cf"
    assert np.all(
        np.diff(ds_era5["time"].values.astype("datetime64[ns]"))
        >= np.timedelta64(0, "ns")
    ), "Time is not monotonically increasing"

    # 2) Station unique
    st = pd.Index(ds_era5["station"].values)
    assert st.is_unique, "Duplicate station IDs found"

    # 3) Big NaN blocks?
    for v in ["Q", "Q_quants"]:
        if v in ds_era5:
            # ratio of NaNs by time slice—shouldn’t be 100% for long blocks
            nan_by_time = ds_era5[v].isnull().mean(dim="station")
            bad = (nan_by_time > 0.95).load()  # compute
            if bad.any():
                print(
                    f"WARNING: {v} has time slices with >95% NaNs. Indices:",
                    np.where(bad.values)[0][:10],
                )
    
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

    # Output the dataset for this county
    ds_era5.to_netcdf(
        os.path.join(
            dir_out, f"Reanalysis_and_Projected_WFLOWdischarges_{cnty}.nc"
        ),
        engine="netcdf4",
        encoding={
            "Q": int_encoding,
            "Q_CmipDiff": int_encoding,
            "Q_quants": int_encoding,
            "lon": coord_float_encoding,
            "lat": coord_float_encoding,
            "contour": coord_float_encoding,
        },
    )

    # ===============================================================================
    # Cluster shutdown and cleanup
    # ===============================================================================

    client.close()
    cluster.close()

    print('Done! ')


if __name__ == "__main__":
    main()
