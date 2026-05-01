# -*- coding: utf-8 -*-
"""
Created on February 16th, 2026

This script combines ERA5 Waves, CMIP6 differences, and tidal only simulation water levels into a single dataset for each station.
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
# %% Import Modules
# ===============================================================================import cdsapi
import os
from glob import glob
import xarray as xr
import geopandas as gpd
import pandas as pd
import re
from datetime import datetime
from dask.distributed import Client, LocalCluster
import numpy as np

# ===============================================================================
# %% Functions
# ===============================================================================


# Helper to normalize strings: remove non-alphanumerics, strip, lowercase
def normalize(s):
    if s is None:
        return None
    return re.sub(r"\W+", "", str(s)).strip().lower()


def main():
    # ===============================================================================
    # User Defined inputs
    # ===============================================================================
    SLR = "025"

    # COUNTIES
    county_list = [
        "Kitsap",
        "Snohomish",
        "IslandCounty",
        "Skagit",
        "Jefferson",
        "King",
        "Pierce",
        "Thurston",
        "Whatcom",
        "Mason",
        "Clallam",
        "SanJuan",
    ]  #

    dir_in = r"Y:\PS_Cosmos\02_models\Wave_LUT\LUT_timeSeries"
    dir_out = r"C:\Temp"

    # Number of cpus to put into the cluster.
    # n = os.cpu_count() or 1
    n = 6

    # Packing information

    PACK_SCALE = 1e-4  # meters per integer count (i.e., meters * 1e4)
    FILL_INT = -9999
    FILL_FLOAT = np.float32(-9999.0)

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
    # Load the data
    # ===============================================================================

    # Load the data

    for county in county_list:
        print(f"Loading data for {county}")

        # ds_era5 = xr.open_mfdataset(
        #     os.path.join(dir_in, f'LUT_output_{county}_ERA5', 'netcdf',f"LUT_output_{county}.nc"),
        #     engine="h5netcdf",
        #     parallel=True,
        #     chunks={"time": 52560, "station": 64},
        #     combine="by_coords",
        # )

        files = glob(os.path.join(dir_in, f"LUT_CMIP6_Diff_{county}", SLR, "*.nc"))

        ds = xr.open_mfdataset(
            files,
            engine="h5netcdf",
            parallel=True,
            chunks={"time": 52560, "station": 64},
            combine="by_coords",
        )

        # Rename label
        ds = ds.rename_vars({"Lat": "lat"})
        ds = ds.rename_vars({"cmip_diff": "hs_CmipDiff"})

        # ===============================================================================
        # Process and convert to new dataset
        # ===============================================================================

        # Create a new dataset to hold the data
        print("Creating dataset...")

        # Drop time from lat lon and bedlevel
        # Safe: strip the 'time' dimension from lon/lat/bedlevel without altering the Dataset coords
        for var in ["lon", "lat"]:
            da = ds[var].isel(
                time=0
            )  # remove time axis by selecting the first time slice
            da = da.reset_coords(
                drop=True
            )  # drop any attached coords from this DataArray only
            ds[var] = da  # assign back

        # Fill Values

        ds["Hs"].attrs = {
            "long_name": "Significant Wave Height",
            "description": "Hs for ERA5 reanalysis periods",
            "units": "meters",
            "precision": "Data encoded as integer with 4 significant digits",
            "note": "Variable scaled with scale_factor . Check if applied correctly by your software",
        }

        ds["Dm"].attrs = {
            "long_name": "Mean wave direction",
            "description": "Dm for ERA5 reanalysis period",
            "units": "degrees",
            "precision": "Data encoded as integer with 4 significant digits",
            "note": "Variable scaled with scale_factor. Check if applied correctly by your software",
            "reference": "Degrees from True North.",
        }

        ds["Tm"].attrs = {
            "long_name": "Mean Wave Period ",
            "description": "Tm for ERA5 reanalysis period",
            "units": "seconds",
            "precision": "Data encoded as integer with 4 significant digits",
            "note": "Variable scaled with scale_factor. Check if applied correctly by your software",
        }

        ds["lon"].attrs = {
            "standard_name": "longitude",
            "long_name": "x-coordinate of station",
            "projection": "WGS 84",
            "epsg": "4326",
            "units": "degrees_east",
        }

        ds["lat"].attrs = {
            "standard_name": "latitude",
            "long_name": "y-coordinate of station",
            "projection": "WGS 84",
            "epsg": "4326",
            "units": "degrees_north",
        }

        ds["hs_quants"].attrs = {
            "units": "1",
            "long_name": "Significant Wave Height Quantile (Computed by Month)",
            "description": "'Quantiles are computed within each month of entire timeseries; for example, all data within every December in the timeseries is used to generate December-quantiles",
            "note": "Variable scaled with scale_factor. Check if applied correctly by your software",
            "precision": "Data encoded as integer with 4 significant digits",
        }

        ds["hs_CmipDiff"].attrs = {
            "long_name": "CMIP6 difference in Significant Wave Height",
            "units": "meters",
            "description": "Predicted change from specified CMIP6 source for each reanalysis (ERA5) wave height value",
            "note": "Variable scaled with scale_factor. Check if applied correctly by your software",
            "precision": "Data encoded as integer with 4 significant digits",
        }

        ds["station"].attrs = {"long_name": "station name"}

        ds["cmip6"].attrs = {
            "long_name": "Cmip6 Model (HighResMIP)",
            "description": "Source model for projected wave height difference",
        }
        # Global Attributes
        ds.attrs["processing_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ds.attrs["author"] = "Kai Parker (USGS PCMSC)"
        ds.attrs["description"] = (
            "This dataset contains modelled bulk wave parameters modeled using a wave lookup table and linear swell propogation for the reanalysis period."
            "Modelled changes to the reanalysis timeseries (as predicted by CMIP6) are also included. Output is for stations in the Salish Sea"
        )
        ds.attrs["DataReleaseCitation"] = "XXXXXX"
        ds.attrs["ModelCitation"] = "XXXXX"
        ds.attrs["InterpretiveProductCitation"] = "XXXXXX"
        del ds.attrs["ProducedBy"]
        del ds.attrs["DataSource"]

        # ===============================================================================
        # Output
        # ===============================================================================
        # Loop through each county and output a netcdf file for each county with the data from the stations within that county.

        ds = ds.persist()

        int_encoding = dict(
            dtype="int32",
            zlib=True,
            shuffle=True,
            complevel=5,
            _FillValue=FILL_INT,
            scale_factor=PACK_SCALE,
            add_offset=0.0,
        )

        float_encoding = dict(
            dtype="float32",
            zlib=True,
            shuffle=True,
            complevel=5,
            _FillValue=FILL_FLOAT,
        )

        print("Outputting data...")

        # Output the dataset for this county
        ds.to_netcdf(
            os.path.join(
                dir_out,
                SLR,
                f"Reanalysis_and_Projected_CoSMoSwaves_{county}_sealevel{SLR}m.nc",
            ),
            engine="netcdf4",
            encoding={
                "Hs": int_encoding,
                "Dm": int_encoding,
                "Tm": int_encoding,
                "hs_CmipDiff": int_encoding,
                "hs_quants": int_encoding,
                "lon": float_encoding,
                "lat": float_encoding,
            },
        )

    # ===============================================================================
    # Cluster shutdown and cleanup
    # ===============================================================================

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
