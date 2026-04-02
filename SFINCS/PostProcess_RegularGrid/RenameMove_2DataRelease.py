# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 10:59:52 2025

This script saves all the data in the final data release naming and folder scheme.


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
import shutil
from glob import glob
import fiona
import rasterio
from rasterio.mask import mask
import numpy as np
import geopandas as gpd

# ===============================================================================
# %% User Defined inputs
# ===============================================================================

# Processing information
dir_base_in = r"Y:\PS_Cosmos\02_models\SFINCS\20250122_synthetic_future_meanchange_100yr_Intel\PostProcess"
RPs = ["daily", "001", "010", "020", "050", "100"]
RP_OutName = [
    "average_conditions",
    "1-year_storm",
    "10-year_storm",
    "20-year_storm",
    "50-year_storm",
    "100-year_storm",
]
SLRs = ["000", "025", "050", "100", "150", "200", "300"]
dir_base_out = r"Y:\PS_Cosmos\06_FinalProducts"

# Settings
county = "02_Pierce"
cnty = "Pierce"

# Clip to MHHW?
clip = False


# ===============================================================================
# %% Define some functions
# ===============================================================================
def copy_shapefile(file_in, file_out):
    files = glob(os.path.splitext(file_in)[0] + ".*")
    for f in files:
        os.path.splitext(file_out)[0]

        f_out = os.path.join(
            os.path.dirname(file_out),
            os.path.splitext(file_out)[0] + os.path.splitext(f)[1],
        )

        shutil.copy(f, f_out)


def copy_clean_shapefile(file_in, file_out):
    gdf = gpd.read_file(file_in)
    gdf = gdf[["geometry"]]
    gdf.index.name = "ID"
    gdf.to_file(file_out)


def copy_extent_shapefile(file_in, file_out):
    gdf = gpd.read_file(file_in)

    if "fldPoly" in file_out:
        gdf["name"] = "FloodExtent"

    elif "min_flood" in file_out:
        gdf["name"] = "LowerLevelFloodUncertainty"

    elif "max_flood" in file_out:
        gdf["name"] = "UpperLevelFloodUncertainty"

    elif "lowlyingPoly" in file_out:
        gdf["name"] = "DisconnectedLowLying"

    gdf = gdf.drop(columns=["DN"])
    gdf.to_file(file_out)


def clean_folder(folder_in):
    # First create folder if it does not exist
    # This is to avoid errors when trying to remove files from a non-existing folder
    if not os.path.exists(folder_in):
        os.makedirs(folder_in)

    # Remove all files in the folder
    # This is to ensure the folder is clean before copying new files
    files = glob(os.path.join(folder_in, "*"))
    for f in files:
        os.remove(f)


def copy_ClippedRaster(file_in, file_out, clip_geom):
    # Load raster (clipped by feature geometry)
    with rasterio.open(file_in) as src:
        data = src.read(1)

        mask_shore, out_transform, _ = rasterio.mask.raster_geometry_mask(
            src, clip_geom, crop=False, invert=True
        )
        crs = src.crs
        trans = src.transform
        out_meta = src.meta.copy()

        # Mask the data based on the MHHW polygon
        data[mask_shore] = -999
        data[np.isnan(data)] = -999

    kwargs2 = dict(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=out_meta["dtype"],
        nodata=-999,
        crs=crs,  # Assuming 'src' is defined from previous operations
        transform=trans,  # Assuming 'src' is defined from previous operations
        tiled=True,
        blockxsize=128,  # reduced this from 256
        blockysize=128,
        compress="deflate",
        predictor=2,  # Adjust based on your data's nature (floating-point or integer)
        zlevel=6,  # reduced to 6 from 9
        profile="COG",
    )

    # Do actual writing
    with rasterio.open(file_out, "w", **kwargs2) as dst:
        dst.write(data, 1)  # Write the first (and only) band


# ===============================================================================
# %% Define some functions
# ===============================================================================

# Read the MHHW clipping file
with fiona.open(
    os.path.join(
        dir_base_in, county, "000", "final_shapefile", "flood_daily_connected.shp"
    ),
    "r",
) as shapefile:
    MHHW_clip = [feature["geometry"] for feature in shapefile]


# ===============================================================================
# %% Load the data
# ===============================================================================


# Loop over SLRs
for cnt, RP in enumerate(RPs):
    # Dealy with "daily" case.  Notated as "daily" but need to convert to 000
    if RP == "daily":
        RP_out = "000"
    else:
        RP_out = RP

    # Get folder names
    dir_flood_depth = os.path.join(
        dir_base_out,
        county,
        f"CoSMoS-PS_flood_depth_projections_{cnty}",
        f"CoSMoS-PS_flood_depth_projections_{RP_OutName[cnt]}_{cnty}",
    )

    dir_flood_duration = os.path.join(
        dir_base_out,
        county,
        f"CoSMoS-PS_flood_duration_projections_{cnty}",
        f"CoSMoS-PS_flood_duration_projections_{RP_OutName[cnt]}_{cnty}",
    )

    dir_water_elevation = os.path.join(
        dir_base_out,
        county,
        f"CoSMoS-PS_water_elevation_projections_{cnty}",
        f"CoSMoS-PS_water_elevation_projections_{RP_OutName[cnt]}_{cnty}",
    )

    dir_velocity_depth = os.path.join(
        dir_base_out,
        county,
        f"CoSMoS-PS_velocity_hazard_projections_{cnty}",
        f"CoSMoS-PS_velocity_hazard_projections_{RP_OutName[cnt]}_{cnty}",
    )

    dir_extent = os.path.join(
        dir_base_out,
        county,
        f"CoSMoS-PS_flood_extent_and_uncertainty_projections_{cnty}",
        f"CoSMoS-PS_flood_extent_projections_{RP_OutName[cnt]}_{cnty}",
    )

    dir_uncertainty = os.path.join(
        dir_base_out,
        county,
        f"CoSMoS-PS_flood_extent_and_uncertainty_projections_{cnty}",
        f"CoSMoS-PS_flood_uncertainty_projections_{RP_OutName[cnt]}_{cnty}",
    )

    # Clean folders
    clean_folder(dir_flood_depth)
    clean_folder(dir_flood_duration)
    clean_folder(dir_water_elevation)
    clean_folder(dir_velocity_depth)
    clean_folder(dir_extent)
    clean_folder(dir_uncertainty)

    # Loop over SLRs
    for SLR in SLRs:
        print(f"Processing SLR {SLR}, Return Period {RP}")

        # Directories
        dir_in = os.path.join(dir_base_in, county, SLR)
        dir_out = os.path.join(dir_base_out, county)

        # Depth raster
        file_in = os.path.join(dir_in, "downscaled_2m", f"depth_{RP}_2m_masked.tif")
        file_out = os.path.join(
            dir_flood_depth,
            f"{cnty}_fldDpth_slr{SLR}_rp{RP_out}.tif",
        )
        if clip:
            copy_ClippedRaster(file_in, file_out, MHHW_clip)
        else:
            shutil.copy(file_in, file_out)

        # Duration raster
        file_in = os.path.join(dir_in, "downscaled_2m", f"tmax_binned_{RP}_2m.tif")
        file_out = os.path.join(
            dir_flood_duration, f"{cnty}_duration_slr{SLR}_rp{RP_out}.tif"
        )

        if clip:
            copy_ClippedRaster(file_in, file_out, MHHW_clip)
        else:
            shutil.copy(file_in, file_out)

        # Water Elevation raster
        file_in = os.path.join(dir_in, "downscaled_2m", f"zsmax_{RP}_2m_masked.tif")
        file_out = os.path.join(
            dir_water_elevation, f"{cnty}_wsel_slr{SLR}_rp{RP_out}.tif"
        )

        shutil.copy(file_in, file_out)

        # Vel-Depth shapefile
        file_in = os.path.join(dir_in, "final_shapefile", f"qmax_binned_{RP}_2m.shp")
        file_out = os.path.join(
            dir_velocity_depth, f"{cnty}_velHzrd_slr{SLR}_rp{RP_out}.tif"
        )

        copy_shapefile(file_in, file_out)

        # Extent shapefile
        file_in = os.path.join(dir_in, "final_shapefile", f"flood_{RP}_connected.shp")
        file_out = os.path.join(dir_extent, f"{cnty}_fldPoly_slr{SLR}_rp{RP_out}.shp")

        # copy_shapefile(file_in, file_out)
        copy_extent_shapefile(file_in, file_out)

        # Extent shapefile
        file_in = os.path.join(
            dir_base_in,
            county,
            f"{SLR}_low",
            "final_shapefile",
            f"flood_{RP}_connected.shp",
        )
        file_out = os.path.join(
            dir_uncertainty, f"{cnty}_min_flood_slr{SLR}_rp{RP_out}.shp"
        )

        # copy_shapefile(file_in, file_out)
        copy_extent_shapefile(file_in, file_out)

        file_in = os.path.join(
            dir_base_in,
            county,
            f"{SLR}_high",
            "final_shapefile",
            f"flood_{RP}_connected.shp",
        )
        file_out = os.path.join(
            dir_uncertainty, f"{cnty}_max_flood_slr{SLR}_rp{RP_out}.shp"
        )

        # copy_shapefile(file_in, file_out)
        copy_extent_shapefile(file_in, file_out)

        file_in = os.path.join(
            dir_in, "final_shapefile", f"flood_{RP}_disconnected.shp"
        )
        file_out = os.path.join(
            dir_extent, f"{cnty}_lowlyingPoly_slr{SLR}_rp{RP_out}.shp"
        )

        # copy_shapefile(file_in, file_out)
        copy_extent_shapefile(file_in, file_out)
