# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 15:23:16 2025

This script bins the depth product into raster/shapefiles
This is to make a product for Ecology (shapefile of depths)

@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"


# ===============================================================================
# %% Import modules
# ===============================================================================

import numpy as np
import rasterio
import time
import pandas as pd
import os
from glob import glob
import sys
import geopandas as gpd
import scipy

# ===============================================================================
# %% User Defined inputs
# ===============================================================================

destin_main = r"Y:\PS_Cosmos\06_FinalProducts"
RPs = ["daily", "001", "010", "020", "050", "100"]
SLRs = ["000", "025", "050", "100", "150", "200", "300"]
counties = ["01_King", "02_Pierce"]
sub_categories = ["_median", "_low", "_high"]

destout_main = r"Y:\PS_Cosmos\02_models\SFINCS\20250122_synthetic_future_meanchange_100yr_Intel\DepthShapefiles"
# RPs = ["daily"]
# SLRs = ["100"]
# counties = ["02_Pierce"]
# sub_categories = ["_low"]

# ===============================================================================
# %% Define functions
# ===============================================================================
sys.path.append(r"C:\Users\kaparker\GitHub\Python\Kai_Python\General_Functions")
from Kai_GeoTools import raster_to_shape_rasterio


# ===============================================================================
# %% Main
# ===============================================================================

# Some processing of the inputs
sub_categories = [sub.replace("_median", "") for sub in sub_categories]

# Base setting per county
res_name = "_2m"


# Smoothing
smoothing = True
smooth_size = 7  # sigma for gaussian filter

## Depth  Categories
d = {
    "ID": np.array([1, 2, 3, 4, 5], dtype="int16"),
    "Category": ["Low", "Medium", "High", "VeryHigh", "Extreme"],
    "Depth_Label_ft": ["<0.5", "0.5-1.0", "1.0-3.0", "3.0-5.0", ">5.0"],
    "Depth_Label_m": ["<0.15", "0.15-0.3", "0.3-0.9", "0.9-1.5", ">1.5"],
    "D_Min": np.array([-np.inf, 0.1524, 0.3048, 0.9144, 1.524]),
    "D_Max": np.array([0.1524, 0.3048, 0.9144, 1.524, np.inf]),
}


D_Cat = pd.DataFrame(d)
D_Cat = D_Cat.set_index("ID")


# Loop over counties
for index, county in enumerate(counties):
    # Loop over SLRs
    for slr in SLRs:
        # Go over sub_categories (standard, low and high)
        for index_cat, category in enumerate(sub_categories):
            # Print for each county and SLR combination
            print(f"Index: {index}, County: {county}, SLR: {slr}, category: {category}")

            # Define the directory paths specific to the county and SLR

            # Create the junk folder
            junkFol = os.path.join(
                destout_main, county, "junk", slr + category
            )  # Temporary folder
            if not os.path.exists(junkFol):
                # Create Final Merged Raster Folder if it doesn't exist
                os.makedirs(junkFol)
                print("New junk folder created")
            else:
                files = glob(os.path.join(junkFol, "*"))
                for f in files:
                    os.remove(f)
                # print('JUNK IN FOLDER')

            # Define return periods (and SLRs - later)
            for rp in RPs:
                # Variable names
                start_time = time.time()
                if rp == "daily":
                    tiff_in = os.path.join(
                        destin_main,
                        county,
                        f"CoSMoS-PS_flood_depth_projections_{county[3:]}",
                        f"CoSMoS-PS_flood_depth_projections_average_conditions_{county[3:]}",
                        f"{county[3:]}_fldDpth_slr{slr}_rp000.tif",
                    )
                else:
                    tiff_in = os.path.join(
                        destin_main,
                        county,
                        f"CoSMoS-PS_flood_depth_projections_{county[3:]}",
                        f"CoSMoS-PS_flood_depth_projections_{str(int(rp))}-year_storm_{county[3:]}",
                        f"{county[3:]}_fldDpth_slr{slr}_rp{rp}.tif",
                    )

                tiff_out = os.path.join(junkFol, "smoothDepth.tif")

                dir_out = os.path.join(destout_main, county, slr + category)
                if not os.path.exists(dir_out):
                    # Create Final depth shapefile Folder if it doesn't exist
                    os.makedirs(dir_out)

                shapefile_out = os.path.join(
                    dir_out, f"{county[3:]}_fldDpthBinned_slr{slr}_rp{rp}.shp"
                )

                print(f"Started with {shapefile_out}")

                # Open file
                with rasterio.open(tiff_in) as src:
                    # Get data
                    data = src.read()  # Read all bands of the image (only 1)
                    data = np.squeeze(data)
                    src_transform = src.transform
                    src_crs = src.crs

                if smoothing:
                    print(f"Smooothing with Gaussian filter with sigma {smooth_size}")

                    # Save nans
                    ind_nan = np.isnan(data)

                    V = data.copy()
                    V[np.isnan(data)] = 0
                    VV = scipy.ndimage.gaussian_filter(
                        V, sigma=smooth_size, truncate=2 * smooth_size
                    )

                    W = 0 * data.copy() + 1
                    W[np.isnan(data)] = 0
                    WW = scipy.ndimage.gaussian_filter(
                        W, sigma=smooth_size, truncate=2 * smooth_size
                    )

                    ind_assess = (VV > 0.0000000001) & (WW > 0.0000000001)
                    data = np.full(VV.shape, np.nan, dtype="float32")
                    data[ind_assess] = VV[ind_assess] / WW[ind_assess]
                    # Re-assert nan locations

                    data[ind_nan] = np.nan

                # Categorized data
                data_C = np.full(data.shape, -999, dtype="int16")
                for ID in D_Cat.index:
                    # Data in this Category
                    ind_Cat = (data >= D_Cat["D_Min"].loc[ID]) & (
                        data < D_Cat["D_Max"].loc[ID]
                    )
                    data_C[ind_Cat] = ID

                # Check
                # downsampled_array = labeled_array[::100, ::100]
                # plt.figure(figsize=(10, 6))
                # plt.pcolormesh(downsampled_array, cmap='viridis')
                # plt.colorbar()  # Add a color bar to the plot to show the scale
                # plt.title('Downsampled Labeled Array')
                # plt.xlabel('Reduced Column Index')
                # plt.ylabel('Reduced Row Index')
                # plt.show()

                ## Write the Raster ####
                print("Writing Raster")
                kwargs2 = dict(
                    driver="GTiff",
                    height=data_C.shape[0],
                    width=data_C.shape[1],
                    count=1,
                    dtype="int32",
                    crs=src_crs,  # Assuming 'src' is defined from previous operations
                    transform=src_transform,  # Assuming 'src' is defined from previous operations
                    tiled=True,
                    blockxsize=128,
                    blockysize=128,
                    compress="deflate",
                    predictor=2,
                    zlevel=6,  # Higher compression level for deflate
                    nodata=-999,
                    profile="COG",
                )

                # Done here so let's print
                with rasterio.open(tiff_out, "w", **kwargs2) as dst:
                    dst.write(data_C, 1)  # Write the modified array as the first band

                ####################### Write Shapefile ######################
                print("Writing Shapefile")
                # First write a shapefile with no metadata
                file_out = os.path.join(junkFol, f"depth_binned_{rp}{res_name}.shp")
                raster_to_shape_rasterio(tiff_out, file_out)

                # Then write a shapefile with dates added

                #### Add attributes to the file ######
                D_data = gpd.read_file(file_out)

                # Remove the no data case
                D_data_out = D_data.iloc[(D_data["ID"] >= 0).values]

                Category = np.full(D_data_out["ID"].shape, " ", dtype="U8")
                Bins = np.full(D_data_out["ID"].shape, " ", dtype="U8")
                for ID_sel in np.unique(D_data_out["ID"]):
                    Category[(D_data_out["ID"] == ID_sel)] = D_Cat.loc[ID_sel].Category
                    Bins[(D_data_out["ID"] == ID_sel)] = D_Cat.loc[ID_sel].Depth_Label_ft

                D_data_out["Severity"] = Category
                D_data_out["Range_ft"] = Bins

                D_data_out = D_data_out.drop(columns=["ID"])

                D_data_out.to_file(shapefile_out)

                # Done with this iteration
                elapsed_time = time.time() - start_time
                print("Elapsed time: %.2f seconds" % elapsed_time)


# Done with the entire things
print("done!")
