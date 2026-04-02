# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 11:17:49 2025

This script bins the duration product into raster/shapefiles

@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"


# ===============================================================================
# %% Imoprt modules
# ===============================================================================

import numpy as np
import rasterio
import time
import pandas as pd
import os
import sys
import scipy

# ===============================================================================
# %% User Defined inputs
# ===============================================================================

destin_main = (
    r"Y:\PS_Cosmos\02_models\SFINCS\20250122_synthetic_future_meanchange_100yr_Intel"
)
# RPs = ["daily", "001", "002", "005", "010", "020", "050", "100"]
# SLRs = ["000", "025", "050", "100", "150", "200", "300"]
# counties = ["01_King", "02_Pierce"]
# sub_categories = ["_median", "_low", "_high"]

RPs                     = ['daily']
SLRs                    = ['100']
counties                = ["02_Pierce"]
sub_categories          = ['_low']

# ===============================================================================
# %% Define functions
# ===============================================================================
sys.path.append(
    r"C:\Users\kaparker\GitHub\Python\Kai_Python\General_Functions"
)
from Kai_GeoTools import raster_to_shape_rasterio

# ===============================================================================
# %% Main
# ===============================================================================

# Some processing of the inputs
sub_categories = [sub.replace("_median", "") for sub in sub_categories]

# Base setting per county
res_name = "_2m"
name_iteration = "downscaled_2m"
destout_main = os.path.join(destin_main, "PostProcess")

smoothing = True
smooth_size = 7  # sigma for gaussian filter

## Duration Categories
d = {
    "ID": np.array([1, 2, 3, 4, 5, 6, 7], dtype="int16"),
    "Category": [
        "Hours/year",
        "days/year",
        "weeks/year",
        "months/year",
        "Season/year",
        "MultipleSeasons/year",
        "ContinuallyFlooded",
    ],
    "Dur_Label": ["< 1", "1-4", "4-10", "10-40", "40-100", "100-200", ">200"],
    "Dur_Min": np.array([0, 1, 4, 10, 40, 120, 240]) * 24,
    "Dur_Max": np.array([1, 4, 10, 40, 120, 240, np.inf]) * 24,
}

Dur_Cat = pd.DataFrame(d)
Dur_Cat = Dur_Cat.set_index("ID")


# Loop over counties
for index, county in enumerate(counties):
    # Loop over SLRs
    for slr in SLRs:
        # Go over sub_categories (standard, low and high)
        for index_cat, category in enumerate(sub_categories):
            # Print for each county and SLR combination
            print(f"Index: {index}, County: {county}, SLR: {slr}, category: {category}")

            # Define the directory paths specific to the county and SLR
            TMP_string = slr + category
            destout = os.path.join(destout_main, county, TMP_string, name_iteration)

            # Create the junk folder
            junkFol = os.path.join(
                destout_main, county, "junk", TMP_string
            )  # Temporary folder
            if not os.path.exists(junkFol):
                # Create Final Merged Raster Folder if it doesn't exist
                os.makedirs(junkFol)
                print("New junk folder created")
            else:
                # files = glob(os.path.join(junkFol,'*'))
                # for f in files:
                #     os.remove(f)
                print("JUNK IN FOLDER")

            # Define return periods (and SLRs - later)
            for rp in RPs:
                # Variable names
                start_time = time.time()
                tiff_in = os.path.join(destout, f"tmax_{rp}{res_name}_masked.tif")
                tiff_out = os.path.join(destout, f"tmax_binned_{rp}{res_name}.tif")
                shapefile_out = os.path.join(
                    destout_main,
                    county,
                    TMP_string,
                    "final_shapefile",
                    f"tmax_binned_{rp}{res_name}.shp",
                )

                print(f"Started with {tiff_out}")

                # Open file
                with rasterio.open(tiff_in) as src:
                    # Get data
                    data = src.read()  # Read all bands of the image (only 1)
                    data = np.squeeze(data)
                    src_transform = src.transform
                    src_crs = src.crs

                    data[data == 0] = np.nan

                # Deal with tooo many hours in the output.
                # set everything over a full year to a full year.
                data[data > 8760] = 8760

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

                data_C = np.rint(data)
                data_C = np.nan_to_num(data_C, nan=-999)
                data_C = data_C.astype(dtype="int32")

                # Categorized data
                # data_C = np.full(data.shape,-999,dtype='int16')
                # for ID in Dur_Cat.index:
                #     # Data in this Category
                #     ind_Cat = (data >= Dur_Cat['Dur_Min'].loc[ID]) & (data < Dur_Cat['Dur_Max'].loc[ID])
                #     data_C[ind_Cat] = ID

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

                # ####################### Write Shapefile ######################
                # print('Writing Shapefile')
                # # First write a shapefile with no metadata
                # file_out =  os.path.join(junkFol,f'qmax_binned_{rp:03}{res_name}.shp')
                # raster_to_shape_rasterio(tiff_out, file_out)

                # # Then write a shapefile with dates added

                # #### Add attributes to the file ######
                # Dur_data = gpd.read_file(file_out)

                # # Remove the no data case
                # Dur_data_out = Dur_data.iloc[(Dur_data['ID']>=0).values]

                # Category = np.full(Dur_data_out['ID'].shape,' ',dtype='U8')
                # Bins     = np.full(Dur_data_out['ID'].shape,' ',dtype='U8')
                # for ID_sel in np.unique(Dur_data_out['ID']):
                #     Category[(Dur_data_out['ID'] == ID_sel)] = Dur_Cat.loc[ID_sel].Category
                #     Bins[(Dur_data_out['ID'] == ID_sel)] = Dur_Cat.loc[ID_sel].Dur_Label

                # Dur_data_out['Duration'] = Category
                # Dur_data_out['Range'] = Bins

                # Dur_data_out.to_file(shapefile_out)

                # Done with this iteration
                elapsed_time = time.time() - start_time
                print("Elapsed time: %.2f seconds" % elapsed_time)


# Done with the entire things
print("done!")

# %%
