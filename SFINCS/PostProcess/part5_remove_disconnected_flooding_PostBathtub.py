# Modules => runs on hydromt-sfincs-dev
import numpy as np
from scipy import ndimage as ndi
import rasterio
import matplotlib.pyplot as plt
import fiona
from shapely.geometry import shape
from shapely.geometry import box
from rasterio import features
import time
from os.path import join

destin_main = r"C:\Users\kai\Documents\KaiRuns"
# RPs = ["RP001", "RP002", "RP005", "RP010", "RP020", "RP050", "RP100"]
RPs = ["RPdaily"]
# SLRs = ["000", "025", "050", "100", "150", "200", "300"]
SLRs = [ "100"]
counties = ["02_Pierce"]
sub_categories = ["_low"]


# Some processing of the inputs
sub_categories = [sub.replace("_median", "") for sub in sub_categories]
# RPs = [int(RP) for RP in RPs]

# Base setting per county
res_name = "_2m"
name_iteration = "downscaled_2m"
hh_min = 0.02
destout_main = join(destin_main, "PostProcess")
polygon_path = r"C:\Users\kai\Documents\KaiRuns\shapefiles\waterlevel_polygon.shp"

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
            destout = join(destout_main, county, TMP_string, name_iteration)

            # Define return periods (and SLRs - later)
            for rp in RPs:
                # Variable names
                start_time = time.time()
                tiff_wanted_full = join(destout, f"depth_{rp[2:]}{res_name}.tif")
                tiff_wanted_bathtub = join(
                    destout, f"depth_{rp[2:]}{res_name}_attentuated.tif"
                )
                tiff_out = join(
                    destout, f"connection_{rp[2:]}{res_name}_attentuated.tif"
                )

                print(f"Started with {tiff_out}")

                # Open the original depth file and read
                with rasterio.open(tiff_wanted_full) as src:
                    # Get data
                    img1 = src.read()  # Read all bands of the image
                    raster_crs = src.crs
                    raster_transform = src.transform

                # Open the attennuated bathtub depth file and read
                with rasterio.open(tiff_wanted_bathtub) as src:
                    # Get data
                    img2 = src.read()  # Read all bands of the image
                    raster_crs = src.crs
                    raster_transform = src.transform

                # Overlay the two images to get a combined image
                img = np.nanmax(np.concatenate((img1, img2), axis=0), axis=0)

                # Calculate the wet dry mask
                wet_dry = np.squeeze(img) > hh_min  # check wet/dry
                s = ndi.generate_binary_structure(2, 2)
                labeled_array, num_features = ndi.label(wet_dry, structure=s)
                criteria_flooded = np.zeros(np.shape(labeled_array))
                criteria_flooded[wet_dry] = int(2)  # so all disconnected, unless ...

                # Check
                # downsampled_array = labeled_array[::100, ::100]
                # plt.figure(figsize=(10, 6))
                # plt.pcolormesh(downsampled_array, cmap='viridis')
                # plt.colorbar()  # Add a color bar to the plot to show the scale
                # plt.title('Downsampled Labeled Array')
                # plt.xlabel('Reduced Column Index')
                # plt.ylabel('Reduced Row Index')
                # plt.show()

                # Open the polygon shapefile
                with fiona.open(polygon_path, "r") as shp:
                    for polygon_counter, feature in enumerate(shp):
                        try:
                            # Read each polygon
                            feature = shp[polygon_counter]
                            polygon = shape(feature["geometry"])

                            # Create a plot
                            # fig, ax = plt.subplots()
                            # plt.plot(polygon.boundary.xy[0], polygon.boundary.xy[1])

                            # Applu mask and find
                            mask = features.geometry_mask(
                                [polygon],
                                out_shape=wet_dry.shape,
                                transform=src.transform,
                                invert=True,
                            )
                            labels_inside_polygon = labeled_array[mask]
                            unique_labels_inside_polygon = np.unique(
                                labels_inside_polygon
                            )
                            for label in unique_labels_inside_polygon:
                                if label != 0:
                                    criteria_flooded[labeled_array == label] = int(1)

                        # Get statement
                        except Exception as e:
                            exception = 1
                            # print(f" => error processing polygon {polygon_counter + 1}: {e}")

                # Count how many of them
                counts = np.bincount(labeled_array.flatten())

                # Iterate over all features, ignoring the background label 0
                # max_region = int(1000*100 / 4 / 2)
                # for label in range(1, num_features + 1):
                #    if counts[label] > max_region:
                #        # Change criteria_flooded for all pixels corresponding to the current label
                #        criteria_flooded[labeled_array == label] = 3  # Set to 1 if feature size is greater than 100 pixels

                # Define writing
                kwargs2 = dict(
                    driver="GTiff",
                    height=criteria_flooded.shape[0],
                    width=criteria_flooded.shape[1],
                    count=1,
                    dtype="int32",
                    crs=raster_crs,
                    transform=raster_transform,
                    tiled=True,
                    blockxsize=128,
                    blockysize=128,
                    compress="deflate",
                    zlevel=6,  # Higher compression level for deflate
                    profile="COG",
                )

                # Done here so let's print
                data_int = criteria_flooded.astype("int32")
                with rasterio.open(tiff_out, "w", **kwargs2) as dst:
                    dst.write(data_int, 1)  # Write the modified array as the first band


# Done with the entire things
print("done!")
