# Python script to read and process SFINCS netcdf outputs
# v0.1  Nederhoff   2023-03-07
# v0.3  Nederhoff   2023-11-24



# Modules needed
import numpy as np
import rasterio
import rasterio.mask
from collections import deque
import time
from os.path import join
import os
import fiona

destin = r"C:\Users\kai\Documents\KaiRuns"
# ---------------------------------------------
# SETTINGS
# ---------------------------------------------
counties = ["01_King", "02_Pierce"]
#SLRs = ["000", "025", "050", "100", "150", "200", "300"]
SLRs = ["100"]
#RPs = ["RP001", "RP002", "RP005", "RP010", "RP020", "RP050", "RP100"]
RPs  = ['RPdaily']
res_name = "_2m"
name_iteration = "downscaled_2m"
#sub_categories = ["_median", "_high", "_low"]  # Add more sub-categories as needed
sub_categories = [ "_low"]  # Add more sub-categories as needed


criteria_wet = 0.02
loss_per_cell = (
    0.04  # for 0.2, since resolution is 2m, this equates to 100 cm/km or 0.1 m per 100m
)
print_every = 1000000  # show some progress every X cells
destout_main = r"C:\Users\kai\Documents\KaiRuns\PostProcess"
dest_DEM = r"C:\Users\kai\Documents\KaiRuns\Dem_AddWeir_20241127"
dir_shoreline = r"C:\Users\kai\Documents\KaiRuns\shapefiles"
dir_clipping = r"C:\Users\kai\Documents\KaiRuns\shapefiles\CountyClippingPolygons"


def delete_file(file):
    """
    Delete a file if it exists.
    With Error handling to avoid issues if the file is not found or if it is a directory.
    file: str
        The path to the file to be deleted.
    """
    
    if os.path.exists(file):
        # Check if it's a file and not a directory to avoid IsADirectoryError
        if os.path.isfile(file):
            try:
                os.remove(file)
                print(f"File '{file}' deleted successfully.")
            except PermissionError:
                print(
                    f"Permission denied to delete the file '{file}'."
                )
            except Exception as e:
                print(f"An error occurred while deleting the file: {e}")
        else:
            print(
                f"'{file}' exists, but it is a directory, not a file."
            )
    else:
        print(f"File '{file}' does not exist.")





# Some processing of the inputs
sub_categories = [sub.replace("_median", "") for sub in sub_categories]


# Read shoreline clipping file
with fiona.open(
    os.path.join(dir_shoreline, "WaterEdge_ClippingPolygon2.shp"), "r"
) as shapefile:
    shore_clip = [feature["geometry"] for feature in shapefile]

# ---------------------------------------------
# MAIN LOOP
# ---------------------------------------------
for county in counties:
    # Read clipping polygon
    with fiona.open(
        os.path.join(dir_clipping, f"ClippingPolygon_{county}.shp"), "r"
    ) as shapefile:
        clip = [feature["geometry"] for feature in shapefile]

    for slr in SLRs:
        for category in sub_categories:
            TMP_string = slr + category
            destout = join(destout_main, county, TMP_string, name_iteration)
            dem_file = join(dest_DEM, f"{county.split('_')[1]}_2m_AddWeir.tif")

            for rp in RPs:
                t0 = time.time()
                wl_file = join(destout, f"zsmax_{rp[2:]}{res_name}_masked.tif")
                output_file = join(destout, f"zsmax_{rp[2:]}{res_name}_attentuated.tif")
                depth_output_file = join(destout, f"depth_{rp[2:]}{res_name}_attentuated.tif")
                
                # Delete existing output files to avoid conflicts
                delete_file(output_file)
                delete_file(depth_output_file)


                print(
                    f"Processing: {county}, SLR: {slr}, RP: {rp}, category: {category}"
                )

                # ---------------------------------------------
                # READ DATA
                # ---------------------------------------------
                with rasterio.open(dem_file) as dem_src:
                    DEM = dem_src.read(1, masked=True)
                    dem_profile = dem_src.profile

                with rasterio.open(wl_file) as wl_src:
                    WL = wl_src.read(1, masked=True)
                    wl_profile = wl_src.profile

                    # Load shoreline mask
                    mask_shore, out_transform, _ = rasterio.mask.raster_geometry_mask(
                        wl_src, shore_clip, crop=False, invert=True
                    )

                    # Load the county mask
                    mask_county, out_transform, _ = rasterio.mask.raster_geometry_mask(
                        wl_src, clip, crop=False, invert=False
                    )

                # Combine masks.
                mask_clip = mask_shore | mask_county

                if DEM.shape != WL.shape:
                    print("DEM and water level rasters do not match in shape.")
                    continue

                rows, cols = DEM.shape

                # ---------------------------------------------
                # INITIAL FLOODED CELLS
                # ---------------------------------------------
                depth_initial = WL - DEM
                initial_flooded = (
                    (depth_initial > criteria_wet) & (~DEM.mask) & (~WL.mask)
                )
                num_initial = np.sum(initial_flooded)

                welev = np.full_like(DEM, np.nan, dtype=np.float32)
                welev[initial_flooded] = WL[initial_flooded]
                flooded = np.zeros(DEM.shape, dtype=bool)
                flooded[initial_flooded] = True

                # Only queue subset
                queue_mask = initial_flooded & (DEM > -15.0)  # & (DEM < 10.0)
                queue_rows, queue_cols = np.where(queue_mask)
                queue_water_levels = welev[queue_mask]
                queue = deque(zip(queue_rows, queue_cols, queue_water_levels))
                num_initial = np.sum(queue_mask)

                # Read a masking file of locations to not procress

                print(f" initial flooded cells: {num_initial}")

                # ---------------------------------------------
                # NEIGHBOR OFFSETS
                # ---------------------------------------------
                directions = [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                ]

                # ---------------------------------------------
                # PROPAGATION LOOP
                # ---------------------------------------------
                processed_count = 0
                while queue:
                    row, col, current_wl = queue.popleft()
                    processed_count += 1

                    # Skip if Dem value is below threshold
                    if DEM[row, col] <= -15.0:
                        continue
                    # Skip if shoreline mask says to not process this cell
                    if mask_clip[row, col]:
                        continue

                    for dr, dc in directions:
                        nr = row + dr
                        nc = col + dc

                        if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                            continue
                        if flooded[nr, nc]:
                            continue

                        if DEM.mask[nr, nc]:
                            continue

                        if mask_clip[nr, nc]:
                            continue

                        new_wl = current_wl - loss_per_cell
                        depth = new_wl - DEM[nr, nc]

                        if depth > criteria_wet:
                            flooded[nr, nc] = True
                            welev[nr, nc] = new_wl
                            queue.append((nr, nc, new_wl))

                    if processed_count % print_every == 0:
                        print(f"  processed {processed_count} cells...")

                print(
                    f"Flood propagation completed. Total cells processed: {processed_count}"
                )

                # ---------------------------------------------
                # WRITE OUTPUT (COMPRESSED)
                # ---------------------------------------------
                out_profile = dem_profile.copy()
                out_profile.update(
                    dtype=rasterio.float32,
                    nodata=np.nan,
                    tiled=True,
                    blockxsize=128,
                    blockysize=128,
                    compress="deflate",
                    predictor=2,
                    zlevel=6,
                    profile="COG",
                )

                with rasterio.open(output_file, "w", **out_profile) as dst:
                    dst.write(welev, 1)

                elapsed = time.time() - t0
                print(f"Flooded water level raster saved to:\n{output_file}")

                # ---------------------------------------------
                # WRITE DEPTH OUTPUT
                # ---------------------------------------------
                depth = welev - DEM
                

                with rasterio.open(depth_output_file, "w", **out_profile) as dst:
                    dst.write(depth, 1)

                print(f"Flooded depth raster saved to:\n{depth_output_file}")

                print(f"Total time: {elapsed:.2f} seconds.")

print("Completely done!")
