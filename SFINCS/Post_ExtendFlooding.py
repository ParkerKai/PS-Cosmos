# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 14:59:37 2025

This extends the disconnected flooding regions for SFINCS

@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"

# Import Modules
import numpy as np
from scipy import ndimage as ndi
import rasterio
import matplotlib
import fiona
from shapely.geometry import shape
from shapely.geometry import box
from rasterio import features
import time
from os.path import join
import xarray as xr 


#===============================================================================
# %% User Defined inputs
#===============================================================================

# Directory to save the downloaded files
dir_in    = r'C:\Users\kai\Documents\KaiRuns\20241018_synthetic_future_withchange_mean_100yr\PostProcess'
dir_bathy = r'C:\Users\kai\Documents\KaiRuns\20241018_synthetic_future_withchange_mean_100yr\AddWeir_20241127'


county = '01_King'
slr    = '000'
rp     = 'RP100'
category  = ''


# Base setting per county
SLRs            = ['000', '025', '050', '100', '150', '200', '300']
#SLRs            = ['']
RPs             = ['RP001', 'RP002', 'RP005', 'RP010', 'RP020', 'RP050', 'RP100']
#RPs             = ['RP100']
res_name        = '_2m'
downscale_name  = 'downscaled_2m'
hh_min          = 0.02
destout_main    = r'Y:\PS_Cosmos\02_models\SFINCS\20241018_synthetic_future_withchange_mean_100yr\PostProcess'

#===============================================================================
# %% Load the Bathymetry
#===============================================================================
# Base setting per county
if county.startswith("01_King"): depfile              = join(dir_bathy,'King_2m_AddWeir.tif')
if county.startswith("02_Pierce"): depfile            = join(dir_bathy,'Pierce_2m_AddWeir.tif')



# Open file
#with rasterio.open(depfile) as src:
    
src = rasterio.open(depfile)

# Get data
topo   = src.read()                          # Read all bands of the image
topo   = topo.squeeze()                       # Only one band of data 

height, width = topo.shape #Find the height and width of the array

#Two arrays with the same shape as the input array/raster, where each value is the x or y index of that cell
cols, rows = np.meshgrid(np.arange(width), np.arange(height)) 

#Two arrays with the same shape as the input array/raster, where each value is the x or y coordinate of that cell 
xs, ys = rasterio.transform.xy(src.transform, rows, cols) 

#They are actually lists, convert them to arrays
xcoords = np.array(xs)
ycoords = np.array(ys)


asdf
# Load topo-bathy
depfile                 = xr.open_dataset(depfile_fn)
depfile                 = depfile['band_data'].isel(band=0)
depfile.attrs['crs']    = 'EPSG:6339'
            
            
#===============================================================================
# %% Load the Connected flooding file
#===============================================================================

# Print for each county and SLR combination
print(f'County: {county}, SLR: {slr}, RP: {rp}, category: {category}')

# Define the directory paths specific to the county and SLR
dest  = join(dir_in, county, slr + category, downscale_name)

# Define return periods (and SLRs - later)

# Variable names
start_time          = time.time()
tiff_wanted         = join(dest,f'connection_{rp}{res_name}.tif')
print(f"Started with {tiff_wanted}")

# Open file
with rasterio.open(tiff_wanted) as src:
    
    # Get data
    img         = src.read()                                    # Read all bands of the image
    img         = img.squeeze()
    
    fig, ax = matplotlib.pyplot.subplots(1,figsize=(10, 6))
    m1 = ax.pcolormesh(img, cmap='viridis')
    fig.colorbar(m1)  # Add a color bar to the plot to show the scale
    ax.set_title('Connected Flooding')
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    #plt.show()

    # Open the polygon shapefile
    with fiona.open(polygon_path, 'r') as shp:
        for polygon_counter, feature in enumerate(shp):
            try:

                # Read each polygon
                feature = shp[polygon_counter]
                polygon = shape(feature['geometry'])

                # Create a plot
                #fig, ax = plt.subplots()
                #plt.plot(polygon.boundary.xy[0], polygon.boundary.xy[1])

                # Applu mask and find
                mask                            = features.geometry_mask([polygon], out_shape=wet_dry.shape, transform=src.transform, invert=True)
                labels_inside_polygon           = labeled_array[mask]
                unique_labels_inside_polygon    = np.unique(labels_inside_polygon)
                for label in unique_labels_inside_polygon:
                    if label != 0:
                        criteria_flooded[labeled_array == label] = int(1)

            # Get statement
            except Exception as e:
                exception =1
                #print(f" => error processing polygon {polygon_counter + 1}: {e}")
        
    # Count how many of them
    counts   = np.bincount(labeled_array.flatten())

    # Iterate over all features, ignoring the background label 0
    #max_region = int(1000*100 / 4 / 2)
    #for label in range(1, num_features + 1):
    #    if counts[label] > max_region:
    #        # Change criteria_flooded for all pixels corresponding to the current label
    #        criteria_flooded[labeled_array == label] = 3  # Set to 1 if feature size is greater than 100 pixels

    # Define writing
    kwargs2                 = dict(
        driver="GTiff",
        height=criteria_flooded.shape[0],
        width=criteria_flooded.shape[1],
        count=1,
        dtype='int32',
        crs=src.crs,                # Assuming 'src' is defined from previous operations
        transform=src.transform,    # Assuming 'src' is defined from previous operations
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
        zlevel=9,                   # Higher compression level for deflate
        profile="COG"
    )

    # Done here so let's print
    data_int = criteria_flooded.astype('int32')
    with rasterio.open(tiff_out, 'w', **kwargs2) as dst:
        dst.write(data_int, 1)  # Write the modified array as the first band

    # Process each variable
    print(f" masking")
    variables_wanted        = ['depth', 'zsmax', 'tmax', 'qmax']
    for var_name in variables_wanted:

        # Define names
        input_filename  = join(destout, f'{var_name}_{rp}{res_name}.tif')
        output_filename = join(destout, f'{var_name}_{rp}{res_name}_masked.tif')
        
        with rasterio.open(input_filename) as src_var:

            # Read the variable data
            var_data = src_var.read(1)
            
            # Create a masked version of the variable data
            masked_var = np.where(data_int == 1, var_data, np.nan)
            
            # Copy the metadata from the source, update data type to float (to support NaNs)
            out_meta = src_var.meta.copy()
            out_meta.update(dtype=rasterio.float32, nodata=np.nan)
            out_meta.update(tiled=True,
                            blockxsize=128,                 # reduced this from 256
                            blockysize=128,
                            compress="deflate",
                            predictor=2,                    # Adjust based on your data's nature (floating-point or integer)
                            zlevel=6,                       # reduced to 6 from 9
                            profile="COG")

            # Write the masked data to a new TIFF file
            with rasterio.open(output_filename, 'w', **out_meta) as dst:
                dst.write(masked_var.astype(rasterio.float32), 1)

    # Done with this iteration
    elapsed_time = time.time() - start_time
    print("Elapsed time: %.2f seconds" % elapsed_time)

