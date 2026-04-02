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

# Base setting per county
counties        = ['01_King', '02_Pierce']
SLRs            = ['000', '025', '050', '100', '150', '200', '300']
#SLRs            = ['']
RPs             = ['RP001', 'RP002', 'RP005', 'RP010', 'RP020', 'RP050', 'RP100']
#RPs             = ['RP100']
res_name        = '_2m'
name_iteration  = 'downscaled_2m'
hh_min          = 0.02
destout_main    = r'Y:\PS_Cosmos\02_models\SFINCS\20241018_synthetic_future_withchange_mean_100yr\PostProcess'
polygon_path    = r'Y:\PS_Cosmos\02_models\SFINCS\shapefiles'
sub_categories  = ['', '_low_v2', '_high_v2']
sub_categories  = ['']

# Loop over counties
for index, county in enumerate(counties):

    # Loop over SLRs
    for slr in SLRs:
        
        # Go over sub_categories (standard, low and high)
        for index_cat, category in enumerate(sub_categories):

            # Print for each county and SLR combination
            print(f'Index: {index}, County: {county}, SLR: {slr}, category: {category}')

            # Define the directory paths specific to the county and SLR
            TMP_string      = slr + category
            destout         = join(destout_main, county, TMP_string, name_iteration)

            # Define return periods (and SLRs - later)
            for rp in RPs:

                # Variable names
                start_time          = time.time()
                tiff_wanted         = join(destout, f'depth_{rp}{res_name}.tif')
                tiff_out            = join(destout, f'connection_{rp}{res_name}.tif') 
                print(f"Started with {tiff_out}")

                # Open file
                with rasterio.open(tiff_wanted) as src:
                    
                    # Get data
                    img         = src.read()                                    # Read all bands of the image
                    wet_dry     = np.squeeze(img) > hh_min                      # check wet/dry
                    s           = ndi.generate_binary_structure(2, 2)
                    labeled_array, num_features     = ndi.label(wet_dry, structure=s)
                    criteria_flooded                = np.zeros(np.shape(labeled_array))
                    criteria_flooded[wet_dry]       = int(2)        # so all disconnected, unless ...
                    
                    # Check
                    #downsampled_array = labeled_array[::100, ::100]
                    #plt.figure(figsize=(10, 6))
                    #plt.pcolormesh(downsampled_array, cmap='viridis')
                    #plt.colorbar()  # Add a color bar to the plot to show the scale
                    #plt.title('Downsampled Labeled Array')
                    #plt.xlabel('Reduced Column Index')
                    #plt.ylabel('Reduced Row Index')
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

# Done with the entire things
print('done!')