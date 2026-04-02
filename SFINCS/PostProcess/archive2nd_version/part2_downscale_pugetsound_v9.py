# Load modules
import os
from os.path import join
os.environ['USE_PYGEOS'] = '0'
from hydromt_sfincs import SfincsModel, utils
import xarray as xr
import numpy as np
import time
import rasterio
import scipy.ndimage as ndimage

# Settings
counties                = ['02_Pierce']
SLRs                    = ['000', '025', '050', '100', '150', '200', '300']
#SLRs                    = ['']      # can be used for ERA5 (i.e. no SLR)
RPs                     = ['RP001', 'RP002', 'RP005', 'RP010', 'RP020', 'RP050', 'RP100']
sub_categories          = [''] #, '_low_v2', '_high_v2']
hmin_value              = 0.02
smoothing               = False
smooth_size             = 3
destin_main             = r'Y:\PS_Cosmos\02_models\SFINCS\20241018_synthetic_future_withchange_mean_100yr\PostProcess'
destout_main            = r'Y:\PS_Cosmos\02_models\SFINCS\20241018_synthetic_future_withchange_mean_100yr\PostProcess'
name_iteration          = 'downscaled_2m'
res_name                = '_2m'

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
            destin          = join(destin_main, county, TMP_string)
            destout         = join(destout_main, county, TMP_string, name_iteration)

            # Base setting per county
            if county.startswith("01_King"): depfile_fn              = r'Y:\PS_Cosmos\01_data\topo_bathymetry\DEM\Pierce_King\AddWeir_20241127\King_2m_AddWeir.tif'
            if county.startswith("02_Pierce"): depfile_fn            = r'Y:\PS_Cosmos\01_data\topo_bathymetry\DEM\Pierce_King\AddWeir_20241127\Pierce_2m_AddWeir.tif'

            # Load topo-bathy
            depfile                 = xr.open_dataset(depfile_fn)
            depfile                 = depfile['band_data'].isel(band=0)
            depfile.attrs['crs']    = 'EPSG:6339'

            # Create destout directory if it doesn't exist
            if not os.path.exists(destout):
                os.makedirs(destout)

            # Box filter
            def nan_box_filter(input_array, size=3):
                def nanmean_filter(values):
                    return np.nanmean(values)

                footprint = np.ones((size, size))
                return ndimage.generic_filter(input_array, nanmean_filter, footprint=footprint, mode='nearest')

                return output_data


            # Define return periods (and SLRs - later)
            for rp in RPs:

                # Variable names
                start_time              = time.time()
                fnc                     = join(destin, f'processed_SFINCS_output_{rp}.nc')
                floodmap_fn             = join(destout, f'depth_{rp}{res_name}.tif')
                floodmap_fn_compress    = join(destout, f'depth_{rp}{res_name}_compressed_redo.tif')
                print(f"Started with {floodmap_fn}")

                # Load netcdf results for water level
                ds                      = xr.open_dataset(fnc)
                zsmax                   = ds['zsmax']
                zsmax.attrs['crs']      = 'EPSG:6339'

                # Downscale results per return period (using default to produce outputs)
                if (smoothing == True): 
                    zsmax.values                   = nan_box_filter(zsmax.values, 3)            # only on values and keep as rasterio
                    utils.downscale_floodmap(zsmax, depfile, floodmap_fn=floodmap_fn, hmin=hmin_value, reproj_method='bilinear')
                else:
                    utils.downscale_floodmap(zsmax, depfile, floodmap_fn=floodmap_fn, hmin=hmin_value)

                # Get a logic / wet or dry from the recently downscaled tiff
                with rasterio.open(floodmap_fn) as src:
                    data        = src.read(1)
                    mask        = data > hmin_value
                    mask_float  = np.where(mask, 1, np.nan)

                # Also provide downscaled water level, qmax and tmax
                variables_wanted        = ['zsmax', 'tmax', 'qmax']
                for var_name in variables_wanted:

                    # Get data
                    print(f" => working now on {var_name}")
                    var_data                = ds[var_name]  # Assuming var_data is accessible this way
                    var_data.attrs['crs']   = 'EPSG:6339'
                    if (smoothing == True): 
                        var_data.values         = nan_box_filter(var_data.values, 3)            # only on values and keep as rasterio
                        var_data                = var_data.raster.reproject_like(depfile, method="bilinear") 
                    else:
                        var_data                = var_data.raster.reproject_like(depfile, method="nearest")  

                    # Apply the mask
                    var_data                = var_data.astype('float32')            # single precision => could try float16
                    masked_var              = var_data * mask_float
                    masked_var              = masked_var.astype('float32')          # single precision => could try float16

                    # Define writing
                    kwargs2                 = dict(
                        driver="GTiff",
                        height=masked_var.shape[0],
                        width=masked_var.shape[1],
                        count=1,
                        dtype='float32',
                        crs=src.crs,  # Assuming 'src' is defined from previous operations
                        transform=src.transform,  # Assuming 'src' is defined from previous operations
                        tiled=True,
                        blockxsize=128,                 # reduced this from 256
                        blockysize=128,
                        compress="deflate",
                        predictor=2,                    # Adjust based on your data's nature (floating-point or integer)
                        zlevel=6,                       # reduced to 6 from 9
                        profile="COG"
                    )
                    
                    # Do actual writing
                    output_filename = join(destout, f'{var_name}_{rp}{res_name}.tif')
                    with rasterio.open(output_filename, 'w', **kwargs2) as dst:
                        dst.write(masked_var, 1)  # Write the first (and only) band

                # Done with this iteration
                ds.close()
                elapsed_time = time.time() - start_time
                print("Elapsed time: %.2f seconds" % elapsed_time)


# Done with the entire things
print('done!')