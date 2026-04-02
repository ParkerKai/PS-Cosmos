# Load modules
import os
from os.path import join
os.environ['USE_PYGEOS'] = '0'

from hydromt_sfincs import utils
import xarray as xr
import numpy as np
import time
import rasterio
import scipy.ndimage as ndimage
import fiona

destin_main             =  r'C:\Users\kai\Documents\KaiRuns'
RPs                     =['1', '2', '5', '10', '20', '50', '100']
SLRs                    = ['000', '025', '050', '100', '150', '200', '300']
counties                = ["01_King","02_Pierce"]
sub_categories          = ['_median','_low','_high']

#Some processing of the inputs
sub_categories = [sub.replace('_median', '') for sub in sub_categories]
RPs = [int(RP) for RP in RPs]

# Settings

hmin_value              = 0.02
smoothing               = False
smooth_size             = 3
clipping                = True                                                                                          

#destin_main             = r'/caldera/hovenweep/projects/usgs/hazards/pcmsc/cosmos/PugetSound/sfincs/20241018_synthetic_future_withchange_mean_100yr/PostProcess'
dest_DEM                = r'C:\Users\kai\Documents\KaiRuns\Dem_AddWeir_20241127'
dir_clipping            = r'C:\Users\kai\Documents\KaiRuns\shapefiles\CountyClippingPolygons'
dir_shoreline           = r'C:\Users\kai\Documents\KaiRuns\shapefiles'
name_iteration          = 'downscaled_2m'
res_name                = '_2m'

include_qmax    = 1
include_tmax    = 1
include_tmax_zs = 1

# Loop over counties
for index, county in enumerate(counties):
    
    if clipping:
        # Read clipping polygon
        with fiona.open(os.path.join(dir_clipping,f'ClippingPolygon_{county}.shp'), "r") as shapefile:                        
                clip = [feature["geometry"] for feature in shapefile]
        
        # Read shoreline clipping file
        with fiona.open(os.path.join(dir_shoreline,'WaterEdge_ClippingPolygon2.shp'), "r") as shapefile:                        
                shore_clip = [feature["geometry"] for feature in shapefile]
        

    # Loop over SLRs
    for slr in SLRs:
        
        # Go over sub_categories (standard, low and high)
        for index_cat, category in enumerate(sub_categories):

            # Print for each county and SLR combination
            print(f'Index: {index}, County: {county}, SLR: {slr}, category: {category}')
            
            # Define the directory paths specific to the county and SLR
            TMP_string      = slr + category
            destin          = join(destin_main,'PostProcess', county, TMP_string)
            destout         = join(destin_main,'PostProcess', county, TMP_string, name_iteration)
            
            # Base setting per county
            if county.startswith("01_King"): depfile_fn   = os.path.join(dest_DEM,'King_2m_AddWeir.tif')
            if county.startswith("02_Pierce"): depfile_fn = os.path.join(dest_DEM,'Pierce_2m_AddWeir.tif')

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


            # Define return periods (and SLRs - later)
            for rp in RPs:

                # Variable names
                start_time              = time.time()
                fnc                     = join(destin, f'processed_SFINCS_output_RP{rp:03}.nc')
                floodmap_fn             = join(destout, f'depth_{rp:03}{res_name}.tif')
                print(f"Started with {floodmap_fn}")

                # Load netcdf results for water level
                ds                      = xr.open_dataset(fnc)
                zsmax                   = ds['zsmax']
                zsmax.attrs['crs']      = 'EPSG:6339'


                # Remove values of water level below zero
                zsmax.values[zsmax.values < 0] = np.nan


                # Downscale results per return period (using default to produce outputs)
                if (smoothing == True): 
                    zsmax.values                   = nan_box_filter(zsmax.values, 3)            # only on values and keep as rasterio
                    utils.downscale_floodmap(zsmax, depfile, floodmap_fn=floodmap_fn, hmin=hmin_value, reproj_method='bilinear')
                else:
                    utils.downscale_floodmap(zsmax, depfile, floodmap_fn=floodmap_fn, hmin=hmin_value)
                

                # Load raster (clipped by feature geometry)
                with rasterio.open(floodmap_fn) as src:
                    if clipping:
                        data, out_transform = rasterio.mask.mask(src, clip,
                                                                 crop=False,
                                                                 nodata=np.nan,
                                                                 filled=True)
                        
                        mask_shore,out_transform,_ =  rasterio.mask.raster_geometry_mask(src, shore_clip,
                                                                 crop=False,
                                                                 invert = True)
                        
                    else:
                        data = src.read(1)
                
                # Get a logic / wet or dry from the recently downscaled tiff
                mask = data.squeeze()
                mask_float  = np.where(mask > hmin_value, 1, np.nan)
                
                
                # Use shoreline polygon to nan out shoreline area. 
                # Area in mask shore needs to be naned 
                mask_float[mask_shore] = np.nan
                
                # Also provide downscaled water level, qmax and tmax
                variables_wanted        = ['depth','zsmax'] 
                if include_qmax == 1:
                    variables_wanted.append('qmax')

                if include_tmax == 1:
                    variables_wanted.append('tmax')

                if include_tmax_zs == 1:
                    variables_wanted.append('tmax_zs')

                for var_name in variables_wanted:

                    # Get data
                    print(f" => working now on {var_name}")
 
                    if var_name == 'depth':   # Already downscaled and exported
                        var_data = data.squeeze()
                        
                        # Remove previous (unclipped) depth raster
                        if os.path.exists(floodmap_fn):
                            os.remove(floodmap_fn)
                            
                    else:
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
                    output_filename = join(destout, f'{var_name}_{rp:03}{res_name}.tif')
                    with rasterio.open(output_filename, 'w', **kwargs2) as dst:
                        dst.write(masked_var, 1)  # Write the first (and only) band
                    
                    
                # fig, ax = matplotlib.pyplot.subplots(1, 1)
                # ax.pcolormesh(masked_var)
                
                # Done with this iteration
                ds.close()
                elapsed_time = time.time() - start_time
                print("Elapsed time: %.2f seconds" % elapsed_time)


# Done with the entire things
print('done!')
