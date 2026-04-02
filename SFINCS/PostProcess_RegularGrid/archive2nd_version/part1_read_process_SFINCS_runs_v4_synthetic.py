# Python script to read and process SFINCS netcdf outputs
# v0.1  Nederhoff   2023-03-07
# v0.3  Nederhoff   2023-11-24


# Modules needed
import os
import netCDF4 as nc
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import xarray as xr

# User settings
return_period_wanted    = [1, 2, 5, 10, 20, 50, 100]     # number of years requested
hh_criteria             = 0.010001                       # just above the treshold from SFINCS

# Directory
destin                  = os.path.normpath(r'Y:\PS_Cosmos\02_models\SFINCS\20241018_synthetic_future_withchange_mean_100yr')
destout                 = os.path.normpath(r'Y:\PS_Cosmos\02_models\SFINCS\20241018_synthetic_future_withchange_mean_100yr\PostProcess')
domains                 = ['01_King', '02_Pierce']
include_qmax            = 1
include_tmax            = 1
SLRs                    = [0, 0.25, 0.5, 1, 1.5, 2, 3]
SLRs_formatted          = [f"{int(slr*100):03}" for slr in SLRs]
sub_categories          = ['']  # , '_low', '_high'

# Go to folder and loop over domains
for domain in domains:

    # Start with this domain first
    print('Started with ' + domain)
    destin_TMP          = os.path.join(destin, domain)

    # Go over SLRs
    for index, slr in enumerate(SLRs):

        # Go over sub_categories (standard, low and high)
        for index_cat, category in enumerate(sub_categories):
            
            # Make print statement
            print(f" => SLR value: {slr} - " + SLRs_formatted[index] + sub_categories[index_cat])
            string_wanted       = 'SY'
            destin_TMP          = os.path.join(destin, domain)
            TMP_string          = SLRs_formatted[index] + category
            destin_TMP          = os.path.join(destin_TMP, TMP_string)
            files               = os.listdir(destin_TMP)

            # Read netcdf in WY folders
            count_years         = 0
            total_years         = len([file for file in os.listdir(destin_TMP) if file.startswith(string_wanted)])
            estimate_runtimes   = []

            # Define matrix for runtimes
            if index == 0: matrix_runtimes = np.full((total_years, len(SLRs)), np.nan)

            for file in files:
                if file.startswith(string_wanted):

                    # List SFINCS map
                    sfincs_map = os.path.join(destin_TMP, file, 'sfincs_map.nc')

                    # Read netcdf 
                    try:

                        # Open the netcdf file
                        #print(file)
                        dataset = nc.Dataset(sfincs_map, "r")

                        # read the variables of interest and process the
                        zsmax   = dataset.variables["zsmax"][:]
                        zsmax   = np.squeeze(zsmax,0)
                        if include_qmax == 1:
                            qmax    = dataset.variables["qmax"][:]
                            qmax    = np.squeeze(qmax,0)
                        if include_tmax == 1:
                            tmax    = dataset.variables["tmax"][:]
                            tmax    = np.squeeze(tmax,0)

                        # Also read the grid and bed levels
                        x       = dataset.variables["x"][:]
                        y       = dataset.variables["y"][:]
                        zb      = dataset.variables["zb"][:]

                        # Make empty matrix
                        if count_years == 0:
                            zsmax_matrix    = np.full((total_years, np.size(zsmax, 0), np.size(zsmax, 1)), np.nan)
                            if include_qmax == 1:
                                qmax_matrix     = np.full((total_years, np.size(zsmax, 0), np.size(zsmax, 1)), np.nan)
                            if include_tmax == 1:
                                tmax_matrix     = np.full((total_years, np.size(zsmax, 0), np.size(zsmax, 1)), np.nan)
                        
                        # Place files in the dictionary 
                        zsmax_matrix[count_years,:,:] = zsmax
                        if include_qmax == 1:
                            qmax_matrix[count_years,:,:]  = qmax
                        if include_tmax == 1:
                            tmax_matrix[count_years,:,:]  = tmax
                        count_years                   = count_years+1

                        # Read runtime
                        total_runtime = dataset.variables["total_runtime"][:].data[0]
                        if (total_runtime > 86400*7):
                            print(" => simulation not finished - " + file)
                            total_runtime = np.nan
                        estimate_runtimes.append(total_runtime)

                        # Close dataset
                        dataset.close()

                    except:

                        # handle the exception if the file cannot be read
                        total_runtime = np.nan
                        estimate_runtimes.append(total_runtime)
                        print(" => cannot read this netcdf: " + sfincs_map)


            # NaN out all points that there 
            id_delete                   = zsmax_matrix == -9.99990000e+04 
            zsmax_matrix[id_delete]     = np.nan
            if include_qmax == 1:
                qmax_matrix[id_delete]      = np.nan
            if include_tmax == 1:
                tmax_matrix[id_delete]      = np.nan

            # TMP: find the largest zsmax => instability
            #a=1
            #max_values = np.nanmax(zsmax_matrix, axis=0)
            #plt.pcolor(np.squeeze(max_values))
            #plt.pcolor(np.squeeze(zsmax_matrix[59,:,:]))
            #plt.plot(np.squeeze(zsmax_matrix[:,732,499]))
            #files[3]

            # TMP: determine maximum depth for different years
            hhmax_matrix = zsmax_matrix - zb
            hhmax        = np.nanmax(hhmax_matrix, axis=1)
            hhmax        = np.nanmax(hhmax, axis=1)
            non_nan_mask = np.isnan(hhmax)
            indices      = np.where(non_nan_mask)[0]

            # Specify return period vector
            Ns, nx,ny       = np.shape(zsmax_matrix)
            lambda_value    = 1.0           # since we simulate individual years
            r_axis          = np.zeros(Ns)
            for ii in range(1, Ns+1):
                r_axis[ii-1] = (Ns+1) / ((Ns+1-ii) * lambda_value)

            # Prep output
            zsmax_out       = np.full((len(return_period_wanted), np.size(zsmax_matrix, 1), np.size(zsmax_matrix, 2)), np.nan)
            if include_qmax == 1:
                qmax_out        = np.full((len(return_period_wanted), np.size(zsmax_matrix, 1), np.size(zsmax_matrix, 2)), np.nan)
            if include_tmax == 1:
                tmax_out        = np.full((len(return_period_wanted), np.size(zsmax_matrix, 1), np.size(zsmax_matrix, 2)), np.nan)

            # loop over the second and third dimensions
            for i in range(zsmax_matrix.shape[1]):
                for j in range(zsmax_matrix.shape[2]):

                    # access the value at (i, j)
                    zsmax           = zsmax_matrix[:, i, j]
                    if include_qmax == 1:
                        qmax            = qmax_matrix[:, i, j]
                    if include_tmax == 1:
                        tmax            = tmax_matrix[:, i, j]

                    # Count number of zeros
                    num_non_nans    = np.count_nonzero(~np.isnan(zsmax))

                    # If we have more than 1 valid numbers we do this
                    if num_non_nans > 0:

                        # Is this a coastal point?
                        if zb[i,j] < 0:
                            a=1

                        # Replace NaNs with 0 and zb
                        zsmax[np.isnan(zsmax)] = zb[i,j]
                        if include_qmax == 1:
                            qmax[np.isnan(qmax)]   = 0.0
                        if include_tmax == 1:
                            tmax[np.isnan(tmax)]   = 0.0

                        # Sort zsmax and find RPs we want
                        ascending_indices   = np.argsort(zsmax)
                        zsmax_wanted        = zsmax[ascending_indices]
                        if include_qmax == 1:
                            qmax_wanted         = qmax[ascending_indices]
                        if include_tmax == 1:
                            tmax_wanted         = tmax[ascending_indices]

                        # Save output for the return periods we want
                        nearest_indices     = np.searchsorted(r_axis, return_period_wanted)
                        nearest_indices     = nearest_indices-1
                        nearest_indices[0]  = 0
                        zsmax_out[:,i,j]    = zsmax_wanted[nearest_indices]
                        if include_qmax == 1:
                            qmax_out[:,i,j]     = qmax_wanted[nearest_indices]
                        if include_tmax == 1:
                            tmax_out[:,i,j]     = tmax_wanted[nearest_indices]

                        # And no also interpolate with log
                        log_r_axis                  = np.log10(r_axis)                  # Take the logarithm of x-axis values
                        log_r_axis[0]               = 0.0                               # Trick to get to yearly
                        log_return_period_wanted    = np.log10(return_period_wanted)    # Take the logarithm of the desired return period

                        # Interpolate using log10 values
                        interpolated_value          = np.interp(log_return_period_wanted, log_r_axis, zsmax_wanted)
                        zsmax_out[:,i,j]            = interpolated_value

                        # Compute a hmax critera => nice to look at intermediate results
                        hh_out                      = np.squeeze(zsmax_out[:,i,j]) - zb[i,j]
                        idfind                      = np.where(hh_out< hh_criteria)
                        zsmax_out[idfind,i,j]       = np.nan
                        if include_qmax == 1:
                            qmax_out[idfind,i,j]        = np.nan
                        if include_tmax == 1:
                            tmax_out[idfind,i,j]        = np.nan

            # Done with this iteration (county and SLR): let's plot and save
            destout_TMP          = os.path.join(destout, domain, TMP_string)
            if not os.path.exists(destout_TMP):
                os.makedirs(destout_TMP)

            # Provide some feedback to on runtime
            estimate_runtimes_in_hours = [runtime / 3600 for runtime in estimate_runtimes]
            plt.plot(estimate_runtimes_in_hours)
            plt.xlabel('simulations')
            plt.ylabel('run time [hr]')
            fname = 'estimate_runtimes'
            fname = os.path.join(destout_TMP, fname)
            plt.savefig(fname, dpi='figure', format=None)
            plt.close()

            # Store run times in general for all SLRs
            matrix_runtimes[:,index] = estimate_runtimes_in_hours

            # Loop over return periods and make plots
            for t in range(zsmax_out.shape[0]):

                # Make figure
                fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

                # Water level
                p1 = axs[0, 0].pcolor(x/1000, y/1000, np.squeeze(zsmax_out[t,:,:]) , cmap='viridis')
                axs[0, 0].set_title("Water level")
                plt.colorbar(p1, ax=axs[0, 0])

                # Flow velocity
                if include_qmax == 1:
                    p2 = axs[0, 1].pcolor(x/1000, y/1000, np.squeeze(qmax_out[t,:,:]) , cmap='Reds')
                    axs[0, 1].set_title("Velocity")
                    plt.colorbar(p2, ax=axs[0, 1])

                    # Duration
                    if include_tmax == 1:
                        p3 = axs[1, 0].pcolor(x/1000, y/1000, np.squeeze(tmax_out[t,:,:]) , cmap='GnBu')
                        axs[1, 0].set_title("Duration")
                        plt.colorbar(p3, ax=axs[1, 0])

                    # Bed level
                    p4 = axs[1, 1].pcolor(x/1000, y/1000, zb , cmap='terrain', vmin=-20, vmax=20)
                    axs[1, 1].set_title("Bed level")
                    plt.colorbar(p4, ax=axs[1, 1])

                    # Set limits
                    p1.set_clim(vmin=0, vmax=10)  
                    if include_qmax == 1:p2.set_clim(vmin=0, vmax=5)  
                    if include_tmax == 1: p3.set_clim(vmin=0, vmax=86400)  
                    p4.set_clim(vmin=-20, vmax=20)  

                    # Print this
                    fname = 'overview_' + str(return_period_wanted[t]) + 'yr.png'
                    fname = os.path.join(destout_TMP, fname)
                    plt.savefig(fname, dpi='figure', format=None)
                    plt.close()

                    # Make one large figure for zsmax only
                    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
                    p1 = axs.pcolor(x/1000, y/1000, np.squeeze(zsmax_out[t,:,:]) , cmap='viridis')
                    p1.set_clim(vmin=0, vmax=10)  # Set the color limits from 0 to 10
                    axs.set_title("Water level")
                    plt.colorbar(p1, ax=axs)
                    fname = 'zsmax_only_' + str(return_period_wanted[t]) + 'yr.png'
                    fname = os.path.join(destout_TMP, fname)
                    plt.savefig(fname, dpi='figure', format=None)
                    plt.close()
                

                # Create an xarray Dataset per return period
                for t in range(zsmax_out.shape[0]):

                    # Get coordinates ready
                    x_coord         = x[1, :]
                    y_coord         = y[:, 1]
                    zsmax_out_now   = np.squeeze(zsmax_out[t, :, :])    

                    # Make base
                    ds              = xr.Dataset()
                    coords          = ('y', 'x')
                    ds['zsmax']     = (coords, np.float32(zsmax_out_now))
                    ds.coords['x']  = x_coord.compressed()
                    ds.coords['y']  = y_coord.compressed()

                    # Get more description
                    ds['zsmax'].attrs['units']          = 'm'
                    ds['zsmax'].attrs['standard_name']  = 'maximum of sea_surface_height_above_mean_sea_level'
                    ds['zsmax'].attrs['long_name']      = 'maximum_water_level'
                    ds['zsmax'].attrs['coordinates']    = 'y x'

                    # Optionally add qmax and tmax if include_qmax and include_tmax are set to 1
                    if include_qmax == 1: 
                        zsmax_out_now   = np.squeeze(qmax_out[t, :, :])    
                        ds['qmax']      = (coords, np.float32(zsmax_out_now))
                    if include_tmax == 1: 
                        zsmax_out_now   = np.squeeze(tmax_out[t, :, :])    
                        ds['tmax']     = (coords, np.float32(zsmax_out_now))

                    # Also add bed level
                    ds['zb']     = (coords, np.float32(zb))

                    # Add global attributes
                    ds.attrs["description"] = "NetCDF file with process SFINCS outputs"
                    ds.attrs["history"] = "Created " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Save the Dataset to a NetCDF file
                    filename = 'processed_SFINCS_output_RP' +  "{:03}".format(int(return_period_wanted[t]))  + '.nc'
                    filename = os.path.join(destout_TMP, filename)
                    ds.to_netcdf(filename)

                    # Additionally write simple wet-dry interface
                    zsmax_out_now   = np.squeeze(zsmax_out[t, :, :])    
                    wet_dry         = np.where(np.isnan(zsmax_out_now), 0, 1)
                    ds              = xr.Dataset()
                    coords          = ('y', 'x')
                    ds.coords['x']  = x_coord.compressed()
                    ds.coords['y']  = y_coord.compressed()
                    ds['wetdry']    = (coords, np.float32(wet_dry))
                    filename = 'wetdry_SFINCS_output_RP' +  "{:03}".format(int(return_period_wanted[t]))  + '.nc'
                    filename = os.path.join(destout_TMP, filename)
                    ds.to_netcdf(filename)

            # handle the exception if the file cannot be read
            print("done with this iteration - " + destout_TMP)
    
    # Do analysis on runtimes (how much done and stuff)
    a=1

    # Done with this particular counties


# Done with the script
print('done!')