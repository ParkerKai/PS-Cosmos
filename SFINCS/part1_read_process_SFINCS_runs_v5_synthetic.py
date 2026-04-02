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
from hydromt_sfincs import SfincsModel, utils
import argparse
import traceback 

# Parse inputs
parser = argparse.ArgumentParser(description='Run_Criterion')
parser.add_argument("Dir", nargs=1,
        default=r'/caldera/hovenweep/projects/usgs/hazards/pcmsc/cosmos/PugetSound/sfincs/20241018_synthetic_future_withchange_mean_100yr')
parser.add_argument("RP", nargs=1, default="1, 2, 5, 10, 20, 50, 100")
parser.add_argument("SLR", nargs=1, default="000, 025, 050, 100, 150, 200, 300")
parser.add_argument("Dom", nargs=1, default="01_King, 02_Pierce")
parser.add_argument("SubCat",nargs=1, default="_median,_low, _high")
args=parser.parse_args()


# Input arguments
print(f'Directory: {args.Dir}')
print(f'Return Periods: {args.RP[0]}')
print(f'Sea Level Rise: {args.SLR[0]}')
print(f'Domains: {args.Dom[0]}')
print(f'Sub Categories: {args.SubCat[0]}')
print(' ')

exec(f'destin               = r"{args.Dir[0]}"')
exec(f"return_period_wanted = {args.RP[0].split(',')}")
exec(f"SLRs_formatted       = {args.SLR[0].split(',')}")
exec(f"domains              = {args.Dom[0].split(',')}")
exec(f"sub_categories       = {args.SubCat[0].split(',')}")

#Some processing of the inputs
return_period_wanted = [int(RP) for RP in return_period_wanted]
sub_categories = [sub.replace('_median', '') for sub in sub_categories]


# User settings
#return_period_wanted    = [1, 2, 5, 10, 20, 50, 100]     # number of years requested
hh_criteria             = 0.010001                       # just above the treshold from SFINCS

# Directory
destout                 = os.path.join(destin,'PostProcess')
include_qmax            = 1
include_tmax            = 1
include_tmax_zs         = 1
read_binary             = 1
# Go to folder and loop over domains
for domain in domains:

    # Start with this domain first
    print(f'Started with {domain}',flush=True)
    destin_TMP          = os.path.join(destin, domain)

    # Go over SLRs
    for index, slr in enumerate(SLRs_formatted):

        # Go over sub_categories (standard, low and high)
        for index_cat, category in enumerate(sub_categories):
            
            # Make print statement
            print(f" => SLR value: {slr} - {sub_categories[index_cat]}",flush=True)
            string_wanted       = 'SY'
            destin_TMP          = os.path.join(destin, domain)
            TMP_string          = slr + category
            destin_TMP          = os.path.join(destin_TMP, TMP_string)
            files               = os.listdir(destin_TMP)

            # Read netcdf in WY folders
            count_years         = 0
            total_years         = len([file for file in os.listdir(destin_TMP) if file.startswith(string_wanted)])
            estimate_runtimes   = []

            # Define matrix for runtimes
            if index == 0: matrix_runtimes = np.full((total_years, len(SLRs_formatted)), np.nan)
            
            for file in files:
                if file.startswith(string_wanted):
                    
                    # Read as binary file?
                    if read_binary == 1:

                        # Files for reading
                        sfincs_root     = os.path.join(destin_TMP, file)
                        sfincs_map      = os.path.join(destin_TMP, file, 'zsmax.dat')
                        sfincs_qmax     = os.path.join(destin_TMP, file, 'qmax.dat')
                        sfincs_tmax     = os.path.join(destin_TMP, file, 'tmax.dat')
                        sfincs_tmax_zs  = os.path.join(destin_TMP, file, 'tmax_zs.dat')

                        # Open the Model configuration files 
                        print(file,flush=True)
                        mod                     = SfincsModel(sfincs_root, mode="r")
                        
                        mod.read_subgrid()
                        x                   = mod.grid.x.values
                        y                   = mod.grid.y.values
                        x, y                = np.meshgrid(x, y)
                        zb                  = mod.grid.dep.values
                        zb                  = mod.subgrid.z_zmin
                        zb                  = zb.values

                        # Read index
                        sfincs_index    = os.path.join(sfincs_root, mod.config['indexfile'])
                        with open(sfincs_index, 'rb') as fid:
                            ind = np.fromfile(fid, dtype="u4")
                            ind = ind[1:] - 1  # convert to zero based index

                        # Initialize
                        zsmax       = np.full( (mod.config['mmax'], mod.config['nmax']), np.nan, "f4")
                        if include_qmax == 1:
                            qmax    = np.full( (mod.config['mmax'], mod.config['nmax']), np.nan, "f4")
                        if include_tmax == 1:
                            tmax    = np.full( (mod.config['mmax'], mod.config['nmax']), np.nan, "f4")
                        if include_tmax_zs == 1:
                            tmax_zs = np.full( (mod.config['mmax'], mod.config['nmax']), np.nan, "f4")
                        nx,ny  = np.shape(zsmax)
                        
                        # Read Binary
                        try:

                            # Read zsmax
                            with open(sfincs_map, 'rb') as fid:
                                zsmax_file      = np.fromfile(sfincs_map, dtype="f4")
                                zsmax_file      = zsmax_file[1:-1]
                                zsmax.flat[ind] = zsmax_file
                                zsmax[zsmax==-9999] = np.nan

                            # Read qmax
                            if include_qmax == 1:
                                with open(sfincs_qmax, 'rb') as fid:
                                    qmax_file      = np.fromfile(sfincs_qmax, dtype="f4")
                                    qmax_file      = qmax_file[1:-1]
                                    qmax.flat[ind] = qmax_file
                                    qmax[qmax==-9999] = np.nan

                            # Read tmax
                            if include_tmax == 1:
                                with open(sfincs_tmax, 'rb') as fid:
                                    tmax_file      = np.fromfile(sfincs_tmax, dtype="f4")
                                    tmax_file      = tmax_file[1:-1]
                                    tmax.flat[ind] = tmax_file
                                    tmax[tmax==-9999] = np.nan

                            # Read tmax_zs
                            if include_tmax_zs == 1:
                                with open(sfincs_tmax_zs, 'rb') as fid:
                                    tmax_zs_file      = np.fromfile(sfincs_tmax_zs, dtype="f4")
                                    tmax_zs_file      = tmax_zs_file[1:-1]
                                    tmax_zs.flat[ind] = tmax_zs_file
                                    tmax[tmax==-9999] = np.nan

                            # Read runtime
                            value_iteration = int(file.replace("SY", ""))
                            name_model      = domain.split('_')[1].strip("'")
                            destin_log      = os.path.join(destin, domain)
                            file_name       = 'SFINCS_' + name_model + '_' + TMP_string + str(value_iteration) + '.log'
                            file_name       = os.path.join(destin_log, file_name)
                            total_runtime   = np.nan

                            # Try to read the file
                            try:
                                # Read the file
                                with open(file_name, 'r') as file:
                                    lines = file.readlines()
                                    for line in lines:
                                        if "Total time" in line:
                                            total_runtime = float(line.split(':')[1].strip())
                                            break
                            except Exception as e:
                                print(f" => cannot read this log file: {file_name}, error: {e}")

                            # Store value
                            estimate_runtimes.append(total_runtime)

                        except:

                            # handle the exception if the file cannot be read
                            total_runtime = np.nan
                            estimate_runtimes.append(total_runtime)

                            print(" => cannot read this binary file: " + sfincs_map)

                        # Transpose matrices so they are in the right shape for saving
                        zsmax       = zsmax.transpose()
                        if include_qmax == 1:
                            qmax    = qmax.transpose()
                        if include_tmax == 1:
                            tmax    = tmax.transpose()
                        if include_tmax_zs == 1:
                            tmax_zs    = tmax_zs.transpose()


                    # If not, let's read as netcdf
                    else:

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
                            if include_tmax_zs == 1:
                                tmax_zs    = dataset.variables["tmax_zs"][:]
                                tmax_zs    = np.squeeze(tmax_zs,0)

                            # Also read the grid and bed levels
                            x       = dataset.variables["x"][:]
                            y       = dataset.variables["y"][:]
                            zb      = dataset.variables["zb"][:]

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
                    
                    # Reading done
                    # Make empty matrix
                    if count_years == 0:
                        zsmax_matrix        = np.full((total_years, np.size(zsmax, 0), np.size(zsmax, 1)), np.nan)
                        if include_qmax == 1:
                            qmax_matrix     = np.full((total_years, np.size(zsmax, 0), np.size(zsmax, 1)), np.nan)
                        if include_tmax == 1:
                            tmax_matrix     = np.full((total_years, np.size(zsmax, 0), np.size(zsmax, 1)), np.nan)
                        if include_tmax_zs == 1:
                            tmax_zs_matrix  = np.full((total_years, np.size(zsmax, 0), np.size(zsmax, 1)), np.nan)

                    Ns, nx,ny       = np.shape(zsmax_matrix)
                    
                    # Place files in the dictionary 
                    zsmax_matrix[count_years,:,:]       = zsmax
                    if include_qmax == 1:
                        qmax_matrix[count_years,:,:]    = qmax
                    if include_tmax == 1:
                        tmax_matrix[count_years,:,:]    = tmax
                    if include_tmax_zs == 1:
                        tmax_zs_matrix[count_years,:,:] = tmax_zs
                    count_years = count_years+1


            # NaN out all points that there 
            id_delete                       = zsmax_matrix < -100 
            zsmax_matrix[id_delete]         = np.nan
            if include_qmax == 1:
                qmax_matrix[id_delete]      = np.nan
            if include_tmax == 1:
                tmax_matrix[id_delete]      = np.nan
            if include_tmax_zs == 1:
                tmax_zs_matrix[id_delete]   = np.nan


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
            if include_tmax_zs == 1:
                tmax_zs_out        = np.full((len(return_period_wanted), np.size(zsmax_matrix, 1), np.size(zsmax_matrix, 2)), np.nan)
            years_out = np.full((len(return_period_wanted), np.size(zsmax_matrix, 1), np.size(zsmax_matrix, 2)),-999, dtype='int64')

            # loop over the second and third dimensions
            for i in range(zsmax_matrix.shape[1]):
                for j in range(zsmax_matrix.shape[2]):

                    # access the value at (i, j)
                    zsmax           = zsmax_matrix[:, i, j]
                    if include_qmax == 1:
                        qmax            = qmax_matrix[:, i, j]
                    if include_tmax == 1:
                        tmax            = tmax_matrix[:, i, j]
                    if include_tmax_zs == 1:
                        tmax_zs            = tmax_zs_matrix[:, i, j]

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
                        if include_tmax_zs == 1:
                            tmax_zs[np.isnan(tmax_zs)]   = 0.0

                        # Sort zsmax and find RPs we want
                        ascending_indices   = np.argsort(zsmax)
                        zsmax_wanted        = zsmax[ascending_indices]
                        if include_qmax == 1:
                            qmax_wanted         = qmax[ascending_indices]
                        if include_tmax == 1:

                            # Option 1 => sorting the indices
                            tmax_wanted         = tmax[ascending_indices]

                            # Option 2 => cumulative sum
                            if np.min(tmax) > 0:
                                ascending_indices   = np.argsort(tmax)
                                tmax_wanted         = tmax[ascending_indices]
                            
                        if include_tmax_zs == 1:
                            tmax_zs_wanted         = tmax_zs[ascending_indices]
                        
                        # Find the indices for the return periods we want
                        nearest_indices_old    = np.searchsorted(r_axis, return_period_wanted)
                        nearest_indices         = []
                        for rp in return_period_wanted:
                            idx = np.searchsorted(r_axis, rp, side='left')
                            if idx == 0:
                                nearest_idx = np.array(0,dtype='int64')
                            elif idx == len(r_axis):
                                nearest_idx = len(r_axis) - 1
                            else:
                                prev_val = r_axis[idx - 1]
                                next_val = r_axis[idx]
                                if abs(rp - prev_val) <= abs(rp - next_val):
                                    nearest_idx = idx - 1
                                else:
                                    nearest_idx = idx
                            nearest_indices.append(nearest_idx)
                        r_axis_found1 = r_axis[nearest_indices_old]
                        r_axis_found2 = r_axis[nearest_indices]
                        
                        # Save Output
                        zsmax_out[:,i,j]    = zsmax_wanted[nearest_indices]
                        if include_qmax == 1:
                            qmax_out[:,i,j]     = qmax_wanted[nearest_indices]
                        if include_tmax == 1:

                            # Option 1 => find the nearest value
                            tmax_out[:,i,j]     = tmax_wanted[nearest_indices]

                            # Option 2 => average duration in hours for the return period
                            # This means = the maximum that occurs on average once every 'rp' years
                            num_samples = 1000  # Number of Monte Carlo samples
                            for idx, rp in enumerate(return_period_wanted):

                                # Check if this is not zero
                                if np.max(tmax_wanted) > 0:

                                    # Generate all random samples at once for better performance
                                    sampled_indices = np.random.choice(tmax_wanted.shape[0], (num_samples, rp))

                                    # Gather the sampled tmax values
                                    sampled_tmax    = tmax_wanted[sampled_indices]  # Shape: (num_samples, rp)

                                    # Compute the maximum for each sample (axis=1)
                                    sampled_max     = np.nanmax(sampled_tmax, axis=1)  # Shape: (num_samples,)

                                    # Compute the mean of the sampled maxima
                                    final_average   = np.nanmean(sampled_max)

                                    # Store the result in hours
                                    tmax_out[idx, i, j] = final_average / 3600  # Convert from seconds to hourseconds to hours

                                else:
                                    tmax_out[:, i, j]   = 0.0
                        if include_tmax_zs == 1:
                            tmax_zs_out[:,i,j]     = tmax_zs_wanted[nearest_indices]
                        
                        # Pull out the years corresopnding to the pulled maximums
                        sim_list = np.array([int(file.replace('SY','')) for file in files])
                        
                        # Sort like all the other output parameters
                        sim_list = sim_list[ascending_indices]

                        # Need to back out the correct indices before sorting.
                        years_out[:,i,j] = sim_list[nearest_indices]

                        # And now also interpolate with log
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
                        if include_tmax_zs == 1:
                            tmax_zs_out[idfind,i,j]     = np.nan

                        years_out[idfind,i,j]           = -999

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
                    if read_binary == 1:
                        ds.coords['x']  = x_coord.squeeze()
                        ds.coords['y']  = y_coord.squeeze()
                    else:
                        ds.coords['x']  = x_coord.compressed()
                        ds.coords['y']  = y_coord.compressed()

                    # Get more description
                    ds['zsmax'].attrs['units']          = 'm'
                    ds['zsmax'].attrs['standard_name']  = 'maximum of sea_surface_height_above_mean_sea_level'
                    ds['zsmax'].attrs['long_name']      = 'maximum_water_level'
                    ds['zsmax'].attrs['coordinates']    = 'y x'

                    # Optionally add qmax and tmax if include_qmax and include_tmax are set to 1
                    if include_qmax == 1: 
                        data_out_now   = np.squeeze(qmax_out[t, :, :])    
                        ds['qmax']      = (coords, np.float32(data_out_now))
                    if include_tmax == 1: 
                        data_out_now   = np.squeeze(tmax_out[t, :, :])    
                        ds['tmax']     = (coords, np.float32(data_out_now))
                    if include_tmax_zs == 1:
                        data_out_now   = np.squeeze(tmax_zs_out[t, :, :])
                        ds['tmax_zs']      = (coords, np.float32(data_out_now))
                    
                    data_out_now   = np.squeeze(years_out[t, :, :])
                    id_bad                      = np.isnan(zsmax_out_now)
                    data_out_now[id_bad]       = -999
                    
                    ds['year_max']     = (coords, data_out_now)
                    ds['year_max'].attrs['units']          = 'Year (integer)'
                    ds['year_max'].attrs['_FillValue']     = '-999'
                    ds['year_max'].attrs['standard_name']  = 'Simulation Year RP occurs (With file header removed)'
                    ds['year_max'].attrs['coordinates']    = 'y x'  

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
                    if read_binary == 1:
                        ds.coords['x']  = x_coord.squeeze()
                        ds.coords['y']  = y_coord.squeeze()
                    else:
                        ds.coords['x']  = x_coord.compressed()
                        ds.coords['y']  = y_coord.compressed()
                    ds['wetdry']    = (coords, np.float32(wet_dry))
                    filename = 'wetdry_SFINCS_output_RP' +  "{:03}".format(int(return_period_wanted[t]))  + '.nc'
                    filename = os.path.join(destout_TMP, filename)
                    ds.to_netcdf(filename)

            # handle the exception if the file cannot be read
            print(f"done with this iteration - {destout_TMP}",flush=True)
    
    # Do analysis on runtimes (how much done and stuff)
    a=1

    # Done with this particular counties


# Done with the script
print('done!')
