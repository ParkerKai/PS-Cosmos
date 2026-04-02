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
from hydromt_sfincs import SfincsModel

# Directory
destin = r"C:\Users\kai\Documents\KaiRuns\20250422_tideonly_MHHW"
destout = r"C:\Users\kai\Documents\KaiRuns\PostProcess"

# SLRs                    = ['000', '025', '050', '100', '150', '200', '300']
SLRs = ["100"]
counties = ["02_Pierce"]
# sub_categories          = ['_median','_low','_high']

sub_categories = ["_low"]

# Some processing of the inputs
sub_categories = [sub.replace("_median", "") for sub in sub_categories]


# User settings
# return_period_wanted    = [1, 2, 5, 10, 20, 50, 100]     # number of years requested
hh_criteria = 0.010001  # just above the treshold from SFINCS

include_qmax = 1
include_tmax = 1
include_tmax_zs = 1

read_binary = 1
# Go to folder and loop over domains
for county in counties:
    # Start with this domain first
    print(f"Started with {county}", flush=True)
    destin_TMP = os.path.join(destin, county)

    # Go over SLRs
    for index, slr in enumerate(SLRs):
        # Go over sub_categories (standard, low and high)
        for index_cat, category in enumerate(sub_categories):
            # Make print statement
            print(f" => SLR value: {slr} - {sub_categories[index_cat]}", flush=True)
            string_wanted = "SY"
            destin_TMP = os.path.join(destin, county)
            TMP_string = slr + category
            destin_TMP = os.path.join(destin_TMP, TMP_string)
            files = os.listdir(destin_TMP)

            if len(files) > 1:
                print("Too many WYs (supposed to only have one)!")
                aasdf

            # Read netcdf in WY folders
            total_years = 1
            file = files[0]

            # Read as binary file?
            if read_binary == 1:
                # Files for reading
                sfincs_root = os.path.join(destin_TMP, file)
                sfincs_map = os.path.join(destin_TMP, file, "zsmax.dat")
                sfincs_qmax = os.path.join(destin_TMP, file, "qmax.dat")
                sfincs_tmax = os.path.join(destin_TMP, file, "tmax.dat")
                sfincs_tmax_zs = os.path.join(destin_TMP, file, "tmax_zs.dat")

                # Open the Model configuration files
                print(file, flush=True)
                mod = SfincsModel(sfincs_root, mode="r")

                mod.read_subgrid()
                x = mod.grid.x.values
                y = mod.grid.y.values
                x, y = np.meshgrid(x, y)
                zb = mod.grid.dep.values
                zb = mod.subgrid.z_zmin
                zb = zb.values

                # Read index
                sfincs_index = os.path.join(sfincs_root, mod.config["indexfile"])
                with open(sfincs_index, "rb") as fid:
                    ind = np.fromfile(fid, dtype="u4")
                    ind = ind[1:] - 1  # convert to zero based index

                # Initialize
                zsmax = np.full((mod.config["mmax"], mod.config["nmax"]), np.nan, "f4")
                if include_qmax == 1:
                    qmax = np.full(
                        (mod.config["mmax"], mod.config["nmax"]), np.nan, "f4"
                    )
                if include_tmax == 1:
                    tmax = np.full(
                        (mod.config["mmax"], mod.config["nmax"]), np.nan, "f4"
                    )
                if include_tmax_zs == 1:
                    tmax_zs = np.full(
                        (mod.config["mmax"], mod.config["nmax"]), np.nan, "f4"
                    )
                nx, ny = np.shape(zsmax)

                # Read Binary
                try:
                    # Read zsmax
                    with open(sfincs_map, "rb") as fid:
                        zsmax_file = np.fromfile(sfincs_map, dtype="f4")
                        zsmax_file = zsmax_file[1:-1]
                        zsmax.flat[ind] = zsmax_file
                        zsmax[zsmax == -9999] = np.nan

                    # Read qmax
                    if include_qmax == 1:
                        with open(sfincs_qmax, "rb") as fid:
                            qmax_file = np.fromfile(sfincs_qmax, dtype="f4")
                            qmax_file = qmax_file[1:-1]
                            qmax.flat[ind] = qmax_file
                            qmax[qmax == -9999] = np.nan

                    # Read tmax
                    if include_tmax == 1:
                        with open(sfincs_tmax, "rb") as fid:
                            tmax_file = np.fromfile(sfincs_tmax, dtype="f4")
                            tmax_file = tmax_file[1:-1]
                            tmax.flat[ind] = tmax_file
                            tmax[tmax == -9999] = np.nan

                    # Read tmax_zs
                    if include_tmax_zs == 1:
                        with open(sfincs_tmax_zs, "rb") as fid:
                            tmax_zs_file = np.fromfile(sfincs_tmax_zs, dtype="f4")
                            tmax_zs_file = tmax_zs_file[1:-1]
                            tmax_zs.flat[ind] = tmax_zs_file
                            tmax[tmax == -9999] = np.nan

                    # Read runtime
                    value_iteration = int(file.replace("SY", ""))
                    name_model = county.split("_")[1].strip("'")
                    destin_log = os.path.join(destin, county)
                    file_name = (
                        "SFINCS_"
                        + name_model
                        + "_"
                        + TMP_string
                        + str(value_iteration)
                        + ".log"
                    )
                    file_name = os.path.join(destin_log, file_name)
                    total_runtime = np.nan

                    # Try to read the file
                    try:
                        # Read the file
                        with open(file_name, "r") as file:
                            lines = file.readlines()
                            for line in lines:
                                if "Total time" in line:
                                    total_runtime = float(line.split(":")[1].strip())
                                    break
                    except Exception as e:
                        print(f" => cannot read this log file: {file_name}, error: {e}")

                except:
                    print(" => cannot read this binary file: " + sfincs_map)

                # Transpose matrices so they are in the right shape for saving
                zsmax = zsmax.transpose()
                if include_qmax == 1:
                    qmax = qmax.transpose()
                if include_tmax == 1:
                    tmax = tmax.transpose()
                if include_tmax_zs == 1:
                    tmax_zs = tmax_zs.transpose()

            # If not, let's read as netcdf
            else:
                # List SFINCS map
                sfincs_map = os.path.join(destin_TMP, file, "sfincs_map.nc")

                # Read netcdf
                try:
                    # Open the netcdf file
                    # print(file)
                    dataset = nc.Dataset(sfincs_map, "r")

                    # read the variables of interest and process the
                    zsmax = dataset.variables["zsmax"][:]
                    zsmax = np.squeeze(zsmax, 0)
                    if include_qmax == 1:
                        qmax = dataset.variables["qmax"][:]
                        qmax = np.squeeze(qmax, 0)
                    if include_tmax == 1:
                        tmax = dataset.variables["tmax"][:]
                        tmax = np.squeeze(tmax, 0)
                    if include_tmax_zs == 1:
                        tmax_zs = dataset.variables["tmax_zs"][:]
                        tmax_zs = np.squeeze(tmax_zs, 0)

                    # Also read the grid and bed levels
                    x = dataset.variables["x"][:]
                    y = dataset.variables["y"][:]
                    zb = dataset.variables["zb"][:]

                    # Close dataset
                    dataset.close()

                except:
                    print(" => cannot read this netcdf: " + sfincs_map)

            # Reading done!!!!!!

            # NaN out all points that there
            id_delete = zsmax < -100
            zsmax[id_delete] = np.nan
            if include_qmax == 1:
                qmax[id_delete] = np.nan
            if include_tmax == 1:
                tmax[id_delete] = np.nan
            if include_tmax_zs == 1:
                tmax_zs[id_delete] = np.nan

            # TMP: determine maximum depth for different years
            hhmax = zsmax - zb
            hhmax = np.nanmax(hhmax, axis=1)
            non_nan_mask = np.isnan(hhmax)
            indices = np.where(non_nan_mask)[0]

            # Done with this iteration (county and SLR): let's plot and save
            destout_TMP = os.path.join(destout, county, TMP_string)
            if not os.path.exists(destout_TMP):
                os.makedirs(destout_TMP)

            # Make plots

            # Make figure
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

            # Water level
            p1 = axs[0, 0].pcolor(x / 1000, y / 1000, np.squeeze(zsmax), cmap="viridis")
            axs[0, 0].set_title("Water level")
            plt.colorbar(p1, ax=axs[0, 0])

            # Flow velocity
            if include_qmax == 1:
                p2 = axs[0, 1].pcolor(x / 1000, y / 1000, np.squeeze(qmax), cmap="Reds")
                axs[0, 1].set_title("Velocity")
                plt.colorbar(p2, ax=axs[0, 1])

                # Duration
            if include_tmax == 1:
                p3 = axs[1, 0].pcolor(x / 1000, y / 1000, np.squeeze(tmax), cmap="GnBu")
                axs[1, 0].set_title("Duration")
                plt.colorbar(p3, ax=axs[1, 0])

            # Bed level
            p4 = axs[1, 1].pcolor(
                x / 1000, y / 1000, zb, cmap="terrain", vmin=-20, vmax=20
            )
            axs[1, 1].set_title("Bed level")
            plt.colorbar(p4, ax=axs[1, 1])

            # Set limits
            p1.set_clim(vmin=0, vmax=10)
            if include_qmax == 1:
                p2.set_clim(vmin=0, vmax=5)
            if include_tmax == 1:
                p3.set_clim(vmin=0, vmax=86400)
            p4.set_clim(vmin=-20, vmax=20)

            # Print this
            fname = "overview_daily.png"
            fname = os.path.join(destout_TMP, fname)
            plt.savefig(fname, dpi="figure", format=None)
            plt.close()

            # Make one large figure for zsmax only
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            p1 = axs.pcolor(x / 1000, y / 1000, np.squeeze(zsmax), cmap="viridis")
            p1.set_clim(vmin=0, vmax=10)  # Set the color limits from 0 to 10
            axs.set_title("Water level")
            plt.colorbar(p1, ax=axs)
            fname = "zsmax_only_daily.png"
            fname = os.path.join(destout_TMP, fname)
            plt.savefig(fname, dpi="figure", format=None)
            plt.close()

            # Create an xarray Dataset per return period

            # Get coordinates ready
            x_coord = x[1, :]
            y_coord = y[:, 1]

            # Make base
            ds = xr.Dataset()
            coords = ("y", "x")
            ds["zsmax"] = (coords, np.float32(zsmax))
            if read_binary == 1:
                ds.coords["x"] = x_coord.squeeze()
                ds.coords["y"] = y_coord.squeeze()
            else:
                ds.coords["x"] = x_coord.compressed()
                ds.coords["y"] = y_coord.compressed()

            # Get more description
            ds["zsmax"].attrs["units"] = "m"
            ds["zsmax"].attrs["standard_name"] = (
                "maximum of sea_surface_height_above_mean_sea_level"
            )
            ds["zsmax"].attrs["long_name"] = "maximum_water_level"
            ds["zsmax"].attrs["coordinates"] = "y x"

            # Optionally add qmax and tmax if include_qmax and include_tmax are set to 1
            if include_qmax == 1:
                ds["qmax"] = (coords, np.float32(qmax))
            if include_tmax == 1:
                ds["tmax"] = (coords, np.float32(tmax))
            if include_tmax_zs == 1:
                ds["tmax_zs"] = (coords, np.float32(tmax_zs))

            # Also add bed level
            ds["zb"] = (coords, np.float32(zb))

            # Add global attributes
            ds.attrs["description"] = "NetCDF file with process SFINCS outputs"
            ds.attrs["history"] = "Created " + datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # Save the Dataset to a NetCDF file
            filename = "processed_SFINCS_output_RPdaily.nc"
            filename = os.path.join(destout_TMP, filename)
            ds.to_netcdf(filename)

            # Additionally write simple wet-dry interface
            wet_dry = np.where(np.isnan(zsmax), 0, 1)
            ds = xr.Dataset()
            coords = ("y", "x")
            if read_binary == 1:
                ds.coords["x"] = x_coord.squeeze()
                ds.coords["y"] = y_coord.squeeze()
            else:
                ds.coords["x"] = x_coord.compressed()
                ds.coords["y"] = y_coord.compressed()
            ds["wetdry"] = (coords, np.float32(wet_dry))
            filename = "wetdry_SFINCS_output_RPdaily.nc"
            filename = os.path.join(destout_TMP, filename)
            ds.to_netcdf(filename)

            # handle the exception if the file cannot be read
            print(f"done with this iteration - {destout_TMP}", flush=True)


# Done with the script
print("done!")
