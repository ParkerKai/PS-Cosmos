# Modules needed
import os
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
from hydromt_sfincs import SfincsModel, utils
import matplotlib


sfincs_root = r"D:\TempSFINCS\100_low\SY1940"  # Location of the PS-Cosmos codebase

# Open the Model configuration files
print(sfincs_root)
mod = SfincsModel(sfincs_root, mode="r")

mod.read_subgrid()
x = mod.grid.x.values
y = mod.grid.y.values
x, y = np.meshgrid(x, y)
zb = mod.grid.dep.values
zb = mod.subgrid.z_zmin
zb = zb.values

forcing = mod.forcing

results = mod.read_results()

ind = utils.read_binary_map_index(os.path.join("D:\TempSFINCS", "sfincs.ind"))

temp = utils.read_binary_map(os.path.join(sfincs_root, "zs.dat"), ind, x.shape)


temp[temp == -9999] = np.nan
temp[temp > 100] = np.nan
temp[temp < -100] = np.nan


fig = plt.subplots(1, 1)
fig = fig[0]
fig.set_size_inches(10, 6)
ax = fig.get_axes()
ax = ax[0]

pcm = ax.pcolormesh(x, y, temp)
fig.colorbar(pcm, ax=ax, label="Z values")


fig = plt.subplots(1, 1)
fig = fig[0]
fig.set_size_inches(10, 6)
ax = fig.get_axes()
ax = ax[0]

ax.plot(forcing["bzs"]["time"], forcing["bzs"].isel(stations=100))
