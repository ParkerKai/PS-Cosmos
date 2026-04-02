# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:54:42 2024

Functions to interface with matlab

@author: kaparker
    USGS: PCMSC
   kaparker@usgs.gov
"""

__author__ = "Kai Parker"
__email__ = "kaparker@usgs.gov"


import xarray as xr
import numpy as np

# ===============================================================================
# %% Define some functions
# ===============================================================================


def load_SfincsQuadtree(file_in):

    # Load quadtree grid data
    qtr = xr.open_dataset(file_in)
    bx = qtr.mesh2d_node_x.values
    by = qtr.mesh2d_node_y.values
    i = qtr.mesh2d_face_nodes.values.astype(int)
    lev = qtr.level.values

    # Get the grid spacing
    qdx = qtr.dx / (2 ** (lev - 1))
    qdy = qtr.dy / (2 ** (lev - 1))

    # Get the corners
    x0, y0 = [], []
    for j in range(len(i)):
        x0.append(bx[i[j][0]])
        y0.append(by[i[j][0]])
    x0 = np.array(x0)
    y0 = np.array(y0)

    # Get the centers
    xc, yc = x0 + qdx / 2, y0 + qdy / 2

    return xc, yc
