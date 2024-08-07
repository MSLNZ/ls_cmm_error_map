# -------------------------------------------------------------------------------
# Name:        ballplate_3D_plots.py
# Purpose:      3D plot of deformation caused by model parameters
#
# Author:      e.howick
#
# Created:     30/04/2013
# Copyright:   (c) e.howick 2013
# Licence:     <your licence>
# -------------------------------------------------------------------------------

import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import design_matrix_linear


def plot_model3d(ax, params, xt, yt, zt, mag, lm="b-", lw=1):
    """
    takes a set of model parameters and produces a 3D magniifed plot of machine
    deformation
    """
    XYZ, eXYZ = design_matrix_linear.machine_deformation(params, xt, yt, zt)
    pXYZ_3D = XYZ + mag * eXYZ

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    nx, ny, nz = XYZ.shape[:3]

    # lines parallel to x axis
    for j in range(ny):
        for k in range(nz):
            xp, yp, zp = pXYZ_3D[:, j, k, :].T
            print(f"{pXYZ_3D[:, j, k, :].T.shape}")
            _newline = ax.plot(xp, yp, zp, lm, linewidth=lw)

    # lines parallel to y axis
    for i in range(nx):
        for k in range(nz):
            xp, yp, zp = pXYZ_3D[i, :, k, :].T
            _newline = ax.plot(xp, yp, zp, lm, linewidth=lw)

    # lines parallel to z axis
    for i in range(nx):
        for j in range(ny):
            xp, yp, zp = pXYZ_3D[i, j, :, :].T
            _newline = ax.plot(xp, yp, zp, lm, linewidth=lw)

    ax.auto_scale_xyz([-100, 900], [-100, 700], [-100, 700])
    ax.pbaspect = [1.0, 0.8, 1.0]
    return XYZ, eXYZ


def main1():
    pfname = (
        r"L:\CMM Leitz\Ballplate_MSL\Python_analysis\ballplate_linear_solutions.pickle"
    )
    fin = open(pfname, "rb")
    d, dtstime, info, dT, c, dE, dM, mmtinfo, dmatrix, y, all_results = pickle.load(fin)
    fin.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # machine origin
    ax.scatter(0.0, 0.0, 0.0, c="k")

    plot_model3d(ax, all_results[-1][1], 0, 0, 0, 5000, "b-")
    # undeformed for comparison
    plot_model3d(ax, all_results[-1][1], 0, 0, 0, 1, "g-")

    plt.show()


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # machine origin
    ax.scatter(0.0, 0.0, 0.0, c="k")

    params = np.zeros(21)

    # undeformed for comparison
    plot_model3d(ax, params, 100, 100, 100, 1, "g-")

    params[0] = 1.33e-05
    params[3] = -1.12e-05
    params[4] = -5.09e-06
    params[6] = 2.6e-05
    params[7] = 4.6e-06
    params[8] = 3.34e-08
    params[9] = 7.49e-09
    params[10] = 1.54e-08
    params[11] = 5e-09
    params[12] = -4.58e-09
    params[13] = -1.43e-08
    params[14] = 2.19e-08
    params[15] = 2.49e-09
    params[16] = -7.94e-10
    params[17] = 4.78e-08

    plot_model3d(ax, params, 100, 100, 100, 5000, "b-")
    ax.set_axis_off()
    ax.view_init(30, 130)
    plt.show()


if __name__ == "__main__":
    main()
