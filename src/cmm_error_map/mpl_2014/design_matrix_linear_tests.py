"""
just the bits of the original 2014 design_matrix_linear.py
required for regression testing 2025 code.
"""

# -------------------------------------------------------------------------------
# Name:        design_matrix_linear.py
# Purpose:     Functions to output linear design matrix for recovering CMM errors
#                from ballplate measurements.
#
# Author:      e.howick
#
# Created:     22/04/2013
# Copyright:   (c) e.howick 2013
# Licence:     <your licence>
# -------------------------------------------------------------------------------
import pickle

import numpy as np

ballspacing = 133.0
posinfo = [
    (1, "XY high", "X", "Y", 9, 10, 11, 4.76412, -193.69498, -355.41323, 0, 1, 2, 2),
    (2, "XY low", "X", "Y", 9, 10, 11, 4.76986, -193.68946, -355.41269, 0, 1, 2, 0),
    (3, "YZ back", "Y", "Z", 10, 11, 9, 135.25222, -193.50367, -244.73282, 1, 2, 0, 2),
    (4, "YZ front", "Y", "Z", 10, 11, 9, -132, 0, -243.5, 1, 2, 0, 2),
    (5, "XZ front", "X", "Z", 9, 11, 10, -0.13234, 132.00091, -243.48525, 0, 2, 1, 2),
    (6, "XZ back", "X", "Z", 9, 11, 10, 0, -132, -243.5, 0, 2, 1, 2),
    (7, "XY long", "X", "Y", 9, 10, 11, 4.76412, -193.69498, -355.41323, 0, 1, 2, 2),
    (8, "XY short", "X", "Y", 9, 10, 11, 4.76412, -193.69498, -355.41323, 0, 1, 2, 2),
]

"""
    Linear Model Parameters

    x*Txx	             -x*Txy	             -x*Txz
   -y*Tyx	              y*Tyy	             -y*Tyz
   -z*Tzx	            - z*Tzy	              z*Tzz
   -(z + zt) * x*Rxy	 (z + zt) * x*Rxx	 -(y + yt) * x*Rxx
    (y + yt) * x*Rxz	-(x + xt) * x*Rxz	  (x + xt) * x*Rxy
   -(z + zt) * y*Ryy	 (z + zt) * y*Ryx	 -yt * y*Ryx
    yt * y*Ryz	        -xt * y*Ryz           xt * y*Ryy
   -zt * z*Rzy	         zt * z*Rzx	         -yt*z*Rzx
    yt * z*Rzz	        -xt * z*Rzz	          xt * z*Rzy
   -(y + yt) * Wxy 	    -(x +xt) * Wxy	     -(x + xt) * Wxz
   -(z + zt) * Wxz	    -(z + zt) * Wyz	     -(y + yt) * Wyz

Each Txx, Tyx etc. term is a constant (slope)
"""


def model_linear(x, y, z, params, xt, yt, zt):
    """
    three arrays (n,) of points x,y,z in machine csy
    params (21,) model parameters
    """
    (
        TxxL,
        TxyL,
        TxzL,
        TyxL,
        TyyL,
        TyzL,
        TzxL,
        TzyL,
        TzzL,
        RxxL,
        RxyL,
        RxzL,
        RyxL,
        RyyL,
        RyzL,
        RzxL,
        RzyL,
        RzzL,
        Wxy,
        Wxz,
        Wyz,
    ) = params

    xE = (
        x * TxxL
        - y * TyxL
        - z * TzxL
        - (z + zt) * x * RxyL
        + (y + yt) * x * RxzL
        - (z + zt) * y * RyyL
        + yt * y * RyzL
        - zt * z * RzyL
        + yt * z * RzzL
        - (y + yt) * Wxy
        - (z + zt) * Wxz
    )

    yE = (
        -x * TxyL
        + y * TyyL
        - z * TzyL
        + (z + zt) * x * RxxL
        - (x + xt) * x * RxzL
        + (z + zt) * y * RyxL
        - xt * y * RyzL
        + zt * z * RzxL
        - xt * z * RzzL
        - (x + xt) * Wxy
        - (z + zt) * Wyz
    )

    zE = (
        -x * TxzL
        - y * TyzL
        + z * TzzL
        - (y + yt) * x * RxxL
        + (x + xt) * x * RxyL
        - yt * y * RyxL
        + xt * y * RyyL
        - yt * z * RzxL
        + xt * z * RzyL
        - (x + xt) * Wxz
        - (y + yt) * Wyz
    )

    return xE, yE, zE


def modelled_mmts_XYZ(RP, xt, yt, zt, params, verbose=False):
    """
     for a plate transformation matrix RP,
     a probe xt,yt,zt and a set of 21 parameters, produces an expected set of
     ball plate measuremets

     RP = [[x5,x20,xn,x0],
           [y5,y20,yn,y0],
           [z5,z20,zn,z0],
           [0,0,0, 1]]

    where (x5,y5,z5) is the direction of the vector from ball 1 to ball 5 (plate X axis)
          (x20,y20,z20) is the direction of the vector from ball 1 to ball 20 (plate Y axis)
          (xn,yn,zn) is the direction perpendicular to the plate (plate Z axis)
          (x0,y0,z0) is the machine position of ball 1

    verbose = True
    returns dXY.T, XYZm, XYZp
    """
    ballnumber = np.arange(25)

    xp = (ballnumber % 5) * ballspacing
    yp = (ballnumber // 5) * ballspacing
    zp = ballnumber * 0.0

    XP = np.vstack((xp, yp, zp, np.ones(25)))
    # transfer to machine CSY
    XM = np.dot(RP, XP)
    eXYZ = np.zeros((4, 25))
    for b in ballnumber:
        eXYZ[:3, b] = model_linear(XM[0, b], XM[1, b], XM[2, b], params, xt, yt, zt)

    # add error to nominal
    XYZm = XM + eXYZ

    # transfer to plate CSY
    XYZp = np.dot(np.linalg.inv(RP), XYZm)
    # subtract ball 1 row from each position
    XYZp[:3, :] = XYZp[:3, :] - XYZp[:3, 0:1]
    # also rotate about plate z so y=0 for ball 5
    ang = -1.0 * np.arctan2(XYZp[1, 4], XYZp[0, 4])
    cang = np.cos(ang)
    sang = np.sin(ang)
    RZ = np.array(
        [
            [cang, -sang, 0.0, 0.0],
            [sang, cang, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    XYZp = np.dot(RZ, XYZp)
    plate_nom = np.vstack((xp, yp))
    dXY = XYZp[:2, :] - plate_nom
    if verbose:
        return dXY.T, eXYZ, XYZp, XM, plate_nom
    else:
        return dXY.T
