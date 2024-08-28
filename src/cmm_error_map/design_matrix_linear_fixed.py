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
import numpy as np

model_parameters_dict = {
    "Txx": 0.0,
    "Txy": 0.0,
    "Txz": 0.0,
    "Tyx": 0.0,
    "Tyy": 0.0,
    "Tyz": 0.0,
    "Tzx": 0.0,
    "Tzy": 0.0,
    "Tzz": 0.0,
    "Rxx": 0.0,
    "Rxy": 0.0,
    "Rxz": 0.0,
    "Ryx": 0.0,
    "Ryy": 0.0,
    "Ryz": 0.0,
    "Rzx": 0.0,
    "Rzy": 0.0,
    "Rzz": 0.0,
    "Wxy": 0.0,
    "Wxz": 0.0,
    "Wyz": 0.0,
}

# some non-zero parmeters for quick tests
model_parameters_test = {
    "Txx": 1.33e-05,
    "Txy": 0.0,
    "Txz": 0.0,
    "Tyx": -1.12e-05,
    "Tyy": -5.09e-06,
    "Tyz": 0.0,
    "Tzx": 2.6e-05,
    "Tzy": 4.6e-06,
    "Tzz": 3.34e-08,
    "Rxx": 7.49e-09,
    "Rxy": 1.54e-08,
    "Rxz": 5e-09,
    "Ryx": -4.58e-09,
    "Ryy": -1.43e-08,
    "Ryz": 2.19e-08,
    "Rzx": 2.49e-09,
    "Rzy": -7.94e-10,
    "Rzz": 4.78e-08,
    "Wxy": 0.0,
    "Wxz": 0.0,
    "Wyz": 0.0,
}

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

# descriptor, machine axis on which term is dependent, number of terms, axes (list) in which term appears, coefficients for term (list expressed as column of mmtinfo matrix), sign of coefficients
modelparameters = [
    ("Txx", 0, 5, (0,), (16,), (1,)),
    ("Txy", 0, 5, (1,), (16,), (-1,)),
    ("Txz", 0, 5, (2,), (16,), (-1,)),
    ("Tyx", 1, 5, (0,), (16,), (-1,)),
    ("Tyy", 1, 5, (1,), (16,), (1,)),
    ("Tyz", 1, 5, (2,), (16,), (-1,)),
    ("Tzx", 2, 4, (0,), (16,), (-1,)),
    ("Tzy", 2, 4, (1,), (16,), (-1,)),
    ("Tzz", 2, 4, (2,), (16,), (1,)),
    ("Rxx", 0, 5, (1, 2), (15, 14), (1, -1)),
    ("Rxy", 0, 5, (0, 2), (15, 13), (-1, 1)),
    ("Rxz", 0, 5, (0, 1), (14, 13), (1, -1)),
    ("Ryx", 1, 5, (1, 2), (15, 11), (1, -1)),
    ("Ryy", 1, 5, (0, 2), (15, 10), (-1, 1)),
    ("Ryz", 1, 5, (0, 1), (11, 10), (1, -1)),
    ("Rzx", 2, 4, (1, 2), (12, 11), (1, -1)),
    ("Rzy", 2, 4, (0, 2), (12, 10), (-1, 1)),
    ("Rzz", 2, 4, (0, 1), (11, 10), (1, -1)),
    ("Wxy", -1, 1, (0, 1), (14, 13), (-1, -1)),
    ("Wxz", -1, 1, (0, 2), (15, 13), (-1, -1)),
    ("Wyz", -1, 1, (1, 2), (15, 14), (-1, -1)),
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


def modelled_mmts(mmtinfo, params):
    """
    Takes mmtinfo and a set of 21 parameters and produces the expected ballplate mmts
    """
    eXYZ = np.zeros((260, 3))

    for i, mmt in enumerate(mmtinfo):
        eXYZ[i, :] = model_linear(
            mmt[7], mmt[8], mmt[9], params, mmt[10], mmt[11], mmt[12]
        )

    # subtract ball 1 row from each position and select which axis corresponds to measurement
    y = np.zeros((260))
    for ri, row in enumerate(eXYZ):
        if mmtinfo[ri, 1] == 1:
            row1 = row
        y[ri] = (row - row1)[int(mmtinfo[ri, 3])]

    return y


def modelled_mmts_XYZ(RP, xt, yt, zt, params, ballspacing=133.0, nballs=(5, 5)):
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

    """
    ball_count = nballs[0] * nballs[1]
    ballnumber = np.arange(ball_count)

    xp = (ballnumber % nballs[0]) * ballspacing
    yp = (ballnumber // nballs[1]) * ballspacing
    zp = ballnumber * 0.0

    XP = np.vstack((xp, yp, zp, np.ones(ball_count)))
    # transfer to machine CSY
    XM = np.dot(RP, XP)
    eXYZ = np.zeros((4, ball_count))
    for b in ballnumber:
        eXYZ[:3, b] = model_linear(XM[0, b], XM[1, b], XM[2, b], params, xt, yt, zt)

    # add error to nominal
    XYZm = XM + eXYZ

    # transfer to plate CSY
    XYZp = np.dot(np.linalg.inv(RP), XYZm)
    # subtract ball 1 row from each position
    XYZp[:3, :] = XYZp[:3, :] - XYZp[:3, 0:1]
    # also rotate about plate z so y=0 for ball 5
    ang = -1.0 * np.arctan2(XYZp[1, nballs[0] - 1], XYZp[0, nballs[1] - 1])
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
    dXY = XYZp[:2, :] - np.vstack((xp, yp))
    return dXY.T


def machine_deformation(params, xt, yt, zt, spacing=200):
    """
    takes a set of 21 params and generates an evenly spaced grid over machine volume
    returns nominal and error as 3D arrays
    """

    nx = 800 // spacing + 1
    ny = 600 // spacing + 1
    nz = 600 // spacing + 1
    npoints = nx * ny * nz

    # generate set of x,y,z values and error values over space of machine
    XYZ = np.zeros((npoints, 3))
    eXYZ = np.zeros((npoints, 3))
    i = 0
    for x in np.arange(0, 801, spacing):
        for y in np.arange(0, 601, spacing):
            for z in np.arange(0, 601, spacing):
                XYZ[i, :] = x, y, z
                eXYZ[i, :] = model_linear(x, y, z, params, xt, yt, zt)
                i = i + 1

    XYZ_3D = np.reshape(XYZ, (nx, ny, nz, 3))
    eXYZ_3D = np.reshape(eXYZ, (nx, ny, nz, 3))
    return XYZ_3D, eXYZ_3D


def design_matrix_linear(mmtinfo):
    """
    returns design matrix implementing linear model
    uses model_linear function
    To calculate each column the corresponding parmaeter is set to 1.0
    and all other parameters to zero.
    """
    d2 = np.zeros((260, 21))

    for i in range(130):
        for j in range(21):
            params = np.zeros(21)
            params[j] = 1.0
            x, y, z = mmtinfo[i, 7:10].T
            xt, yt, zt = mmtinfo[i, 10:13].T
            xyzs = model_linear(x, y, z, params, xt, yt, zt)
            pos = mmtinfo[i, 0]
            d2[i, j] = xyzs[int(posinfo[int(pos - 1)][10])]
            d2[i + 130, j] = xyzs[int(posinfo[int(pos - 1)][11])]

    # subtract ball 1 row from each position
    d3 = np.zeros((260, 21))
    for ri, row in enumerate(d2):
        if mmtinfo[ri, 1] == 1:
            row1 = row
        d3[ri, :] = row - row1
    return d3


def designmatrix_linear_mp(mmtinfo):
    """
    returns design matrix implementing linear model
    uses model parameters data structure
    design_matrix_linear is the prefered method but this one is here for
    checking purposes
    """
    dmatrix = np.zeros((260, 21))
    ci = 0
    for param in modelparameters:
        for ri, mmt in enumerate(mmtinfo):
            machineaxis = mmt[3]
            if machineaxis in param[3]:
                n = param[3].index(machineaxis)
                sign = param[5][n]
                coeff = mmt[param[4][n]]
                if param[1] == -1:
                    # W parameter are independent of distance
                    distance = 1
                else:
                    # T, R parameters are dependent on distance
                    distance = mmt[param[1] + 7]
                dmatrix[ri, ci] = sign * coeff * distance
        ci = ci + 1  # increment column index

    # subtract ball 1 row from each position
    dmatrix2 = np.zeros((260, 21))
    for ri, row in enumerate(dmatrix):
        if mmtinfo[ri, 1] == 1:
            row1 = row
        dmatrix2[ri, :] = row - row1

    return dmatrix2


def measurement_vector(meandata):
    """
    form vector of results b for solving matrix equation A.x = b for vector of
    parameters x given design matrix A.
    """
    b = np.zeros((260))
    b[:130] = meandata[:, 3]
    b[130:] = meandata[:, 4]
    return b
