"""
plot ballplate measurements in 3D
DONE - plot nominal lines
DONE - plot measured errors on magnified scale
- the measured errors need to have a non-zero value at ball 1 for each plate position
  if they are to be visually fitted to a deformation model for the whole machine

"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

import pickle


# position, descriptor, machine axis for X axis plate (chr), machine axis for Y axis plate (chr),
#      column of ballplatedataT that give temeperature for plate X axis, column of ballplatedataT that give temeperature for plate Y axis, column of ballplatedataT that give temeperature for plate Z axis,
# xt, yt, zt (probe offsets as returned by SHOW PRBPIN),
# machine axis for X axis plate (int), machine axis for Y axis plate (int), machine axis for Z axis plate (int), ball multiple for z axis of plate
posinfo = [
    (1, "XY high", "X", "Y", 9, 10, 11, 4.76412, -193.69498, -355.41323, 0, 1, 2, 2),
    (2, "XY low", "X", "Y", 9, 10, 11, 4.76986, -193.68946, -355.41269, 0, 1, 2, 0),
    (3, "YZ back", "Y", "Z", 10, 11, 9, 135.25222, -193.50367, -244.73282, 1, 2, 0, 2),
    (4, "YZ front", "Y", "Z", 10, 11, 9, -132, 0, -243.5, 1, 2, 0, 2),
    (5, "XZ front", "X", "Z", 9, 11, 10, -0.13234, 132.00091, -243.48525, 0, 2, 1, 2),
    (6, "XZ back", "X", "Z", 9, 11, 10, 0, -132, -243.5, 0, 2, 1, 2),
]

params = [
    9.86e-06,
    3.32e-06,
    -1.16e-05,
    0.00e00,
    -3.03e-06,
    0.00e00,
    1.87e-05,
    4.01e-06,
    2.36e-07,
    1.72e-09,
    -1.38e-08,
    -4.00e-09,
    2.66e-09,
    -2.33e-09,
    -3.28e-08,
    -5.40e-09,
    9.19e-09,
    -1.31e-08,
    0.00e00,
    0.00e00,
    0.00e00,
]

lines = []
dataLines = []


# need to make this work over a vector
def model_linear(x, y, z, params, probe=(0, 0, 0)):
    """
     three arrays (n,) of points x,y,z in machine csy
     params (21,) model parameters

     Linear Model Parameters

     x*Txx	             -x*Txy	             -x*Txz
    -y*Tyx	              y*Tyy	             -y*Tyz
    -z*Tzx	            - z*Tzy	              z*Tzz
    -(z + zt) * x*Rxy	 (z + zt) * x*Rxx	 -(y + yt) * x*Rxx
     (y + yt) * x*Rxz	-(x + xt) * x*Rxz	  (x + xt) * x*Rxy
    -(z + zt) * y*Ryy	 (z + zt) * x*Ryx	 -yt * y*Ryx
     yt * y*Ryz	        -xt * y*Ryz           xt * y*Ryy
    -zt * z*Rzy	         zt * z*Rzx	         -yt*z*Rzx
     yt * z*Rzz	        -xt * z*Rzz	          xt * z*Rzy
    -(y + yt) * Wxy 	    -(x +xt) * Wxy	     -(x + xt) * Wxz
    -(z + zt) * Wxz	    -(z + zt) * Wyz	     -(y + yt) * Wyz
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
    xt, yt, zt = probe
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


def plot_magn(magn, eXYZ, nXYZ, pind, ls="-"):
    """
    magn: magnification
    eXYZ: n rows 3 columns error to plot
    nXYZ: n rows 3 columns nominal to plot
    pind: n rows integer from 1 to 6 representing plate position

    plots magn*eXYZ + nXYZ

    """

    for pos in range(1, 7):
        pos_eXYZ = eXYZ[(pind[:] == pos)]
        pos_nXYZ = nXYZ[(pind[:] == pos)]
        pXYZ = magn * pos_eXYZ + pos_nXYZ
        xm, ym, zm = pXYZ.T
        color = colors[pos - 1]
        # ax.scatter(xm, ym, zm, c= color)
        max_y = max_ylist[pos - 1]
        for i in range(0, max_y):
            ind = np.arange(i * 5, (i + 1) * 5)
            xp, yp, zp = pXYZ[ind, :].T
            newline = ax.plot(xp, yp, zp, color + ls)
            lines.append(newline)
            dataLines.append((pos_eXYZ[ind], pos_nXYZ[ind]))
        for i in range(0, 5):
            ind = np.arange(0, 5 * max_y, 5) + i
            xp, yp, zp = pXYZ[ind, :].T
            newline = ax.plot(xp, yp, zp, color + ls)
            lines.append(newline)
            dataLines.append((pos_eXYZ[ind], pos_nXYZ[ind]))
    return lines, dataLines


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# load previuously calculated data
pfname = "./src/cmm_error_map/mpl_2014/data/ballplate_tmp_data_02_n.pickle"
fin = open(pfname, "rb")
(d, dtstime, info, dT, c, dE, dM, bm_linear, mmtinfo) = pickle.load(
    fin, encoding="latin1"
)
fin.close()

# dM (130,9)
# position number, ball number, count, mEX, mEY, mEZ, sEX, sEY, sEZ

# mmtinfo (260,17)
# pos, ball number, plate axis, machine axis, x ball multiple, y ball multiple, z ball multiple,x,y,z,xt,yt,zt,x+xt,y+yt,z+zt,1

magn = 5000
max_ylist = (5, 5, 4, 4, 4, 4)
colors = ("g", "r", "c", "m", "y", "k")

# machine origin
ax.scatter(0.0, 0.0, 0.0, c="k")

# plot nominal lines

for j, ind0 in enumerate((0, 25, 50, 90)):
    max_y = max_ylist[j]
    for i in range(0, max_y):
        ind = np.arange(i * 5, (i + 1) * 5) + ind0
        xp, yp, zp = mmtinfo[ind, 7:10].T
        ax.plot(xp, yp, zp, "b-")
    for i in range(0, 5):
        ind = np.arange(0, 5 * max_y, 5) + i + ind0
        xp, yp, zp = mmtinfo[ind, 7:10].T
        ax.plot(xp, yp, zp, "b-")


# plot actual measurements


magn = 5000
max_ylist = (5, 5, 4, 4, 4, 4)
colors = ("g", "r", "c", "m", "y", "k")

# transfer measured errors from plate axis to machine axis before plotting
# need to add error at ball 1 to each plate mmt
eXYZ = np.zeros((130, 3))
for pos in range(1, 7):
    ind = dM[:, 0] == pos  # row indexes for this position
    colX = posinfo[pos - 1][10]  # machine axis for plate X
    colY = posinfo[pos - 1][11]  # machine axis for plate Y
    eXYZ[ind, colX] = dM[ind, 3]
    eXYZ[ind, colY] = dM[ind, 4]

nXYZ = mmtinfo[:130, 7:10]
pind = dM[:, 0]

lines, dataLines = plot_magn(magn, eXYZ, nXYZ, pind)

# plot nominal deformed by model lines
modEXYZ = np.zeros((130, 3))
for i, mmt in enumerate(mmtinfo[:130]):
    modEXYZ[i, :] = model_linear(
        mmt[7], mmt[8], mmt[9], params, probe=(mmt[10], mmt[11], mmt[12])
    )

modlines, moddataLines = plot_magn(magn, modEXYZ, nXYZ, pind, ls="--")


# if this has been done correctly then the differences between the measured and the model
# over the correct INTERVAL
# should be equal to the residuals found during the fitting process
# only considering the 260 measurements made along plate X and Y

# calculate model error over each plate interval
# subtract ball 1 row from each position
modEXYZ0 = np.zeros((130, 3))
for ri, row in enumerate(modEXYZ):
    if mmtinfo[ri, 1] == 1:
        row1 = row
    modEXYZ0[ri, :] = row - row1

residuals = eXYZ - modEXYZ0  # 130 * 3 = 390 measurements
# discard plate Z measurements
res_plateXY = np.zeros((130, 2))
for pos in range(1, 7):
    ind = dM[:, 0] == pos  # row indexes for this position
    colX = posinfo[pos - 1][10]  # machine axis for plate X
    colY = posinfo[pos - 1][11]  # machine axis for plate Y

    res_plateXY[ind, 0] = residuals[ind, colX]
    res_plateXY[ind, 1] = residuals[ind, colY]

rs = res_plateXY.flatten("F")
df_resid = 244
rmse_res = (np.dot(rs, rs) / df_resid) ** 0.5
print("root mean square residuals", rmse_res)


# add slider to change magnification


axmagn = plt.axes([0.25, 0.05, 0.65, 0.03])
smagn = Slider(axmagn, "Magn.", 1, 10.0, valinit=magn / 1000)


def update(val):
    magn = smagn.val * 1000
    for line, data in zip(lines, dataLines):
        xp, yp, zp = (magn * data[0] + data[1]).T
        line[0].set_data((xp, yp))
        line[0].set_3d_properties(zp)
    plt.draw()


smagn.on_changed(update)
print

plt.show()
