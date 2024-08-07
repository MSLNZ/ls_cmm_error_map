import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import pickle


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# DONE set up 3D grid of points
# DONE plot it as points
# DONE plot it as grids
# DONE deform it by model matrix
# DONE magnify deformation
# DONE plot it

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

    xs = (
        x
        + xt
        + x * TxxL
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

    ys = (
        y
        + yt
        - x * TxyL
        + y * TyyL
        - z * TzyL
        + (z + zt) * x * RxxL
        - (x + xt) * x * RxzL
        + (z + zt) * x * RyxL
        - xt * y * RyzL
        + zt * z * RzxL
        - xt * z * RzzL
        - (x + xt) * Wxy
        - (z + zt) * Wyz
    )

    zs = (
        z
        + zt
        - x * TxzL
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

    return xs, ys, zs


offsetpos1ball1 = (158.86, 38.62, 150.0)  # Z???placeholder value !!!!
ballspacing = 133.0

# scatter points as x, y, z
ngrid = 5
pind = np.arange(0, ngrid**3)
xs = (pind % ngrid) * ballspacing + offsetpos1ball1[0]
ys = (pind // ngrid % ngrid) * ballspacing + offsetpos1ball1[1]
zs = (pind // (ngrid) ** 2) * ballspacing + offsetpos1ball1[2]

# ax.scatter(xs, ys, zs)

# plot each grid line
##arind = pind.reshape((ngrid,ngrid,ngrid))
##for i in range(ngrid):
##    for j in range(ngrid):
##        indx = arind[i,j,0:5]
##        ax.plot(xs[indx],ys[indx],zs[indx], 'b-')
##        indy = arind[i,0:5,j]
##        ax.plot(xs[indy],ys[indy],zs[indy], 'b-')
##        indz = arind[0:5,i,j]
##        ax.plot(xs[indz],ys[indz],zs[indz], 'b-')
##
##xd,yd,zd = model_linear(xs,ys,zs,params)
##
##magn = 5000
##xd = magn*(xd - xs) + xd
##yd = magn*(yd - ys) + yd
##zd = magn*(zd - zs) + zd
##
##for i in range(ngrid):
##    for j in range(ngrid):
##        indx = arind[i,j,0:5]
##        ax.plot(xd[indx],yd[indx],zd[indx], 'r-')
##        indy = arind[i,0:5,j]
##        ax.plot(xd[indy],yd[indy],zd[indy], 'r-')
##        indz = arind[0:5,i,j]
##        ax.plot(xd[indz],yd[indz],zd[indz], 'r-')

ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.set_zlabel("Z Label")

# plot just measured points
pfname = "./src/cmm_error_map/mpl_2014/data/ballplate_tmp_data_02_n.pickle"

# load previuously calculated data
pfname = "./src/cmm_error_map/mpl_2014/data/ballplate_tmp_data_02_n.pickle"
fin = open(pfname, "rb")
(d, dtstime, info, dT, c, dE, dM, bm_linear, mmtinfo) = pickle.load(
    fin, encoding="latin1"
)
fin.close()
# mmtinfo
# pos, ball number, plate axis, machine axis, x ball multiple, y ball multiple, z ball multiple,x,y,z,xt,yt,zt,x+xt,y+yt,z+zt,1
xm, ym, zm = mmtinfo[:, 7:10].T
ax.scatter(xm, ym, zm, c="g")


plt.show()
