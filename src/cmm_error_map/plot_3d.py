"""
standalaone pyqtgraph app to plot the 3d deformation of a CMM volume
for a linear model of the 21 error parameters
"""

import numpy as np

import pyqtgraph as pg
import pyqtgraph.opengl as gl

import cmm_error_map.design_matrix_linear_fixed as design


def plot_model3d(w: gl.GLViewWidget, params, xt, yt, zt, mag, col="white"):
    """
    takes a set of model parameters and produces a 3D magniifed plot of machine
    deformation
    """
    XYZ, eXYZ = design.machine_deformation(params, xt, yt, zt)
    pXYZ_3D = XYZ + mag * eXYZ

    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")

    nx, ny, nz = XYZ.shape[:3]

    nlines = 0

    # lines parallel to x axis
    for j in range(ny):
        for k in range(nz):
            pts = pXYZ_3D[:, j, k, :]
            plt = gl.GLLinePlotItem(pos=pts, color=pg.mkColor(col), antialias=True)
            w.addItem(plt)
            nlines += 1

    # lines parallel to y axis
    for i in range(nx):
        for k in range(nz):
            pts = pXYZ_3D[i, :, k, :]
            plt = gl.GLLinePlotItem(pos=pts, color=pg.mkColor(col), antialias=True)
            w.addItem(plt)
            nlines += 1

    # lines parallel to z axis
    for i in range(nx):
        for j in range(ny):
            pts = pXYZ_3D[i, j, :, :]
            plt = gl.GLLinePlotItem(pos=pts, color=pg.mkColor(col), antialias=True)
            w.addItem(plt)
            nlines += 1

    plt.width = 4
    return XYZ, eXYZ


app = pg.mkQApp("CMM 3D Deformation")
w = gl.GLViewWidget()
w.show()
w.setWindowTitle("CMM 3D Deformation")


# origin

org = gl.GLScatterPlotItem(
    pos=(0, 0, 0), size=20, color=pg.mkColor("white"), pxMode=False
)
w.addItem(org)


w.setCameraPosition(distance=2000)


params0 = np.zeros(21)
params = np.zeros(21)
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

# undeformed for comparison
plot_model3d(w, params0, 100, 100, 100, 1, col="green")
plot_model3d(w, params, 100, 100, 100, 5000, col="blue")

# axis
axis = gl.GLAxisItem()
axis.setSize(x=1000, y=700, z=700)
# setattr(axis, "width", 20)

w.addItem(axis)


xlabel = gl.GLTextItem(pos=(1000.0, 0.0, 0.0), text="X")
w.addItem(xlabel)
ylabel = gl.GLTextItem(pos=(0.0, 700.0, 0.0), text="Y")
w.addItem(ylabel)
zlabel = gl.GLTextItem(pos=(0.0, 0.0, 700.0), text="Z")
w.addItem(zlabel)

if __name__ == "__main__":
    pg.exec()
